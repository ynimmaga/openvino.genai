// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Text generation over a ggml backend, using GenAI's own sampling stack.
//
// The shape of this file is deliberately the same as the stateful pipeline's generation loop
// (lm_encoding.cpp): schedule tokens, run the model, hand the logits to Sampler, stream, repeat.
// Only the execution step differs -- ov::InferRequest is replaced by ov::ggml_emitter::GgmlModel.
// Everything above it is shared GenAI code, which is the entire point of putting this inside the
// library rather than in an application.

#include "openvino/genai/ggml_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "openvino/ggml_emitter/emitter.hpp"

#include "openvino/genai/text_streamer.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

// fp16 -INF, written directly: the mask is f16 and building an ov::Tensor to convert one
// constant would be gratuitous.
constexpr uint16_t FP16_NEG_INF = 0xFC00;

std::shared_ptr<StreamerBase> resolve_streamer(StreamerVariant v, const Tokenizer& tok) {
    if (auto* cb = std::get_if<std::function<StreamingStatus(std::string)>>(&v)) {
        return std::make_shared<TextStreamer>(tok, *cb);
    }
    if (auto* ptr = std::get_if<std::shared_ptr<StreamerBase>>(&v)) {
        return *ptr;
    }
    return nullptr;
}

}  // namespace

class GgmlPipeline::Impl {
public:
    Impl(std::shared_ptr<ov::ggml_emitter::GgmlModel> model, Tokenizer tokenizer)
        : m_model(std::move(model)),
          m_tokenizer(std::move(tokenizer)),
          m_sampler(m_tokenizer),
          m_logits(m_model->logits_size()) {
        m_n_kv = m_model->context_size();
        OPENVINO_ASSERT(m_n_kv > 0, "[GGML] model did not report a context size");
        m_config.eos_token_id = m_tokenizer.get_eos_token_id();
    }

    /// One forward pass at absolute position `pos`, writing K/V into cache slot `pos`.
    void forward(int64_t token, int32_t pos) {
        const int32_t tok32 = static_cast<int32_t>(token);
        const int32_t out_id = 0;
        const int64_t slot = pos;
        m_model->write_input("inp_tokens", &tok32, sizeof(tok32));
        m_model->write_input("inp_pos", &pos, sizeof(pos));
        m_model->write_input("inp_out_ids", &out_id, sizeof(out_id));
        // There can be more than one KV row-index input: a dumped llama.cpp graph carries a
        // separate one per cache, while the OpenVINO builder emits a single shared one.
        for (const auto& name : m_model->input_names()) {
            if (name.rfind("inp_kv_idx", 0) == 0) {
                m_model->write_input(name, &slot, sizeof(slot));
            }
        }
        // Causal mask for the single query token: it attends to slots 0..pos and nothing beyond.
        for (const auto& name : m_model->input_names()) {
            if (name.rfind("self_kq_mask", 0) != 0 && name.rfind("attn_inp_kq_mask", 0) != 0) {
                continue;
            }
            std::vector<uint16_t> mask(m_model->input_size(name), FP16_NEG_INF);
            for (int j = 0; j <= pos && j < static_cast<int>(m_n_kv); j++) {
                mask[j] = 0;
            }
            m_model->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
        OPENVINO_ASSERT(m_model->compute(), "[GGML] compute failed at position ", pos);
        m_model->read_logits(m_logits.data());
    }

    EncodedResults generate(const std::vector<int64_t>& input_ids,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) {
        GenerationConfig config = generation_config.value_or(m_config);
        if (config.eos_token_id == -1) {
            config.set_eos_token_id(m_config.eos_token_id);
        }
        config.validate();
        OPENVINO_ASSERT(!config.is_beam_search(),
                        "[GGML] beam search needs KV cache reordering across beams, which the "
                        "ggml backend does not implement; use greedy or multinomial sampling");
        OPENVINO_ASSERT(config.num_return_sequences == 1,
                        "[GGML] num_return_sequences > 1 requires sequence forking, which needs "
                        "KV cache copy-on-write; not implemented for the ggml backend");

        const size_t budget = m_chat_kv_used + input_ids.size() + config.max_new_tokens;
        OPENVINO_ASSERT(budget <= m_n_kv,
                        "[GGML] prompt + max_new_tokens (", budget, ") exceeds the context this "
                        "graph was built for (", m_n_kv, "). The graph is shape-static, so this "
                        "cannot grow at runtime -- rebuild with a larger n_kv.");

        auto streamer_ptr = resolve_streamer(std::move(streamer), m_tokenizer);
        auto group = std::make_shared<SequenceGroup>(0, input_ids, config);

        // Prefill. The graph is single-token, so the prompt is fed one token per pass; the
        // causal mask makes this equivalent to a batched prefill, only slower.
        int32_t pos = static_cast<int32_t>(m_chat_kv_used);
        for (size_t i = 0; i < input_ids.size(); i++) {
            forward(input_ids[i], pos++);
        }
        group->schedule_tokens(group->get_prompt_len());
        group->set_output_seq_len(1);

        std::vector<SequenceGroup::Ptr> groups{group};
        ov::Tensor logits_tensor(ov::element::f32, ov::Shape{1, 1, m_logits.size()},
                                 m_logits.data());
        m_sampler.sample(groups, logits_tensor);

        size_t streamed = 0;
        auto stream_new_tokens = [&]() -> bool {
            if (!streamer_ptr) {
                return false;
            }
            const auto running = group->get_running_sequences();
            if (running.empty()) {
                return false;
            }
            const auto& gen = running.front()->get_generated_ids();
            while (streamed < gen.size()) {
                if (streamer_ptr->write(gen[streamed++]) != StreamingStatus::RUNNING) {
                    return true;  // cancelled by the caller
                }
            }
            return false;
        };

        bool cancelled = stream_new_tokens();

        // Generation. Feed back the token the sampler chose, exactly as the stateful loop does.
        while (!cancelled && !group->has_finished() && pos < static_cast<int32_t>(m_n_kv)) {
            const auto running = group->get_running_sequences();
            if (running.empty()) {
                break;
            }
            const auto& gen = running.front()->get_generated_ids();
            if (gen.empty()) {
                break;
            }
            forward(gen.back(), pos++);
            group->schedule_tokens(1);
            m_sampler.sample(groups, logits_tensor);
            cancelled = stream_new_tokens();
        }
        if (streamer_ptr) {
            streamer_ptr->end();
        }

        EncodedResults results;
        // A sequence that hit EOS or a stop condition moves to the finished list; one stopped by
        // the KV budget or a cancelling streamer is still running. Take whichever exists.
        const auto finished = group->get_finished_sequences();
        if (!finished.empty()) {
            results.tokens.push_back(finished.front()->get_generated_ids());
            results.scores.push_back(finished.front()->get_cumulative_log_prob());
        } else {
            const auto running = group->get_running_sequences();
            OPENVINO_ASSERT(!running.empty(), "[GGML] generation produced no sequence");
            results.tokens.push_back(running.front()->get_generated_ids());
            results.scores.push_back(running.front()->get_cumulative_log_prob());
        }

        m_sampler.clear_request_info(group->get_request_id());
        if (m_in_chat) {
            m_chat_kv_used = static_cast<size_t>(pos);
        }
        return results;
    }

    DecodedResults generate(const std::string& prompt,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) {
        std::string text = prompt;
        if (m_in_chat) {
            m_history.push_back({{"role", "user"}, {"content", prompt}});
            text = m_tokenizer.apply_chat_template(m_history, true);
        }
        const auto ids = m_tokenizer.encode(text, ov::genai::add_special_tokens(!m_in_chat));
        const auto* p = ids.input_ids.data<int64_t>();
        std::vector<int64_t> input_ids(p, p + ids.input_ids.get_size());

        auto encoded = generate(input_ids, generation_config, std::move(streamer));
        DecodedResults decoded;
        for (const auto& toks : encoded.tokens) {
            decoded.texts.push_back(m_tokenizer.decode(toks));
        }
        decoded.scores = encoded.scores;
        if (m_in_chat && !decoded.texts.empty()) {
            m_history.push_back({{"role", "assistant"}, {"content", decoded.texts.front()}});
        }
        return decoded;
    }

    void start_chat(const std::string& system_message) {
        m_in_chat = true;
        m_chat_kv_used = 0;
        m_history.clear();
        if (!system_message.empty()) {
            m_history.push_back({{"role", "system"}, {"content", system_message}});
        }
    }

    void finish_chat() {
        m_in_chat = false;
        m_chat_kv_used = 0;
        m_history.clear();
    }

    GenerationConfig m_config;
    Tokenizer m_tokenizer;
    std::shared_ptr<ov::ggml_emitter::GgmlModel> m_model;

private:
    Sampler m_sampler;
    std::vector<float> m_logits;
    size_t m_n_kv = 0;
    // Chat mode keeps the KV cache across turns, so the next prompt starts after what is
    // already written rather than at slot 0.
    bool m_in_chat = false;
    size_t m_chat_kv_used = 0;
    ChatHistory m_history;
};

GgmlPipeline::GgmlPipeline(const std::filesystem::path& models_path,
                           size_t n_kv,
                           const std::string& backend)
    : m_impl(std::make_unique<Impl>(
          ov::ggml_emitter::GgmlModel::build(models_path.string(), static_cast<int>(n_kv), backend),
          Tokenizer(models_path))) {}

GgmlPipeline::GgmlPipeline(const std::filesystem::path& cgraph_path,
                           const std::filesystem::path& models_path,
                           const std::string& backend)
    : m_impl(std::make_unique<Impl>(
          ov::ggml_emitter::GgmlModel::from_cgraph(cgraph_path.string(), models_path.string(),
                                                   backend),
          Tokenizer(models_path))) {}

GgmlPipeline::~GgmlPipeline() = default;

DecodedResults GgmlPipeline::generate(const std::string& prompt,
                                      OptionalGenerationConfig generation_config,
                                      StreamerVariant streamer) {
    return m_impl->generate(prompt, generation_config, std::move(streamer));
}

EncodedResults GgmlPipeline::generate(const std::vector<int64_t>& input_ids,
                                      OptionalGenerationConfig generation_config,
                                      StreamerVariant streamer) {
    return m_impl->generate(input_ids, generation_config, std::move(streamer));
}

void GgmlPipeline::start_chat(const std::string& system_message) {
    m_impl->start_chat(system_message);
}

void GgmlPipeline::finish_chat() {
    m_impl->finish_chat();
}

GenerationConfig GgmlPipeline::get_generation_config() const {
    return m_impl->m_config;
}

void GgmlPipeline::set_generation_config(const GenerationConfig& config) {
    m_impl->m_config = config;
    m_impl->m_config.validate();
}

Tokenizer GgmlPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

size_t GgmlPipeline::context_size() const {
    return m_impl->m_model->context_size();
}

std::string GgmlPipeline::backend_name() const {
    return m_impl->m_model->backend_name();
}

}  // namespace genai
}  // namespace ov
