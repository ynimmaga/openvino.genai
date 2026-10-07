// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Text generation over a ggml backend, using GenAI's own sampling stack.
//
// The shape of this file is deliberately the same as the stateful pipeline's generation loop
// (lm_encoding.cpp): schedule tokens, run the model, hand the logits to Sampler, stream, repeat.
// Only the execution step differs -- ov::InferRequest is replaced by ov::ggml_cgraph_loader::GgmlModel.
// Everything above it is shared GenAI code, which is the entire point of putting this inside the
// library rather than in an application.

#include "openvino/genai/ggml_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"

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
    Impl(std::map<size_t, std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>> buckets, Tokenizer tokenizer)
        : m_buckets(std::move(buckets)), m_tokenizer(std::move(tokenizer)), m_sampler(m_tokenizer) {
        OPENVINO_ASSERT(m_buckets.count(1), "[GGML] a bucket size 1 is required (decode is always "
                                           "single-token)");
        m_model = m_buckets.at(1);
        m_n_kv = m_model->context_size();
        OPENVINO_ASSERT(m_n_kv > 0, "[GGML] model did not report a context size");
        m_logits.resize(m_model->logits_size());
        for (const auto& kv : m_buckets) {
            OPENVINO_ASSERT(kv.second->context_size() == m_n_kv &&
                                kv.second->logits_size() == m_logits.size(),
                            "[GGML] all prefill buckets must share one context size and vocabulary");
        }
        m_config.eos_token_id = m_tokenizer.get_eos_token_id();
        // forward_batch() routes padding rows to the top slots (n_kv-1, n_kv-2, ...); keep real
        // tokens below them so padding never overwrites live K/V.
        m_scratch_slots = m_buckets.rbegin()->first - 1;
        OPENVINO_ASSERT(m_scratch_slots < m_n_kv, "[GGML] largest prefill bucket exceeds the context size");
    }

    // Each bucket owns separate KV cache tensors, so a step on one doesn't update the others.
    // Only the model about to run needs the current state, and only if a different bucket ran
    // last (decode stays on bucket 1 call after call, so this is a no-op almost every step).
    void sync_before(const std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>& next) {
        if (m_active && m_active != next) {
            for (const auto& name : m_active->input_names()) {
                if (name.rfind("cache_", 0) != 0) {
                    continue;
                }
                const size_t nbytes = m_active->input_nbytes(name);
                if (m_cache_buf.size() < nbytes) {
                    m_cache_buf.resize(nbytes);
                }
                if (m_active->read_input(name, m_cache_buf.data(), nbytes)) {
                    next->write_input(name, m_cache_buf.data(), nbytes);
                }
            }
        }
        m_active = next;
    }

    /// One forward pass at absolute position `pos`, writing K/V into cache slot `pos`.
    void forward(int64_t token, int32_t pos) {
        sync_before(m_model);
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

    // Smallest bucket that fits `remaining`, else the largest available (consumed with no
    // padding, looping again for the rest).
    size_t pick_bucket(size_t remaining) const {
        auto it = m_buckets.lower_bound(remaining);
        return it != m_buckets.end() ? it->first : m_buckets.rbegin()->first;
    }

    // One batched forward pass: `chunk` real tokens (ids[0..chunk)) at base_pos, base_pos+1, ...,
    // padded up to `bucket` with a dummy token routed to a reserved scratch KV slot so it can
    // never collide with a real future position. Real rows only ever unmask their own causal
    // prefix, so padding rows are already outside every real row's attention.
    void forward_batch(const int64_t* ids, size_t chunk, size_t bucket, int32_t base_pos) {
        auto& model = m_buckets.at(bucket);
        sync_before(model);
        std::vector<int32_t> tok_buf(bucket, 0), pos_buf(bucket, 0);
        std::vector<int64_t> slot_buf(bucket, 0);
        for (size_t i = 0; i < chunk; i++) {
            tok_buf[i] = static_cast<int32_t>(ids[i]);
            pos_buf[i] = base_pos + static_cast<int32_t>(i);
            slot_buf[i] = base_pos + static_cast<int64_t>(i);
        }
        for (size_t i = chunk; i < bucket; i++) {
            slot_buf[i] = static_cast<int64_t>(m_n_kv) - 1 - static_cast<int64_t>(i - chunk);
        }
        model->write_input("inp_tokens", tok_buf.data(), tok_buf.size() * sizeof(int32_t));
        model->write_input("inp_pos", pos_buf.data(), pos_buf.size() * sizeof(int32_t));
        for (const auto& name : model->input_names()) {
            if (name.rfind("inp_kv_idx", 0) == 0) {
                model->write_input(name, slot_buf.data(), slot_buf.size() * sizeof(int64_t));
            }
        }
        const int32_t out_id = static_cast<int32_t>(chunk) - 1;  // last real row
        model->write_input("inp_out_ids", &out_id, sizeof(out_id));

        for (const auto& name : model->input_names()) {
            if (name.rfind("self_kq_mask", 0) != 0 && name.rfind("attn_inp_kq_mask", 0) != 0) {
                continue;
            }
            std::vector<uint16_t> mask(model->input_size(name), FP16_NEG_INF);
            for (size_t r = 0; r < chunk; r++) {
                for (int32_t c = 0; c <= base_pos + static_cast<int32_t>(r); c++) {
                    if (static_cast<size_t>(c) < m_n_kv) mask[r * m_n_kv + c] = 0;
                }
            }
            model->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
        OPENVINO_ASSERT(model->compute(), "[GGML] batched compute failed at position ", base_pos);
        model->read_logits(m_logits.data());
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

        OPENVINO_ASSERT(!input_ids.empty(), "[GGML] empty prompt");
        const size_t usable = m_n_kv - m_scratch_slots;
        const size_t budget = input_ids.size() + config.max_new_tokens;
        OPENVINO_ASSERT(budget <= usable,
                        "[GGML] prompt + max_new_tokens (", budget, ") exceeds the context this "
                        "graph was built for (", usable, " usable of ", m_n_kv, " slots). The graph is "
                        "shape-static, so this cannot grow at runtime -- rebuild with a larger n_kv.");

        // Reuse the K/V of the longest prefix already in the cache. A chat turn re-renders the
        // whole history, so this is what keeps a turn from re-prefilling every earlier one. The
        // last prompt token is always recomputed: its logits seed the first sample.
        size_t reuse = 0;
        const size_t reuse_limit = std::min(m_kv_tokens.size(), input_ids.size() - 1);
        while (reuse < reuse_limit && m_kv_tokens[reuse] == input_ids[reuse]) {
            reuse++;
        }
        m_kv_tokens.resize(reuse);

        auto streamer_ptr = resolve_streamer(std::move(streamer), m_tokenizer);
        auto group = std::make_shared<SequenceGroup>(0, input_ids, config);

        // Prefill in as few batched forward passes as the available buckets allow (degenerates
        // to one token per pass when the only registered bucket is size 1).
        int32_t pos = static_cast<int32_t>(reuse);
        size_t offset = reuse, remaining = input_ids.size() - reuse;
        while (remaining > 0) {
            const size_t bucket = pick_bucket(remaining);
            const size_t chunk = std::min(remaining, bucket);
            forward_batch(input_ids.data() + offset, chunk, bucket, pos);
            m_kv_tokens.insert(m_kv_tokens.end(), input_ids.begin() + offset, input_ids.begin() + offset + chunk);
            pos += static_cast<int32_t>(chunk);
            offset += chunk;
            remaining -= chunk;
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
        while (!cancelled && !group->has_finished() && pos < static_cast<int32_t>(usable)) {
            const auto running = group->get_running_sequences();
            if (running.empty()) {
                break;
            }
            const auto& gen = running.front()->get_generated_ids();
            if (gen.empty()) {
                break;
            }
            forward(gen.back(), pos++);
            m_kv_tokens.push_back(gen.back());
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
        return results;
    }

    DecodedResults generate(const std::string& prompt,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) {
        if (!m_in_chat) {
            return generate_text(prompt, true, generation_config, std::move(streamer));
        }
        m_history.push_back({{"role", "user"}, {"content", prompt}});
        auto decoded = generate_text(m_tokenizer.apply_chat_template(m_history, true), false, generation_config,
                                     std::move(streamer));
        if (!decoded.texts.empty()) {
            m_history.push_back({{"role", "assistant"}, {"content", decoded.texts.front()}});
        }
        return decoded;
    }

    /// Same contract as LLMPipeline::generate(ChatHistory): the history -- including any tools /
    /// extra_context set on it -- is rendered by the model's chat template. Repeated calls with a
    /// growing history only prefill the new suffix (see the prefix reuse in generate(ids)).
    DecodedResults generate(const ChatHistory& history,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) {
        const GenerationConfig config = generation_config.value_or(m_config);
        OPENVINO_ASSERT(config.apply_chat_template, "[GGML] chat template must be applied when using ChatHistory");
        OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "[GGML] the model has no chat template");
        return generate_text(m_tokenizer.apply_chat_template(history, true), false, generation_config,
                             std::move(streamer));
    }

    DecodedResults generate_text(const std::string& text,
                                 bool add_special_tokens,
                                 OptionalGenerationConfig generation_config,
                                 StreamerVariant streamer) {
        const auto ids = m_tokenizer.encode(text, ov::genai::add_special_tokens(add_special_tokens));
        const auto* p = ids.input_ids.data<int64_t>();
        std::vector<int64_t> input_ids(p, p + ids.input_ids.get_size());

        auto encoded = generate(input_ids, generation_config, std::move(streamer));
        DecodedResults decoded;
        for (const auto& toks : encoded.tokens) {
            decoded.texts.push_back(m_tokenizer.decode(toks));
        }
        decoded.scores = encoded.scores;
        return decoded;
    }

    void start_chat(const std::string& system_message) {
        m_in_chat = true;
        m_history.clear();
        if (!system_message.empty()) {
            m_history.push_back({{"role", "system"}, {"content", system_message}});
        }
    }

    void finish_chat() {
        m_in_chat = false;
        m_history.clear();
    }

    GenerationConfig m_config;
    Tokenizer m_tokenizer;
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> m_model;  // == m_buckets.at(1); used for decode

private:
    std::map<size_t, std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>> m_buckets;
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> m_active;  // whose cache is authoritative
    std::vector<char> m_cache_buf;  // scratch for sync_before()
    Sampler m_sampler;
    std::vector<float> m_logits;
    size_t m_n_kv = 0;
    size_t m_scratch_slots = 0;  // top slots reserved for prefill padding rows
    // Tokens whose K/V currently occupy slots 0..size()-1. generate() reuses the longest common
    // prefix with its prompt instead of recomputing it; whatever lies beyond is stale and masked.
    std::vector<int64_t> m_kv_tokens;
    bool m_in_chat = false;
    ChatHistory m_history;
};

GgmlPipeline::GgmlPipeline(const std::filesystem::path& cgraph_path,
                           const std::filesystem::path& models_path,
                           const std::string& backend)
    : m_impl(std::make_unique<Impl>(
          std::map<size_t, std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>>{
              {1, ov::ggml_cgraph_loader::GgmlModel::from_cgraph(cgraph_path.string(),
                                                                 models_path.string(), backend)}},
          Tokenizer(models_path))) {}

GgmlPipeline::GgmlPipeline(const std::vector<std::pair<size_t, std::filesystem::path>>& bucket_cgraphs,
                           const std::filesystem::path& models_path,
                           const std::string& backend) {
    std::map<size_t, std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>> buckets;
    for (const auto& b : bucket_cgraphs) {
        buckets[b.first] = ov::ggml_cgraph_loader::GgmlModel::from_cgraph(b.second.string(),
                                                                          models_path.string(),
                                                                          backend);
    }
    m_impl = std::make_unique<Impl>(std::move(buckets), Tokenizer(models_path));
}

GgmlPipeline::~GgmlPipeline() = default;

// Parsers (generation_config.parsers, TextParserStreamer) run through the same wrapper as
// LLMPipeline, with the effective config -- so parsers set via set_generation_config apply too.
DecodedResults GgmlPipeline::generate(const std::string& prompt,
                                      OptionalGenerationConfig generation_config,
                                      StreamerVariant streamer) {
    const OptionalGenerationConfig effective = generation_config.value_or(m_impl->m_config);
    return utils::run_generate_with_parsers(effective, streamer, [&]() -> DecodedResults {
        return m_impl->generate(prompt, generation_config, streamer);
    });
}

DecodedResults GgmlPipeline::generate(const ChatHistory& history,
                                      OptionalGenerationConfig generation_config,
                                      StreamerVariant streamer) {
    const OptionalGenerationConfig effective = generation_config.value_or(m_impl->m_config);
    return utils::run_generate_with_parsers(effective, streamer, [&]() -> DecodedResults {
        return m_impl->generate(history, generation_config, streamer);
    });
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
