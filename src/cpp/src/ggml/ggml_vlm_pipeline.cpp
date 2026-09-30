// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Two GgmlModel instances (vision tower, embedding-input decoder) sharing one KV cache,
// driven by the same Sampler/SequenceGroup stack GgmlPipeline uses for text.

#include "openvino/genai/ggml_vlm_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"

#include "openvino/genai/text_streamer.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"

namespace ov {
namespace genai {

namespace {

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

// F16 -> F32, without pulling in a ggml header for one scalar op.
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            exp = 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Bilinear-resize an [H,W,3] (or [N,H,W,3], first image only) uint8 RGB tensor to size x size,
// normalise per channel, and pack into ggml's expected planar CHW layout (verified against
// clip.cpp's inp_raw fill loop).
std::vector<float> preprocess_image(const ov::Tensor& image, int size, const std::vector<float>& mean,
                                    const std::vector<float>& std_dev) {
    const auto& shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 || shape.size() == 4,
                    "[GGML VLM] image must be [H,W,3] or [N,H,W,3], got rank ", shape.size());
    const size_t off = shape.size() == 4 ? 1 : 0;
    const int src_h = static_cast<int>(shape[off]);
    const int src_w = static_cast<int>(shape[off + 1]);
    OPENVINO_ASSERT(shape[off + 2] == 3, "[GGML VLM] image must be RGB (3 channels)");
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8, "[GGML VLM] image must be uint8");
    const auto* src = image.data<uint8_t>();

    std::vector<float> out(static_cast<size_t>(size) * size * 3);
    const float sx = static_cast<float>(src_w) / size;
    const float sy = static_cast<float>(src_h) / size;
    const int plane = size * size;

    for (int y = 0; y < size; y++) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, src_h - 1);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const float wy = std::clamp(fy - y0, 0.0f, 1.0f);
        for (int x = 0; x < size; x++) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, src_w - 1);
            const int x1 = std::min(x0 + 1, src_w - 1);
            const float wx = std::clamp(fx - x0, 0.0f, 1.0f);
            for (int c = 0; c < 3; c++) {
                auto px = [&](int yy, int xx) {
                    return static_cast<float>(src[(static_cast<size_t>(yy) * src_w + xx) * 3 + c]);
                };
                const float top = px(y0, x0) * (1 - wx) + px(y0, x1) * wx;
                const float bot = px(y1, x0) * (1 - wx) + px(y1, x1) * wx;
                const float v = (top * (1 - wy) + bot * wy) / 255.0f;
                out[c * plane + y * size + x] = (v - mean[c]) / std_dev[c];
            }
        }
    }
    return out;
}

}  // namespace

class GgmlVLMPipeline::Impl {
public:
    Impl(std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> vision,
        std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> decoder, Tokenizer tokenizer)
        : m_vision(std::move(vision)),
          m_decoder(std::move(decoder)),
          m_tokenizer(std::move(tokenizer)),
          m_sampler(m_tokenizer),
          m_logits(m_decoder->logits_size()) {
        m_n_kv = m_decoder->context_size();
        OPENVINO_ASSERT(m_n_kv > 0, "[GGML VLM] decoder did not report a context size");
        m_config.eos_token_id = m_tokenizer.get_eos_token_id();

        int32_t sz = 0;
        std::vector<float> mean, std_dev;
        OPENVINO_ASSERT(m_vision->gguf_meta_i32("clip.vision.image_size", sz),
                        "[GGML VLM] mmproj is missing clip.vision.image_size");
        OPENVINO_ASSERT(m_vision->gguf_meta_f32_array("clip.vision.image_mean", mean) &&
                            m_vision->gguf_meta_f32_array("clip.vision.image_std", std_dev),
                        "[GGML VLM] mmproj is missing clip.vision.image_mean/image_std");
        m_image_size = sz;
        m_mean = mean;
        m_std = std_dev;
    }

    // One embedding through the decoder at position `pos`. `embd` is the only live input path
    // into this graph (see header).
    void forward_embd(const float* embd, size_t n_embd, int32_t pos) {
        const int32_t oid = 0;
        const int64_t slot = pos;
        m_decoder->write_input("embd", embd, n_embd * sizeof(float));
        m_decoder->write_input("inp_pos", &pos, sizeof(pos));
        m_decoder->write_input("inp_out_ids", &oid, sizeof(oid));
        for (const auto& name : m_decoder->input_names()) {
            if (name.rfind("inp_kv_idx", 0) == 0) {
                m_decoder->write_input(name, &slot, sizeof(slot));
            }
        }
        for (const auto& name : m_decoder->input_names()) {
            if (name.rfind("self_kq_mask", 0) != 0 && name.rfind("attn_inp_kq_mask", 0) != 0) {
                continue;
            }
            std::vector<uint16_t> mask(m_decoder->input_size(name), FP16_NEG_INF);
            for (int j = 0; j <= pos && j < static_cast<int>(m_n_kv); j++) {
                mask[j] = 0;
            }
            m_decoder->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
        OPENVINO_ASSERT(m_decoder->compute(), "[GGML VLM] decoder compute failed at position ", pos);
    }

    std::vector<float> token_embedding(int64_t token_id, size_t n_embd) {
        const size_t row_bytes = n_embd * sizeof(uint16_t);
        std::vector<uint16_t> f16(n_embd);
        OPENVINO_ASSERT(m_decoder->read_weight("token_embd.weight",
                                               static_cast<size_t>(token_id) * row_bytes,
                                               f16.data(), row_bytes),
                        "[GGML VLM] token_embd.weight lookup failed for token ", token_id);
        std::vector<float> out(n_embd);
        for (size_t i = 0; i < n_embd; i++) {
            out[i] = fp16_to_fp32(f16[i]);
        }
        return out;
    }

    DecodedResults generate(const std::string& prompt, const ov::Tensor& image,
                            OptionalGenerationConfig generation_config, StreamerVariant streamer) {
        GenerationConfig config = generation_config.value_or(m_config);
        if (config.eos_token_id == -1) {
            config.set_eos_token_id(m_config.eos_token_id);
        }
        config.validate();
        OPENVINO_ASSERT(!config.is_beam_search() && config.num_return_sequences == 1,
                        "[GGML VLM] beam search / multiple return sequences need KV reordering "
                        "or copy-on-write, not implemented for the ggml backend");

        // vision: one forward pass, no KV, no position
        auto pixels = preprocess_image(image, m_image_size, m_mean, m_std);
        OPENVINO_ASSERT(m_vision->write_input("inp_raw", pixels.data(), pixels.size() * sizeof(float)),
                        "[GGML VLM] vision graph has no inp_raw input");
        OPENVINO_ASSERT(m_vision->compute(), "[GGML VLM] vision compute failed");
        const size_t n_embd = m_vision->logits_size();
        // read_logits() copies one row; the projector output is [n_embd, n_patches], so read
        // it all and derive the patch count.
        std::vector<float> patch_embd(m_vision->output_size());
        m_vision->read_output(patch_embd.data());
        const size_t n_patches = patch_embd.size() / n_embd;

        // Hand-rendered prompt (see header): "<|im_start|>User:" + [N image positions] +
        // "{prompt}<end_of_utterance>\nAssistant:"
        auto tokenize = [&](const std::string& s) {
            auto ids = m_tokenizer.encode(s, ov::AnyMap{{"add_special_tokens", false}}).input_ids;
            const auto* p = ids.data<int64_t>();
            return std::vector<int64_t>(p, p + ids.get_size());
        };
        const auto prefix_ids = tokenize("<|im_start|>User:");
        const auto suffix_ids = tokenize(prompt + "<end_of_utterance>\nAssistant:");

        auto streamer_ptr = resolve_streamer(std::move(streamer), m_tokenizer);
        auto group = std::make_shared<SequenceGroup>(0, std::vector<int64_t>{0}, config);
        std::vector<SequenceGroup::Ptr> groups{group};

        int32_t pos = 0;
        for (int64_t id : prefix_ids) {
            forward_embd(token_embedding(id, n_embd).data(), n_embd, pos++);
        }
        for (size_t p = 0; p < n_patches && pos < static_cast<int32_t>(m_n_kv); p++) {
            forward_embd(patch_embd.data() + p * n_embd, n_embd, pos++);
        }
        for (int64_t id : suffix_ids) {
            forward_embd(token_embedding(id, n_embd).data(), n_embd, pos++);
        }
        m_decoder->read_logits(m_logits.data());
        ov::Tensor logits_tensor(ov::element::f32, ov::Shape{1, 1, m_logits.size()}, m_logits.data());
        group->schedule_tokens(1);
        group->set_output_seq_len(1);
        m_sampler.sample(groups, logits_tensor);

        size_t streamed = 0;
        auto stream_new = [&]() -> bool {
            if (!streamer_ptr) return false;
            const auto running = group->get_running_sequences();
            if (running.empty()) return false;
            const auto& gen = running.front()->get_generated_ids();
            while (streamed < gen.size()) {
                if (streamer_ptr->write(gen[streamed++]) != StreamingStatus::RUNNING) return true;
            }
            return false;
        };
        bool cancelled = stream_new();

        while (!cancelled && !group->has_finished() && pos < static_cast<int32_t>(m_n_kv)) {
            const auto running = group->get_running_sequences();
            if (running.empty()) break;
            const auto& gen = running.front()->get_generated_ids();
            if (gen.empty()) break;
            forward_embd(token_embedding(gen.back(), n_embd).data(), n_embd, pos++);
            m_decoder->read_logits(m_logits.data());
            group->schedule_tokens(1);
            m_sampler.sample(groups, logits_tensor);
            cancelled = stream_new();
        }
        if (streamer_ptr) streamer_ptr->end();

        DecodedResults out;
        const auto finished = group->get_finished_sequences();
        const auto& seq = !finished.empty() ? finished.front() : group->get_running_sequences().front();
        out.texts.push_back(m_tokenizer.decode(seq->get_generated_ids()));
        out.scores.push_back(seq->get_cumulative_log_prob());
        m_sampler.clear_request_info(group->get_request_id());
        return out;
    }

    GenerationConfig m_config;
    Tokenizer m_tokenizer;
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> m_vision, m_decoder;

private:
    Sampler m_sampler;
    std::vector<float> m_logits;
    size_t m_n_kv = 0;
    int m_image_size = 0;
    std::vector<float> m_mean, m_std;
};

GgmlVLMPipeline::GgmlVLMPipeline(const std::filesystem::path& vision_cgraph_path,
                                 const std::filesystem::path& mmproj_path,
                                 const std::filesystem::path& decoder_cgraph_path,
                                 const std::filesystem::path& text_model_path,
                                 const std::string& backend)
    : m_impl(std::make_unique<Impl>(
          ov::ggml_cgraph_loader::GgmlModel::from_cgraph(vision_cgraph_path.string(),
                                                         mmproj_path.string(), backend),
          ov::ggml_cgraph_loader::GgmlModel::from_cgraph(decoder_cgraph_path.string(),
                                                         text_model_path.string(), backend),
          Tokenizer(text_model_path))) {}

GgmlVLMPipeline::~GgmlVLMPipeline() = default;

DecodedResults GgmlVLMPipeline::generate(const std::string& prompt, const ov::Tensor& image,
                                         OptionalGenerationConfig generation_config,
                                         StreamerVariant streamer) {
    return m_impl->generate(prompt, image, generation_config, std::move(streamer));
}

GenerationConfig GgmlVLMPipeline::get_generation_config() const {
    return m_impl->m_config;
}

void GgmlVLMPipeline::set_generation_config(const GenerationConfig& config) {
    m_impl->m_config = config;
    m_impl->m_config.validate();
}

Tokenizer GgmlVLMPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

size_t GgmlVLMPipeline::context_size() const {
    return m_impl->m_decoder->context_size();
}

std::string GgmlVLMPipeline::backend_name() const {
    return m_impl->m_decoder->backend_name();
}

}  // namespace genai
}  // namespace ov
