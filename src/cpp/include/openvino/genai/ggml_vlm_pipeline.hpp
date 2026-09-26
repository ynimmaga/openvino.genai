// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief EXPERIMENTAL. VLM text generation over a ggml backend, from two dumped llama.cpp
 * cgraph artifacts: a vision tower (mmproj) and an embedding-input text decoder. Sibling of
 * GgmlPipeline (text-only) and VLMPipeline (InferRequest-based).
 *
 * Shares Sampler/SequenceGroup/streaming with GgmlPipeline; only prefill differs (image patch
 * + text embeddings instead of tokens).
 *
 * Scope: single-tile preprocessing only (no llava-UHD tiling); the prompt is hand-built to
 * match one validated template shape (SmolVLM/idefics3), not rendered via Jinja; no
 * multi-image/video, no chat-mode KV retention across turns.
 */
class OPENVINO_GENAI_EXPORTS GgmlVLMPipeline {
public:
    /**
     * @param vision_cgraph_path  vision-tower artifact from dump_vision (mmproj graph)
     * @param mmproj_path         the mmproj .gguf the vision weights live in
     * @param decoder_cgraph_path embedding-input decoder artifact from dump_cgraph DUMP_EMBEDDINGS=1
     * @param text_model_path     the text-model .gguf the decoder weights live in
     * @param backend             ggml backend device name; empty auto-selects
     */
    GgmlVLMPipeline(const std::filesystem::path& vision_cgraph_path,
                    const std::filesystem::path& mmproj_path,
                    const std::filesystem::path& decoder_cgraph_path,
                    const std::filesystem::path& text_model_path,
                    const std::string& backend = "");
    ~GgmlVLMPipeline();

    /// `image` is uint8 RGB, [H,W,3] or [N,H,W,3] -- same contract as VLMPipeline::generate.
    /// Only the first image is used; multi-image is not supported (see class docs).
    DecodedResults generate(const std::string& prompt,
                            const ov::Tensor& image,
                            OptionalGenerationConfig generation_config = std::nullopt,
                            StreamerVariant streamer = std::monostate{});

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& config);
    Tokenizer get_tokenizer();

    size_t context_size() const;
    std::string backend_name() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace genai
}  // namespace ov
