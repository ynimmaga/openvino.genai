// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

/**
 * @brief EXPERIMENTAL. Text generation on a ggml backend (Vulkan / CPU / CUDA / Metal).
 *
 * LLMPipeline is built around ov::InferRequest and so cannot drive a ggml backend. This is its
 * sibling: the model is executed by ov::ggml_emitter::GgmlModel while everything above the
 * executor -- tokenizer, chat template, sampler, stop conditions, streaming -- is the SAME GenAI
 * code the other pipelines use.
 *
 * This class exists because the sampling stack is internal to this library: Sampler::sample()
 * takes a SequenceGroup, and neither is exported, so an external application linking
 * libopenvino_genai cannot reach them. Living inside GenAI is what makes full sampling
 * (temperature, top-k/top-p/min-p, repetition/presence/frequency penalties, seeded multinomial,
 * stop strings, logprobs) available over a ggml graph.
 *
 * Two graph sources are accepted, both ending at the same executor:
 *  - a .gguf file, built by OpenVINO's native GGUF decoder builder;
 *  - a cgraph artifact dumped offline from llama.cpp, replayed verbatim. llama.cpp is a
 *    build-time tool in that case and is absent at runtime.
 *
 * LIMITATIONS. The graph is single-token, so a prompt is prefilled one token per forward pass --
 * correct, but not batched. Context length is fixed when the model is built (and, for a cgraph
 * artifact, when it was dumped). Continuous batching, beam search and LoRA are not supported.
 */
class OPENVINO_GENAI_EXPORTS GgmlPipeline {
public:
    /**
     * @brief Build from a .gguf using OpenVINO's GGUF decoder builder.
     * @param models_path path to the .gguf file
     * @param n_kv KV slots, i.e. the maximum context this pipeline can serve
     * @param backend ggml backend device name; empty selects GPU, then integrated GPU, then CPU
     */
    GgmlPipeline(const std::filesystem::path& models_path,
                 size_t n_kv = 2048,
                 const std::string& backend = "");

    /**
     * @brief Build from a cgraph artifact dumped offline from llama.cpp.
     * @param cgraph_path artifact produced by the dump_cgraph tool ("ov-cgraph-v1")
     * @param models_path the .gguf the weights are read from
     * @param backend ggml backend device name; empty auto-selects
     *
     * The artifact fixes the context length, so n_kv is taken from it rather than given.
     */
    GgmlPipeline(const std::filesystem::path& cgraph_path,
                 const std::filesystem::path& models_path,
                 const std::string& backend);

    ~GgmlPipeline();

    DecodedResults generate(const std::string& prompt,
                            OptionalGenerationConfig generation_config = std::nullopt,
                            StreamerVariant streamer = std::monostate{});

    EncodedResults generate(const std::vector<int64_t>& input_ids,
                            OptionalGenerationConfig generation_config = std::nullopt,
                            StreamerVariant streamer = std::monostate{});

    /// Retain the KV cache across generate() calls and apply the chat template to each prompt.
    void start_chat(const std::string& system_message = "");
    void finish_chat();

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& config);
    Tokenizer get_tokenizer();

    /// KV slots the underlying graph was built for.
    size_t context_size() const;
    /// ggml backend actually selected, e.g. "Vulkan0".
    std::string backend_name() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace genai
}  // namespace ov
