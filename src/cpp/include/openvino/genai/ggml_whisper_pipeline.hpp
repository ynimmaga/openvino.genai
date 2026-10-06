// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

/**
 * @brief EXPERIMENTAL. Speech recognition over a ggml backend, from a whisper.cpp model dumped
 * offline by dump_whisper (ggml-cgraph-generator): an encoder artifact, a one-token decoder
 * artifact and a .gguf holding the weights plus whisper.* metadata. Sibling of WhisperPipeline
 * (InferRequest-based) with the same generate() contract; whisper.cpp is not needed at runtime.
 *
 * Log-mel features come from GenAI's WhisperFeatureExtractor. Token ids, language codes and the
 * vocabulary come from the .gguf, so no OpenVINO tokenizer model is required.
 *
 * Scope: greedy decoding without timestamps. Audio longer than 30 s is transcribed as
 * consecutive 30 s windows (no timestamp-driven seeking, so a word cut at a window boundary may
 * be lost or garbled). Not supported: return_timestamps, word_timestamps, initial_prompt,
 * hotwords, beam search.
 */
class OPENVINO_GENAI_EXPORTS GgmlWhisperPipeline {
public:
    /**
     * @param encoder_cgraph_path <prefix>.encoder.json from dump_whisper
     * @param decoder_cgraph_path <prefix>.decoder.json from dump_whisper
     * @param model_gguf_path     <prefix>.gguf from dump_whisper (weights + whisper.* metadata)
     * @param backend             ggml backend device name; empty auto-selects
     */
    GgmlWhisperPipeline(const std::filesystem::path& encoder_cgraph_path,
                        const std::filesystem::path& decoder_cgraph_path,
                        const std::filesystem::path& model_gguf_path,
                        const std::string& backend = "");
    ~GgmlWhisperPipeline();

    /// `raw_speech` is mono 16 kHz audio normalized to about [-1, 1] -- same contract as
    /// WhisperPipeline::generate. A std::function streamer receives decoded text pieces; a
    /// StreamerBase receives token ids.
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech,
                                   OptionalWhisperGenerationConfig generation_config = std::nullopt,
                                   StreamerVariant streamer = std::monostate{});

    /// Defaults with token ids and lang_to_id taken from the model, so `language` ("<|en|>") and
    /// `task` ("transcribe" / "translate") work as with WhisperPipeline.
    WhisperGenerationConfig get_generation_config() const;
    void set_generation_config(const WhisperGenerationConfig& config);

    std::string backend_name() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace genai
}  // namespace ov
