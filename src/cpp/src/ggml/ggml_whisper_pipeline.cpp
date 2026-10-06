// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Two GgmlModel instances from one whisper.cpp model: the encoder (log-mel -> cross-attention
// K/V) and a one-token decoder. The encoder's cache_{k,v}_cross are copied into the decoder's
// once per 30 s window; the decoder then steps greedily over its own self-attention cache.

#include "openvino/genai/ggml_whisper_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov {
namespace genai {

namespace {

using ov::ggml_cgraph_loader::GgmlModel;

int32_t meta_i32(const GgmlModel& m, const std::string& key) {
    int32_t v = 0;
    OPENVINO_ASSERT(m.gguf_meta_i32(key, v), "[GGML Whisper] model .gguf is missing ", key,
                    " -- was it written by dump_whisper?");
    return v;
}

// Length of the longest prefix of `s` that does not end inside a UTF-8 sequence. Tokens are
// byte-level, so one character can span tokens; streaming must hold the tail back.
size_t utf8_complete_prefix(const std::string& s) {
    for (size_t back = 1; back <= 4 && back <= s.size(); back++) {
        const auto c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) {
            continue;  // continuation byte, keep looking for the lead byte
        }
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return back >= need ? s.size() : s.size() - back;
    }
    return s.size();
}

}  // namespace

class GgmlWhisperPipeline::Impl {
public:
    Impl(std::shared_ptr<GgmlModel> encoder, std::shared_ptr<GgmlModel> decoder)
        : m_encoder(std::move(encoder)),
          m_decoder(std::move(decoder)),
          m_n_mels(meta_i32(*m_decoder, "whisper.n_mels")),
          m_feature_extractor(m_n_mels, 16000, 400, 160) {
        const GgmlModel& d = *m_decoder;
        m_n_text_ctx = meta_i32(d, "whisper.n_text_ctx");
        m_eot = meta_i32(d, "whisper.token.eot");
        m_sot = meta_i32(d, "whisper.token.sot");
        m_not = meta_i32(d, "whisper.token.not");
        m_multilingual = meta_i32(d, "whisper.multilingual") != 0;
        m_n_kv = d.context_size();
        m_logits.resize(d.logits_size());

        // The encoder graph consumes one fixed-size window: [2 * n_audio_ctx frames, n_mels].
        m_window_frames = m_encoder->input_size("mel") / m_n_mels;
        OPENVINO_ASSERT(m_window_frames > 0 && m_window_frames * m_n_mels == m_encoder->input_size("mel"),
                        "[GGML Whisper] encoder artifact has no usable 'mel' input");
        OPENVINO_ASSERT(m_n_kv >= static_cast<size_t>(m_n_text_ctx),
                        "[GGML Whisper] decoder KV (", m_n_kv, ") smaller than n_text_ctx (", m_n_text_ctx, ")");
        // WhisperFeatureExtractor's numeric constructor disables pre-padding; restore whisper's
        // 30 s chunking so audio shorter than one window is padded with real silence frames.
        m_feature_extractor.nb_max_frames = m_window_frames;
        m_feature_extractor.n_samples = m_window_frames * m_feature_extractor.hop_length;
        m_feature_extractor.chunk_length = m_feature_extractor.n_samples / m_feature_extractor.sampling_rate;

        std::vector<uint8_t> bytes;
        std::vector<int32_t> offsets;
        OPENVINO_ASSERT(d.gguf_meta_u8_array("whisper.vocab.bytes", bytes) &&
                            d.gguf_meta_i32_array("whisper.vocab.offsets", offsets) &&
                            offsets.size() == m_logits.size() + 1,
                        "[GGML Whisper] model .gguf has no usable whisper.vocab.*");
        m_vocab.resize(m_logits.size());
        for (size_t i = 0; i < m_vocab.size(); i++) {
            m_vocab[i].assign(bytes.begin() + offsets[i], bytes.begin() + offsets[i + 1]);
        }

        m_config.set_eos_token_id(m_eot);
        m_config.pad_token_id = m_eot;
        m_config.decoder_start_token_id = m_sot;
        m_config.no_timestamps_token_id = m_not;
        m_config.translate_token_id = meta_i32(d, "whisper.token.translate");
        m_config.transcribe_token_id = meta_i32(d, "whisper.token.transcribe");
        m_config.prev_sot_token_id = meta_i32(d, "whisper.token.prev");
        m_config.is_multilingual = m_multilingual;
        m_config.lang_to_id.clear();
        std::vector<std::string> langs;
        if (m_multilingual && d.gguf_meta_str_array("whisper.languages", langs)) {
            for (size_t i = 0; i < langs.size(); i++) {
                m_config.lang_to_id["<|" + langs[i] + "|>"] = m_sot + 1 + static_cast<int64_t>(i);
            }
        }
        // whisper.cpp's suppress_blank: neither a bare space nor end-of-text may come first.
        m_config.begin_suppress_tokens = {m_eot};
        const auto space = std::find(m_vocab.begin(), m_vocab.begin() + m_eot, std::string(" "));
        if (space != m_vocab.begin() + m_eot) {
            m_config.begin_suppress_tokens.push_back(space - m_vocab.begin());
        }
        m_config.suppress_tokens.clear();
    }

    WhisperDecodedResults generate(const RawSpeechInput& raw_speech, OptionalWhisperGenerationConfig opt_config,
                                   StreamerVariant streamer) {
        WhisperGenerationConfig config = opt_config.value_or(m_config);
        // Token ids are a property of the model, not a user choice: a default-constructed
        // WhisperGenerationConfig carries multilingual-vocabulary ids that are wrong for e.g. a
        // .en model.
        config.set_eos_token_id(m_config.eos_token_id);
        config.decoder_start_token_id = m_config.decoder_start_token_id;
        config.no_timestamps_token_id = m_config.no_timestamps_token_id;
        config.translate_token_id = m_config.translate_token_id;
        config.transcribe_token_id = m_config.transcribe_token_id;
        config.prev_sot_token_id = m_config.prev_sot_token_id;
        config.is_multilingual = m_config.is_multilingual;
        config.lang_to_id = m_config.lang_to_id;
        config.validate();
        OPENVINO_ASSERT(!config.return_timestamps && !config.word_timestamps,
                        "[GGML Whisper] timestamps are not implemented for the ggml backend");
        OPENVINO_ASSERT(!config.initial_prompt && !config.hotwords,
                        "[GGML Whisper] initial_prompt / hotwords are not implemented for the ggml backend");
        OPENVINO_ASSERT(!config.is_beam_search() && config.num_return_sequences == 1,
                        "[GGML Whisper] only greedy decoding is implemented for the ggml backend");

        auto features = m_feature_extractor.extract(raw_speech, true);

        std::vector<int64_t> tokens;
        std::string text;
        size_t streamed_bytes = 0;
        float score = 0.0f;
        bool cancelled = false;
        int64_t lang_token = -1;

        auto* stream_text = std::get_if<std::function<StreamingStatus(std::string)>>(&streamer);
        auto* stream_ptr = std::get_if<std::shared_ptr<StreamerBase>>(&streamer);
        auto emit = [&](int64_t token) {
            if (stream_ptr && *stream_ptr) {
                cancelled = (*stream_ptr)->write(token) != StreamingStatus::RUNNING;
            } else if (stream_text && *stream_text) {
                const size_t end = utf8_complete_prefix(text);
                if (end > streamed_bytes) {
                    cancelled = (*stream_text)(text.substr(streamed_bytes, end - streamed_bytes)) !=
                                StreamingStatus::RUNNING;
                    streamed_bytes = end;
                }
            }
        };

        for (size_t offset = 0; !cancelled && (offset == 0 || offset < features.n_active_frames);
             offset += m_window_frames) {
            encode(features.get_data_with_offset(offset, m_window_frames));

            if (m_multilingual && lang_token < 0) {
                lang_token = config.language ? language_token(config, *config.language) : detect_language(config);
            }
            std::vector<int32_t> prompt{static_cast<int32_t>(m_sot)};
            if (m_multilingual) {
                prompt.push_back(static_cast<int32_t>(lang_token));
                prompt.push_back(static_cast<int32_t>(config.task && *config.task == "translate"
                                                          ? config.translate_token_id
                                                          : config.transcribe_token_id));
            }
            prompt.push_back(static_cast<int32_t>(m_not));

            int32_t pos = 0;
            for (int32_t t : prompt) {
                decode(t, pos++);
            }
            // whisper.cpp caps one window's output at half the text context.
            const size_t window_limit = static_cast<size_t>(m_n_text_ctx / 2);
            for (size_t n = 0; n < window_limit && pos < m_n_text_ctx && tokens.size() < config.max_new_tokens; n++) {
                const int64_t next = pick(config, n == 0, score);
                if (next == m_eot) {
                    break;
                }
                tokens.push_back(next);
                text += m_vocab[next];
                emit(next);
                if (cancelled) {
                    break;
                }
                decode(static_cast<int32_t>(next), pos++);
            }
            if (tokens.size() >= config.max_new_tokens) {
                break;
            }
        }
        if (stream_text && *stream_text && !cancelled && streamed_bytes < text.size()) {
            (*stream_text)(text.substr(streamed_bytes));
        }
        if (stream_ptr && *stream_ptr) {
            (*stream_ptr)->end();
        }

        WhisperDecodedResults out;
        out.texts.push_back(text);
        out.scores.push_back(score);
        if (!m_multilingual) {
            out.language = "en";
        } else {
            for (const auto& kv : config.lang_to_id) {
                if (kv.second == lang_token) {
                    out.language = kv.first.substr(2, kv.first.size() - 4);  // "<|en|>" -> "en"
                }
            }
        }
        return out;
    }

    WhisperGenerationConfig m_config;
    std::shared_ptr<GgmlModel> m_encoder, m_decoder;

private:
    // Run the encoder on one [n_mels, window_frames] window and hand its cross-attention K/V to the
    // decoder. The two models own separate copies of those caches, so this is a host round trip --
    // a few MB, once per window.
    void encode(const std::vector<float>& mel) {
        OPENVINO_ASSERT(m_encoder->write_input("mel", mel.data(), mel.size() * sizeof(float)),
                        "[GGML Whisper] encoder has no 'mel' input");
        OPENVINO_ASSERT(m_encoder->compute(), "[GGML Whisper] encoder compute failed");
        for (const char* name : {"cache_k_cross", "cache_v_cross"}) {
            const size_t n = m_encoder->input_nbytes(name);
            OPENVINO_ASSERT(n > 0 && n == m_decoder->input_nbytes(name),
                            "[GGML Whisper] encoder and decoder disagree on ", name);
            m_cross.resize(n);
            m_encoder->read_input(name, m_cross.data(), n);
            m_decoder->write_input(name, m_cross.data(), n);
        }
    }

    // One token through the decoder at `pos`, writing its K/V to cache cell `pos`.
    void decode(int32_t token, int32_t pos) {
        const int64_t cell = pos;
        m_mask.assign(m_n_kv, -std::numeric_limits<float>::infinity());
        std::fill(m_mask.begin(), m_mask.begin() + pos + 1, 0.0f);
        m_decoder->write_input("embd", &token, sizeof(token));
        m_decoder->write_input("position", &pos, sizeof(pos));
        m_decoder->write_input("inp_kv_idx", &cell, sizeof(cell));
        m_decoder->write_input("KQ_mask", m_mask.data(), m_mask.size() * sizeof(float));
        OPENVINO_ASSERT(m_decoder->compute(), "[GGML Whisper] decoder compute failed at position ", pos);
        m_decoder->read_logits(m_logits.data());
    }

    // Accepts "<|en|>" (WhisperGenerationConfig's convention) or a bare "en".
    static int64_t language_token(const WhisperGenerationConfig& config, const std::string& language) {
        const std::string key = language.rfind("<|", 0) == 0 ? language : "<|" + language + "|>";
        auto it = config.lang_to_id.find(key);
        OPENVINO_ASSERT(it != config.lang_to_id.end(), "[GGML Whisper] model does not support language ", language);
        return it->second;
    }

    // Language identification as in whisper.cpp / WhisperPipeline: after <|startoftranscript|>,
    // the most likely language token.
    int64_t detect_language(const WhisperGenerationConfig& config) {
        decode(static_cast<int32_t>(m_sot), 0);
        int64_t best = -1;
        for (const auto& kv : config.lang_to_id) {
            if (best < 0 || m_logits[kv.second] > m_logits[best]) {
                best = kv.second;
            }
        }
        OPENVINO_ASSERT(best >= 0, "[GGML Whisper] model declares no languages");
        return best;
    }

    // Greedy pick over text tokens + end-of-text. Without timestamps whisper.cpp suppresses every
    // id above <|endoftext|> (control, language, task and timestamp tokens).
    int64_t pick(const WhisperGenerationConfig& config, bool first, float& score) {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        std::fill(m_logits.begin() + m_eot + 1, m_logits.end(), neg_inf);
        for (int64_t t : config.suppress_tokens) {
            if (t >= 0 && static_cast<size_t>(t) < m_logits.size()) m_logits[t] = neg_inf;
        }
        if (first) {
            for (int64_t t : config.begin_suppress_tokens) {
                if (t >= 0 && static_cast<size_t>(t) < m_logits.size()) m_logits[t] = neg_inf;
            }
        }
        const auto best = std::max_element(m_logits.begin(), m_logits.end());
        double sum = 0.0;
        for (float l : m_logits) {
            sum += std::exp(static_cast<double>(l - *best));
        }
        score -= static_cast<float>(std::log(sum));  // log-softmax of the argmax
        return best - m_logits.begin();
    }

    int32_t m_n_mels;
    WhisperFeatureExtractor m_feature_extractor;
    int32_t m_n_text_ctx = 0;
    int64_t m_eot = 0, m_sot = 0, m_not = 0;
    bool m_multilingual = false;
    size_t m_n_kv = 0, m_window_frames = 0;
    std::vector<std::string> m_vocab;
    std::vector<float> m_logits, m_mask;
    std::vector<uint8_t> m_cross;
};

GgmlWhisperPipeline::GgmlWhisperPipeline(const std::filesystem::path& encoder_cgraph_path,
                                         const std::filesystem::path& decoder_cgraph_path,
                                         const std::filesystem::path& model_gguf_path,
                                         const std::string& backend)
    : m_impl(std::make_unique<Impl>(
          GgmlModel::from_cgraph(encoder_cgraph_path.string(), model_gguf_path.string(), backend),
          GgmlModel::from_cgraph(decoder_cgraph_path.string(), model_gguf_path.string(), backend))) {}

GgmlWhisperPipeline::~GgmlWhisperPipeline() = default;

WhisperDecodedResults GgmlWhisperPipeline::generate(const RawSpeechInput& raw_speech,
                                                    OptionalWhisperGenerationConfig generation_config,
                                                    StreamerVariant streamer) {
    return m_impl->generate(raw_speech, generation_config, std::move(streamer));
}

WhisperGenerationConfig GgmlWhisperPipeline::get_generation_config() const {
    return m_impl->m_config;
}

void GgmlWhisperPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    m_impl->m_config = config;
    m_impl->m_config.validate();
}

std::string GgmlWhisperPipeline::backend_name() const {
    return m_impl->m_decoder->backend_name();
}

}  // namespace genai
}  // namespace ov
