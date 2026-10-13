// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include "openvino/core/core.hpp"
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/properties.hpp>

namespace ov {
namespace genai {

/**
 * @brief A map of models for VLMPipeline constructor.
 * Key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
 * and value is a pair of model IR as string and weights as tensor.
 */
using ModelsMap = std::map<std::string, std::pair<std::string, ov::Tensor>>;

/**
 * @brief A map of already compiled models for VLMPipeline constructor, with the same keys as ModelsMap
 * (e.g. "vision_embeddings", "text_embeddings", "language").
 */
using CompiledModelsMap = std::map<std::string, ov::CompiledModel>;

/**
 * @brief blob_path property defines a path to a directory containing compiled blobs previously exported with
 * `pipeline.export_model` method.
 *
 * Use of compiled blobs can significantly reduce model load time, especially for large models.
 */
static constexpr ov::Property<std::filesystem::path> blob_path{"blob_path"};

}  // namespace genai
}  // namespace ov
