#include "core/aeon_loader.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    const std::string model_dir = "models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon";
    aeon::core::AeonModelLoader loader;
    loader.open_model(model_dir);

    assert(loader.expert_format_version() == 2);
    assert(loader.num_layers() == 43);
    assert(loader.experts_per_layer() == 256);
    const auto& format = loader.expert_format();
    assert(format.kind == aeon::core::ExpertFormatKind::SWIZZLED_W4A16);
    assert(format.artifact_version == 2);
    assert(format.sector_size == aeon::core::AEON_SECTOR_SIZE);
    assert(format.num_layers == 43);
    assert(format.experts_per_layer == 256);
    assert(format.payload_bytes == aeon::core::AEON_EXPERT_BYTES);
    assert(loader.dense_file_size() > 0);
    assert(loader.get_expert_location(0, 0).byte_length == aeon::core::AEON_EXPERT_BYTES);
    assert(loader.get_expert_location(42, 255).file_offset ==
           static_cast<uint64_t>(42 * 256 + 255) * aeon::core::AEON_EXPERT_BYTES);
    assert(loader.get_expert_data(0, 0) != nullptr);

    aeon::core::AeonModelManifest manifest;
    manifest.model_family = "deepseek";
    manifest.architecture = "deepseek_v4";
    manifest.weight_backend = "swizzled_w4a16";
    manifest.artifact = aeon::core::make_current_swizzled_artifact_spec();
    manifest.dense_file_bytes = loader.dense_file_size();
    manifest.num_layers = loader.num_layers();
    manifest.experts_per_layer = loader.experts_per_layer();

    const auto& backend = aeon::core::ExpertBackendRegistry::resolve(
        manifest.weight_backend, loader.expert_format());
    assert(backend.name == "swizzled_w4a16");
    assert(backend.supports_v4_pipeline);

    auto incompatible_format = loader.expert_format();
    incompatible_format.artifact_version += 1;
    bool rejected_backend_format_mismatch = false;
    try {
        (void)aeon::core::ExpertBackendRegistry::resolve(
            manifest.weight_backend, incompatible_format);
    } catch (const std::runtime_error&) {
        rejected_backend_format_mismatch = true;
    }
    assert(rejected_backend_format_mismatch);

    aeon::core::AeonModelLoader manifest_loader;
    manifest_loader.open_model(model_dir, manifest);
    assert(manifest_loader.expert_format().kind == aeon::core::ExpertFormatKind::SWIZZLED_W4A16);

    manifest.weight_backend = "gptq";
    bool rejected_backend_mismatch = false;
    try {
        aeon::core::AeonModelLoader incompatible_manifest_loader;
        incompatible_manifest_loader.open_model("missing-model-directory", manifest);
    } catch (const std::invalid_argument&) {
        rejected_backend_mismatch = true;
    }
    assert(rejected_backend_mismatch);

    auto incompatible = aeon::core::make_current_swizzled_artifact_spec();
    incompatible.expected_expert_payload_bytes += aeon::core::AEON_SECTOR_SIZE;
    bool rejected_incompatible_artifact = false;
    try {
        aeon::core::AeonModelLoader incompatible_loader;
        incompatible_loader.open_model(model_dir, incompatible);
    } catch (const std::runtime_error&) {
        rejected_incompatible_artifact = true;
    }
    assert(rejected_incompatible_artifact);

    std::cout << "[PASS] Version-2 swizzled Aeon expert loader" << std::endl;
    return 0;
}