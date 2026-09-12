# Backend Generalization Execution Plan

**Date:** 2026-09-11
**Status:** Open; artifact and expert-supply foundation implemented
**Scope:** Generalize storage, artifact selection, and runtime ownership boundaries while preserving the current DeepSeek-V4 swizzled backend.

## Decision

Project Aeon should generalize around explicit contracts, not around a universal
tensor or kernel abstraction. The current V4 graph and swizzled kernels remain
specialized. Artifact metadata and tiered expert supply become reusable by any
future backend that can describe a fixed, sector-aligned expert payload.

The first implementation slice is complete:

- `ExpertFormatDescriptor` owns artifact version, sector size, model catalog
  dimensions, and expert payload length.
- `AeonArtifactSpec` owns native filenames, expected artifact identity, and
  expert-sector alignment.
- `AeonModelLoader` derives expert metadata from the index and exposes dense
  file size instead of requiring callers to repeat it.
- `ExpertPayloadPool` owns opaque VRAM slot allocation and DMA.
- `HostExpertPool`, `PrefetchStagingArena`, `MemoryBudgetEngine`, and
  `DirectIOReader` consume runtime payload and alignment metadata.
- `UnifiedVRAMExpertPool` retains only the current swizzled W1/W2/W3 views and
  rejects those views for another format kind.
- The current `V4Pipeline` rejects a non-swizzled artifact before dense device
  weights are allocated.

The manifest contract sub-step is also complete:

- `AeonModelManifest` owns versioned model-family, architecture, backend, dense
  size, and catalog identity alongside the native artifact specification.
- `AeonModelLoader::open_model` accepts an explicit manifest and validates the
  complete loaded identity before a caller can allocate model device weights.
- The legacy artifact overload remains available while conversion-sidecar
  loading and backend factory selection are added in the next sub-step.

The backend-selection and dense-binding sub-steps are now also complete:

- `ExpertBackendRegistry` resolves the manifest backend name or loaded format,
  validates the exact current swizzled artifact contract, and reports whether
  the selected backend supports the V4 pipeline.
- `V4DenseWeightBinding` owns V4 tensor-name mapping and dense device-weight
  allocation/cleanup, while `V4Layer` retains its existing pointer contract and
  continues to own layer metadata and KV cache state.

## Stable boundaries

### V4 architecture

`V4Pipeline`, `V4Layer`, attention, routing, HC, KV state, tokenizer, and
generation remain DeepSeek-V4-specific. Their fixed kernel geometry is an
architecture contract, not a quantization-format contract.

### Artifact and supply

`AeonArtifactSpec` selects files and expected identity. `ExpertFormatDescriptor`
describes the opaque expert records. `ExpertPayloadPool`, Warm storage, staging,
direct I/O, registry capacity, and telemetry must not inspect quantization planes.

### Current weight execution

The swizzled W4A16 views and fused kernels remain behind the current backend
surface. The registry currently provides identity and capability selection, not
semantic kernel dispatch. A future backend must provide its own views and
dispatch rather than reusing these accessors by byte count or pointer
reinterpretation.

## Acceptance evidence

The current backend remains validated by:

- `test_aeon_swizzled_loader`: v2 metadata, index bounds, and incompatible
  payload rejection;
- `test_dynamic_expert_pool`: descriptor-driven budget, opaque payload DMA,
  swizzled-view guarding, and end-to-end dynamic pipeline execution;
- `test_expert_registry_warm_state`: Warm ownership and staging transitions;
- `test_direct_io`: default 4 KiB sector validation;
- `test_aeon_swizzled_pipeline`: two-layer silicon pipeline smoke;
- `ctest --test-dir build -R 'test_aeon_(loader|swizzled_loader)'`.

No performance result is attributed to this refactor. The existing v2 model
artifact and execution path remain the regression baseline.

## Remaining stages

1. **Manifest completion and backend factory.** Add sidecar parsing and
  conversion-time generation for the versioned manifest, then select an
  explicit backend factory. Keep the current default artifact path compatible
  while the manifest is introduced.
2. **Linear dispatch boundary.** Introduce semantic projection operations for a
  second working backend, with backend-owned device storage and kernels. The
  current fused swizzled path must continue to use its existing correctness
  tests.
3. **Architecture composition.** If a future model family differs in attention,
   routing, or cache semantics, add a sibling architecture orchestrator that
   consumes the shared artifact and supply contracts. Do not make V4 constants
   global defaults for unrelated architectures.
4. **Controlled comparison.** Compare backends with identical token IDs,
   residency state, context, and generation settings. Report dense bytes,
   source-tier bytes, exposed wait, transfer time, and kernel time separately.

## Explicit non-goals

- No GPTQ converter, decoder, or kernel is implemented by this plan.
- No group32 requantization or format guessing is allowed.
- No retirement or alteration of the current swizzled artifact is implied.
- No universal model tensor graph is introduced before two concrete backends
  demonstrate a shared operation contract.