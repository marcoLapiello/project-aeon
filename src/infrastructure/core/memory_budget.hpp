#pragma once

// -----------------------------------------------------------------------------
// Memory budget — umbrella header.
//
// The budget was one 744-line header mixing three concerns; it is now four pieces,
// each with its own job, all model-agnostic:
//
//   * `runtime_config.hpp`         — the knobs, the safety margins, the scratch
//                                    allowances, and the staging slot counts.
//   * `model_memory_geometry.hpp`  — the architecture-supplied input the engine
//                                    reads (the seam that keeps this model-free).
//   * `memory_budget_report.hpp`   — `MemoryBudgetReport` and its derived terms.
//   * `memory_budget_engine.hpp`   — the device/host query and the feasibility
//                                    evaluation (`MemoryBudgetEngine`).
//
// A caller that wants only the report shape can include that header directly; a
// caller that evaluates the budget supplies a `ModelMemoryGeometry` built by its
// architecture (V4's is `architecture/deepseek_v4/spec/v4_memory_geometry.hpp`).
// -----------------------------------------------------------------------------

#include "infrastructure/core/runtime_config.hpp"
#include "infrastructure/core/model_memory_geometry.hpp"
#include "infrastructure/core/memory_budget_report.hpp"
#include "infrastructure/core/memory_budget_engine.hpp"
