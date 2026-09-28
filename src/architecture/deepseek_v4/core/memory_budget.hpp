#pragma once

// -----------------------------------------------------------------------------
// Memory budget — umbrella header.
//
// The budget was one 744-line header mixing three concerns; it is now three, each
// with its own job:
//
//   * `runtime_config.hpp`         — the knobs, the safety margins, the scratch
//                                    allowances, and the staging slot counts (in
//                                    `infrastructure/core/`; model-agnostic).
//   * `memory_budget_report.hpp`   — `MemoryBudgetReport` and its derived terms.
//   * `memory_budget_engine.hpp`   — the device/host query and the feasibility
//                                    evaluation (`MemoryBudgetEngine`).
//
// This umbrella preserves the original include so callers are unchanged; a caller
// that wants only the report shape can include that header directly.
// -----------------------------------------------------------------------------

#include "infrastructure/core/runtime_config.hpp"
#include "architecture/deepseek_v4/core/memory_budget_report.hpp"
#include "architecture/deepseek_v4/core/memory_budget_engine.hpp"
