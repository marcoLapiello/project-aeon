#pragma once

// -----------------------------------------------------------------------------
// DeepSeek-V4 attention kernels — umbrella.
//
// The kernels live by concern, each in its own header:
//   * `v4_attention_config.hpp`    — the DSV4_* hyperparameters;
//   * `v4_attention_kernels.hpp`   — sliding / cached-sliding / cached-compressed
//     attention, compressor state + materialization, and the indexer scores;
//   * `v4_grouped_wo.hpp`          — the grouped W_o_a projection and half->float;
//   * `v4_hc_head_kernel.hpp`      — the Hyper-Connections head reduction.
//
// The LM-head argmax moved to `platform/ops/argmax.hpp`: it is a model-agnostic
// reduction, not an attention kernel.
//
// The Wave32 RoPE and GEMV kernels are focused modules too, included here for
// this header's callers; there is no second definition. The Wave32 RMSNorm moved
// to `platform/ops/rmsnorm.hpp` (it is model-agnostic).
//
// The system and HIP includes below are kept exactly as they were so a caller
// that relied on including this header for them still compiles.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <cassert>

#include "architecture/deepseek_v4/kernels/v4_attention_config.hpp"
#include "architecture/deepseek_v4/kernels/v4_attention_kernels.hpp"
#include "architecture/deepseek_v4/kernels/v4_grouped_wo.hpp"
#include "architecture/deepseek_v4/kernels/v4_hc_head_kernel.hpp"

#include "platform/ops/argmax.hpp"

#include "platform/ops/rmsnorm.hpp"
#include "architecture/deepseek_v4/kernels/v4_rope.hpp"
#include "architecture/deepseek_v4/kernels/v4_gemv.hpp"
