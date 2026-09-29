# Source Layout

Aeon is organized by ownership boundary rather than by file type alone.

```text
src/infrastructure/                 Model-independent runtime services
  memory/                            Runtime config, budget engine, memory geometry
  expert/residency/                  Expert residency registry and its partition
  expert/transport/                  Tiered supply, staging, telemetry, direct I/O
  expert/storage/                    Payload pools, host region, format descriptor
  expert/                            Tier state and loader, and the neutral ports
  prefill/                           Sweep, lookahead, controller, carry
  routing/                           Routing profile, counters, reuse profiler
  artifact/                          .aeon container loading, manifest, tensors
  io/                                Aligned allocation and direct NVMe I/O
  text/                              Generic generation loop
  backend_registry/                 Backend identity and capability selection

src/architecture/deepseek_v4/       DeepSeek-V4 graph and model semantics
  spec/                              Config, spec, contract, memory geometry, dense binding
  layer/                             Layer, state, body phases, scratch, attention trace
  moe/                               Routed-expert executor and its supply adapter
  runtime/                           Host, resources, graph, engine, prefill workspace
  kernels/                           V4 attention, routing, HC, and pipeline operations
  text/                              DSV4 tokenizer and chat formatting
  reference/                         The independent fp64 oracle

src/backend/swizzled_w4a16/         Current quantized expert representation and kernels
  core/                              Swizzled payload layout and device views
  kernels/                           W4A16 swizzle, GEMV, and fused expert kernels

src/platform/rdna3/                 RDNA3/HIP device selection and platform hooks
src/platform/ops/                   Model-agnostic GPU primitives (rmsnorm, gemv, argmax)
```

The infrastructure owns where model bytes live and how they move. The
architecture owns what the model computes. The backend owns how a weight
representation is interpreted and executed. The platform directory owns
hardware-specific setup; the current GPU compilation target is configured in
the top-level CMake file.