# Source Layout

Aeon is organized by ownership boundary rather than by file type alone.

```text
src/infrastructure/                 Model-independent runtime services
  core/                              Artifact loading, manifests, pools, registry, telemetry
  io/                                Aligned allocation and direct NVMe I/O
  text/                              Generic generation loop
  backend_registry/                 Backend identity and capability selection

src/architecture/deepseek_v4/       DeepSeek-V4 graph and model semantics
  core/                              V4 config, pipeline, layers, dense binding, KV state
  kernels/                           V4 attention, routing, HC, and pipeline operations
  text/                              DSV4 tokenizer and chat formatting

src/backend/swizzled_w4a16/         Current quantized expert representation and kernels
  core/                              Swizzled payload layout and device views
  kernels/                           W4A16 swizzle, GEMV, and fused expert kernels

src/platform/rdna3/                 RDNA3/HIP device selection and platform hooks
```

The infrastructure owns where model bytes live and how they move. The
architecture owns what the model computes. The backend owns how a weight
representation is interpreted and executed. The platform directory owns
hardware-specific setup; the current GPU compilation target is configured in
the top-level CMake file.