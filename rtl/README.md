# RTL

This folder contains two kinds of SystemVerilog code:

- legacy validation RTL at the root of `rtl/`
- new production TinyLlama RTL under subfolders such as `rtl/common/`, `rtl/control/`, and later `rtl/memory/`, `rtl/compute/`, `rtl/nonlinear/`, and `rtl/top/`

The new production path should use the subfolder-based code. The flat root-level files are still useful as references and validation scaffolding, but they are not the final TinyLlama runtime.

## Production Foundation

These are the first production RTL files that now exist.

| File | What it is | Smoke test |
|------|------------|------------|
| `common/tinyllama_pkg.sv` | Shared architectural constants, widths, tiling parameters, and enums for the TinyLlama accelerator. | Syntax-check it together with the dependent common files using the package compile command below. |
| `common/tinyllama_bus_pkg.sv` | Shared packed bus, descriptor, and sideband tag types used across the production RTL. | Syntax-check it together with the dependent common files using the package compile command below. |
| `common/stream_fifo.sv` | Parameterized ready/valid FIFO primitive used at elastic datapath boundaries. | Run `rtl/tb/tb_stream_fifo.sv`. |
| `common/skid_buffer.sv` | Two-entry skid buffer built on top of `stream_fifo.sv` for short backpressure absorption. | Covered by the common syntax check; no standalone testbench yet. |
| `common/descriptor_fifo.sv` | Descriptor-oriented FIFO wrapper used for command and DMA-style queues. | Run `rtl/tb/tb_descriptor_fifo.sv`. |

### Common Package Smoke Test

Run this from the project root:

```powershell
iverilog -g2012 -t null `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/common/stream_fifo.sv `
  rtl/common/skid_buffer.sv `
  rtl/common/descriptor_fifo.sv
```

This is the quickest syntax/integration check for the current production common layer.

## Phase 1 Control Skeleton

These are the first production control-path modules now implemented under `rtl/control/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `control/axi_lite_ctrl_slave.sv` | AXI4-Lite front-end that translates host register reads and writes into the internal register-file interface. | Run `rtl/tb/tb_axi_lite_ctrl_slave.sv`. |
| `control/kernel_reg_file.sv` | Concrete launch/status register file with the frozen control-register map, sticky status bits, and command outputs. | Run `rtl/tb/tb_axi_lite_ctrl_slave.sv`. |
| `control/host_cmd_status_mgr.sv` | Concrete PC30 command/status DMA manager that fetches one command beat per launch, writes one launch/busy status beat on `START`, and writes one terminal status beat on done/error/stop. | Run `rtl/tb/tb_host_cmd_status_mgr.sv` and `rtl/tb/tb_kernel_top_smoke.sv`. |
| `control/prefill_decode_controller.sv` | Runtime controller for `wait command -> prompt read (prefill only) -> layer pass -> LM head/argmax -> token/stop`. | Run `rtl/tb/tb_prefill_decode_controller.sv` and `rtl/tb/tb_prefill_decode_smoke.sv`. |
| `control/layer_controller.sv` | Reused 22-layer controller that now emits the real per-layer `block_start/block_id/q_head_id/kv_head_id` schedule for the fixed decoder-layer block order. | Run `rtl/tb/tb_prefill_decode_controller.sv` and `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `control/stop_condition_unit.sv` | Stop-condition block for EOS, max-token, and host-abort termination. | Run `rtl/tb/tb_prefill_decode_controller.sv`. |

When a command in this folder produces simulator output, waveform dumps, or logs, put them under `sim/`.

## Phase 2 Memory And Buffer Skeleton

These production memory-path files now exist under `rtl/memory/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `memory/hbm_port_router.sv` | Fixed-function HBM arbitration and steering layer between internal DMA clients and the shared shell-side read/write path. | Run `rtl/tb/tb_hbm_port_router.sv`. |
| `memory/prompt_token_reader.sv` | Prompt token DMA reader that fetches token IDs from the host I/O region and emits a `token_bus` stream. | Run `rtl/tb/tb_prompt_token_reader.sv`. |
| `memory/generated_token_writer.sv` | Generated-token ring-buffer writer for host-visible token output in `PC30`. | Run `rtl/tb/tb_generated_token_writer.sv`. |
| `memory/weight_dma_reader.sv` | Decoder-layer weight DMA reader with descriptor-handshake gating, tensor-to-block tag mapping, and streamed multi-beat `wt_bus` output. | Run `rtl/tb/tb_weight_dma_reader.sv`. |
| `memory/embedding_lmhead_dma_reader.sv` | Shared multi-beat reader for embedding/gamma beat streams, LM-head weight beats, and aggregated scale metadata. | Run `rtl/tb/tb_embedding_lmhead_dma_reader.sv`. |
| `memory/kv_cache_dma_reader.sv` | K/V cache DMA reader with descriptor-handshake gating and streamed multi-beat `act_bus` output. | Run `rtl/tb/tb_kv_cache_dma_reader.sv`. |
| `memory/kv_cache_dma_writer.sv` | Quantized K/V cache writer with buffered write data and clean ready/valid behavior. | Run `rtl/tb/tb_kv_cache_dma_writer.sv`. |
| `memory/debug_dma_writer.sv` | Debug capture DMA writer stub for the dedicated debug buffer in `PC31`. | Covered by the Phase 2 syntax/integration compile. |
| `memory/scale_metadata_store.sv` | Multi-port scale metadata store for activation/KV/LM-head scale vectors. | Run `rtl/tb/tb_scale_metadata_store.sv`. |
| `memory/tile_buffer_bank.sv` | Generic ping/pong banked tile buffer used for memory/compute decoupling. | Run `rtl/tb/tb_tile_buffer_bank.sv`. |
| `memory/kv_cache_manager.sv` | Deterministic KV-cache descriptor/address generator for layer/head/token windows. | Run `rtl/tb/tb_kv_cache_manager.sv`. |

## Phase 3 Shared GEMM Compute Core

These production compute-path files now exist under `rtl/compute/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `compute/mac_lane.sv` | Signed INT8xINT8->INT32 MAC leaf used as the lane-level arithmetic reference for the shared engine. | Run `rtl/tb/tb_mac_lane.sv`. |
| `compute/accumulator_bank.sv` | 512-lane INT32 accumulator storage with explicit clear/load/tag update behavior. | Run `rtl/tb/tb_accumulator_bank.sv`. |
| `compute/requantize_unit.sv` | Bank-scaled INT32->INT8 requantizer using unsigned Q16.16 multipliers, round-to-nearest-even, and saturating clamp. | Run `rtl/tb/tb_requantize_unit.sv`, which now also consumes exported Phase 3 trace fixtures. |
| `compute/shared_gemm_engine.sv` | Output-stationary 512-lane shared GEMM core with clear-on-first-slice accumulation and buffered result snapshotting. | Run `rtl/tb/tb_shared_gemm_engine.sv`, which now replays an exported Phase 3 q-projection trace slice. |
| `compute/gemm_operand_router.sv` | Mode-driven operand selector that routes activation, weight, score, and KV tiles into the shared GEMM core and normalizes routed operand tags to the active GEMM block. | Run `rtl/tb/tb_gemm_operand_router.sv` and `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `compute/gemm_result_router.sv` | Mode-driven result router that emits quantized projection outputs and raw score/LM-head accumulators to the next stage with normalized active-mode output tags. | Run `rtl/tb/tb_gemm_result_router.sv` and `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `compute/gemm_op_scheduler.sv` | Deterministic tile-loop scheduler with a production block-driven mode for decoder-layer integration plus the retained legacy full-layer mode used by the isolated GEMM scheduler smoke bench. | Run `rtl/tb/tb_gemm_op_scheduler.sv` and `rtl/tb/tb_decoder_layer_smoke.sv`. |

## Phase 4 Attention-Path Leaves

These production attention-path files now also exist under `rtl/compute/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `compute/rope_cos_rom.memh` | Generated Q16.16 cosine table source for the RoPE ROM. | Regenerate it with `python model/export_fpga_vectors.py --phase phase4 --layer 0 --output-dir sim/golden_traces`. |
| `compute/rope_sin_rom.memh` | Generated Q16.16 sine table source for the RoPE ROM. | Regenerate it with `python model/export_fpga_vectors.py --phase phase4 --layer 0 --output-dir sim/golden_traces`. |
| `compute/rope_lut_rom.sv` | Token-major RoPE lookup ROM that expands one `8 x 64` head slice of sine/cosine values from the generated memh tables. | Covered by `rtl/tb/tb_rope_unit.sv`. |
| `compute/rope_unit.sv` | Real INT8 rotary datapath that applies TinyLlama's rotate-half RoPE to one `8 x 64` Q slice and one `8 x 64` K slice while preserving the input scale. | Run `rtl/tb/tb_rope_unit.sv`, which consumes exported Phase 4 traces. |
| `compute/gqa_router.sv` | Grouped-query attention routing leaf that validates `q_head -> kv_head` mapping and rewrites K/V tags for score vs weighted-sum use. | Run `rtl/tb/tb_gqa_router.sv`. |
| `compute/causal_mask_unit.sv` | Real pre-softmax score-mask leaf for one `8 x 64` score chunk using the frozen `MASK_NEG_INF` fill contract. | Run `rtl/tb/tb_causal_mask_unit.sv`, which consumes exported Phase 4 traces. |

## Phase 5 Nonlinear Wrappers

These production nonlinear-wrapper files now exist under `rtl/nonlinear/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `nonlinear/rmsnorm_wrapper.sv` | RTL wrapper that converts INT8 activation tiles plus FP16 gamma beats into the fixed-point `rmsnorm_core_hls` stream contract, then requantizes the normalized output tile back to INT8. | Regenerate the Phase 5 fixtures, then run `rtl/tb/tb_rmsnorm_wrapper.sv`. |
| `nonlinear/softmax_wrapper.sv` | RTL wrapper that dequantizes masked INT32 score tiles into Q16.16 chunks for `softmax_core_hls`, then emits the fixed probability scale and INT8 probability tile. | Regenerate the Phase 5 fixtures, then run `rtl/tb/tb_softmax_wrapper.sv`. |
| `nonlinear/silu_wrapper.sv` | RTL wrapper that dequantizes gate-projection INT8 tiles into Q16.16 chunks for `silu_core_hls`, then requantizes the SiLU output back to INT8. | Regenerate the Phase 5 fixtures, then run `rtl/tb/tb_silu_wrapper.sv`. |

## Phase 6 Embedding / FFN / LM-Head / Debug

These Phase 6 production compute files now exist under `rtl/compute/`.

| File | What it is | Smoke test |
|------|------------|------------|
| `compute/embedding_lookup.sv` | Embedding-row fetch controller that turns one token into one full `4096-byte` `TENSOR_EMBED` DMA request, assembles `128` `256-bit` beats, and emits one FP16 embedding row plus token metadata. | Regenerate the Phase 6 fixtures, then run `rtl/tb/tb_embedding_lookup.sv`. |
| `compute/embedding_quantizer.sv` | FP16 embedding-row quantizer that now quantizes one `N_TILE = 32` row-local feature slice per cycle during ingest, buffers per-row INT8 feature tiles, emits one `BLOCK_EMBED` scale vector, and streams `64` row-major INT8 activation tiles into the decoder datapath without the old `512`-way output-time divide/modulo fanout. | Regenerate the Phase 6 fixtures, then run `rtl/tb/tb_embedding_quantizer.sv`. |
| `compute/residual_add.sv` | Aligned INT32 residual leaf that captures matching `acc_bus` tiles, retags them as residual1/residual2, and sums active lanes with zero-filled tails. | Regenerate the Phase 6 fixtures, then run `rtl/tb/tb_residual_add.sv`. |
| `compute/elementwise_mul.sv` | Real SwiGLU leaf that captures SiLU and `up_proj` activation tiles, multiplies them lane-wise as `INT8 x INT8 -> INT32`, and forwards the result to the down-projection requantization path. | Regenerate the Phase 6 fixtures, then run `rtl/tb/tb_elementwise_mul.sv`. |
| `compute/lm_head_controller.sv` | Outer vocabulary-tile controller for LM head that reuses the one-tile GEMM scheduler 250 times, holds the final hidden-state context stable, and retags partial logits for the argmax path. | Run `rtl/tb/tb_lm_head_controller.sv`. |
| `compute/argmax_reduction.sv` | Greedy vocab reduction leaf with lower-token-id tie-break on exact logit ties. | Regenerate the Phase 6 fixtures, then run `rtl/tb/tb_argmax_reduction.sv`. |
| `compute/debug_capture_mux.sv` | Non-backpressuring debug-source selector that filters by layer/block selection and reports dropped captures when the downstream debug path is not ready. | Run `rtl/tb/tb_debug_capture_mux.sv`. |

## Phase 7 Decoder-Layer Integration

These Phase 7 integration files now harden the reused decoder-layer path.

| File | What it is | Smoke test |
|------|------------|------------|
| `control/layer_controller.sv` | Concrete per-layer orchestrator for the fixed `RMSNorm -> attention -> FFN -> requantize` block order, including the repeated `score -> mask -> softmax -> weighted_sum` attention loop across all `32` query heads. | Run `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `compute/gemm_op_scheduler.sv` | Production block-driven GEMM tile scheduler that expands one logical GEMM block at a time from `layer_controller.sv`. | Run `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `compute/gemm_operand_router.sv` | Integration-ready operand router used by the decoder-layer smoke path to validate the correct source pair for projection, score, and weighted-sum GEMMs. | Run `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `compute/gemm_result_router.sv` | Integration-ready result router used by the decoder-layer smoke path to validate quantized-vs-raw score/LM-head routing. | Run `rtl/tb/tb_decoder_layer_smoke.sv`. |
| `tb/tb_decoder_layer_smoke.sv` | Exported Phase 7 trace-backed smoke test for one concrete decoder-layer pass at real TinyLlama dimensions and prefill/decode token counts. | Regenerate the Phase 7 fixtures, then run `rtl/tb/tb_decoder_layer_smoke.sv`. |

## Phase 8 Runtime-Core Integration

These Phase 8 files now harden the top-level runtime shell around the reused
decoder-layer engine.

| File | What it is | Smoke test |
|------|------------|------------|
| `top/tinyllama_u55c_kernel_top.sv` | Runtime-core top-level with AXI-Lite control, normalized shell DMA read/write boundary, PC30 command/status handling, real prefill embedding ingress, generated-token writeback, and currently stubbed layer/LM/token closure beyond the embedding slice. | Run `rtl/tb/tb_kernel_top_smoke.sv`. |
| `top/runtime_embedding_frontend.sv` | First real-inference closure helper that fetches embedding scale metadata, issues embedding-row DMA requests, assembles FP16 rows, and emits INT8 embedding activation tiles. | Run `rtl/tb/tb_runtime_embedding_frontend.sv`. |
| `tb/tb_prefill_decode_smoke.sv` | Exported Phase 8 trace-backed runtime-control smoke for prefill plus a few decode steps using the real TinyLlama reference to generate expected token IDs and runtime counts. | Regenerate the Phase 8 fixtures, then run `rtl/tb/tb_prefill_decode_smoke.sv`. |
| `tb/tb_kernel_top_smoke.sv` | Exported Phase 8 top-level smoke for AXI-Lite launch, PC30 command fetch, prompt read beats, generated-token writes, final status payload, and interrupt observation against a fake shell DMA model. | Regenerate the Phase 8 fixtures, then run `rtl/tb/tb_kernel_top_smoke.sv`. |

## Phase 9 Runtime Acceptance And Shell Wrapper

These Phase 9 files harden the runtime shell with broader acceptance coverage
and the first platform-facing wrapper step around the normalized DMA boundary.

| File | What it is | Smoke test |
|------|------------|------------|
| `top/tinyllama_u55c_shell_wrapper.sv` | First shell-facing wrapper around `tinyllama_u55c_kernel_top.sv` with elastic read buffering plus coupled write-request buffering at the normalized shell DMA seam. | Run `rtl/tb/tb_shell_wrapper_smoke.sv`. |
| `tb/tb_kernel_top_acceptance.sv` | Exported Phase 9 acceptance bench for abort during `RUN_LAYERS`, relaunch, sticky-status clear, host-visible status words, and integrated command/prompt/writeback counts. | Regenerate the Phase 9 fixtures, then run `rtl/tb/tb_kernel_top_acceptance.sv`. |
| `tb/tb_shell_wrapper_smoke.sv` | Exported Phase 9 wrapper smoke that runs the runtime case through `tinyllama_u55c_shell_wrapper.sv` under shell-side backpressure. | Regenerate the Phase 9 fixtures, then run `rtl/tb/tb_shell_wrapper_smoke.sv`. |

## Current Integration Frontier

The repo has moved past the original Phase 9 runtime harness and is now in the
first real-inference closure slice.

What is now true:

- `top/runtime_embedding_frontend.sv` performs real prefill embedding ingress
  through `embedding_lmhead_dma_reader.sv`, `embedding_lookup.sv`, and the
  hardened `embedding_quantizer.sv`
- `top/tinyllama_u55c_kernel_top.sv` waits for that embedding ingress to finish
  before launching the reused layer path
- `tb_embedding_quantizer.sv`, `tb_runtime_embedding_frontend.sv`,
  `tb_kernel_top_smoke.sv`, `tb_kernel_top_acceptance.sv`, and
  `tb_shell_wrapper_smoke.sv` all pass against the current slice

What is still intentionally incomplete:

- `tinyllama_u55c_kernel_top.sv` still uses synthetic block/LM/token closure
  after the embedding slice
- the decoder datapath, final RMSNorm, real LM head, and real argmax are the
  next integration milestone

## Synthesis Readiness

The production RTL under `rtl/common/`, `rtl/control/`, and the later
subfolders is being written as synthesizable RTL for the FPGA.

That means:

- production modules should use synthesis-friendly constructs such as
  `always_ff`, `always_comb`, bounded `for` loops, enums, packed structs, and
  generate blocks
- testbenches under `rtl/tb/` are not synthesizable and are only for simulation
- passing a smoke test does **not** by itself prove Vivado/Vitis synthesis is
  clean

Use this checklist as we add new production RTL:

- module compiles in the shared syntax/integration check
- module has no simulation-only logic in the production path
  - no `initial` blocks for normal operation
  - exception: ROM initialization with generated memh source files is allowed
  - no `#` delays
  - no `wait`, `fork/join`, or testbench-only tasks
- loop variables and procedural state are local and unambiguous across
  processes
- widths, casts, signedness, and enum usage follow `rtl/common/tinyllama_pkg.sv`
  and `rtl/common/tinyllama_bus_pkg.sv`
- any local scratch/output artifacts from checks go under `sim/`
- module has at least one smoke test or is covered by a higher-level syntax
  integration command
- later, before calling a block synthesis-ready in the strongest sense, it still
  needs a real vendor-tool synthesis pass

Current vendor-tool checkpoint status for the post-Phase-9 slice:

- `embedding_quantizer.sv` now synthesizes cleanly in Vivado as a leaf top
- `runtime_embedding_frontend.sv` now synthesizes cleanly as the next parent
- `tinyllama_u55c_kernel_top.sv` now synthesizes cleanly as the current runtime
  top
- the shell-wrapper rerun after the current quantizer/frontend hardening is
  still pending

One important interpretation note:

- the current kernel-top synth is structurally useful, but its utilization is
  artificially low because the emitted embedding activation/scale payloads are
  not yet consumed by the downstream decoder datapath

## Legacy Validation Files

These root-level files are older validation infrastructure. They are still useful, but they do not define the final TinyLlama production architecture.

| File | What it is | Smoke test |
|------|------------|------------|
| `control_fsm.sv` | Legacy tiled matrix-vector control FSM. | Run `tb_control_fsm.sv`. |
| `top.sv` | Legacy top-level wrapper connecting the FSM to the MAC array and BRAM stubs. | Run `tb_top.sv`. |
| `mac_array.sv` | Legacy 8-lane INT8 MAC array with accumulators. | Run `tb_mac_array.sv` or `tb_top.sv`. |
| `mac_unit.sv` | Legacy single-lane combinational INT8 multiply-accumulate unit. | Covered by `tb_mac_array.sv` and `tb_top.sv`. |
| `tb_control_fsm.sv` | Legacy FSM-only testbench. | See command below. |
| `tb_top.sv` | Legacy end-to-end top-level testbench. | See command below. |
| `tb_mac_array.sv` | Legacy MAC-array testbench that uses generated vectors. | See command below. |

### Legacy Smoke Tests

All commands below run from the project root.

FSM-only smoke test:

```powershell
iverilog -g2012 -o sim/tb_control_fsm.vvp rtl/control_fsm.sv rtl/tb_control_fsm.sv
vvp sim/tb_control_fsm.vvp
```

Legacy top-level smoke test:

```powershell
iverilog -g2012 -o sim/tb_top.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/control_fsm.sv rtl/top.sv rtl/tb_top.sv
vvp sim/tb_top.vvp
```

Legacy MAC-array smoke test:

```powershell
iverilog -g2012 -o sim/tb_mac_array.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/tb_mac_array.sv
vvp sim/tb_mac_array.vvp
```

This test expects local vector files under `sim/`.

## Where To Look Next

- For production testbenches, see [rtl/tb/README.md](tb/README.md).
- For HLS common utilities and header-only smoke checks, see [hls/README.md](../hls/README.md).
