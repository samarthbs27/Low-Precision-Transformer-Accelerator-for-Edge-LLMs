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
| `control/host_cmd_status_mgr.sv` | Stub PC30 command/status DMA manager that fetches one command beat per launch and writes one status beat on terminal events. | Run `rtl/tb/tb_host_cmd_status_mgr.sv`. |
| `control/prefill_decode_controller.sv` | Runtime controller stub for `IDLE -> prefill/decode layer pass -> LM head -> token/stop -> DONE`. | Run `rtl/tb/tb_prefill_decode_controller.sv`. |
| `control/layer_controller.sv` | Reused 22-layer loop controller that iterates `layer_id = 0..21` for one full decoder pass. | Run `rtl/tb/tb_prefill_decode_controller.sv`. |
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
| `compute/gemm_operand_router.sv` | Mode-driven operand selector that routes activation, weight, score, and KV tiles into the shared GEMM core. | Run `rtl/tb/tb_gemm_operand_router.sv`. |
| `compute/gemm_result_router.sv` | Mode-driven result router that emits quantized projection outputs and raw score/LM-head accumulators to the next stage. | Run `rtl/tb/tb_gemm_result_router.sv`. |
| `compute/gemm_op_scheduler.sv` | Deterministic tile-loop scheduler for the GEMM-backed decoder-layer operations and LM-head-only mode. | Run `rtl/tb/tb_gemm_op_scheduler.sv`. |

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
