# TinyLlama U55C FPGA Inference Accelerator

> A mixed-precision FPGA implementation target for TinyLlama 1.1B on the Xilinx Alveo U55C, with prompt prefill, autoregressive decode, HBM-resident KV cache, and a reused INT8 decoder-layer engine.

---

## Project Overview

This repository is centered on implementing TinyLlama inference on the U55C.
The current architecture is no longer a single decoder-layer demo or a host-loop
prototype. The intended runtime is:

- host tokenizes the prompt and launches the FPGA
- FPGA performs embedding lookup, prefill, decode, KV-cache updates, final RMSNorm, LM head, and greedy argmax
- FPGA emits generated token ids until EOS or `max_new_tokens`
- host reads token ids back and detokenizes them for display

The implementation strategy is reuse-heavy:

- one TinyLlama decoder-layer engine reused for all 22 layers
- one shared INT8 GEMM engine reused across Q, K, V, attention scores, weighted sum, O, gate, up, down, and LM head
- HBM used as the backing store for weights, KV cache, and debug buffers

The current repo frontier is the first post-Phase-9 real-inference closure
slice:

- `runtime_embedding_frontend.sv` now performs real prefill and decode-token
  embedding ingress
- `runtime_decoder_datapath.sv` now consumes those embedding outputs and drives
  TinyLlama-scale block completion across all 22 layers, including a coherent
  FFN side chain that now uses the real `shared_gemm_engine.sv` on
  `BLOCK_GATE`, `BLOCK_UP`, and `BLOCK_DOWN`, the real `silu_wrapper.sv` leaf
  on `BLOCK_SILU`, the real `elementwise_mul.sv` leaf on `BLOCK_GLU_MUL`, and
  a coherent `BLOCK_WEIGHTED_SUM -> BLOCK_O -> BLOCK_RESIDUAL1` attention
  output chain where `BLOCK_O` now also uses the real
  `shared_gemm_engine.sv`
- `runtime_final_rmsnorm_tail.sv` now fetches real final RMSNorm gamma and
  runs the real `rmsnorm_wrapper.sv` inside the runtime top-level
- `runtime_lm_head_tail.sv` now wraps the real `lm_head_controller.sv`,
  `shared_gemm_engine.sv`, and `argmax_reduction.sv` on top of that
  post-RMSNorm stream
- `tinyllama_u55c_kernel_top.sv` waits for completed embedding ingress before
  launching or relaunching the reused layer path
- the top-level now uses the real final-RMSNorm wrapper plus the real
  LM-head DMA/shared-GEMM/controller/argmax tail, but the decoder's current
  final-hidden stream is still a deterministic scaffold rather than the final
  math path

---

## Finalized Architecture

| Item | Decision |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Layers | `22` reused decoder layers |
| Attention | `32` Q heads, `4` KV heads, `head_dim = 64` |
| Hidden size | `2048` |
| FFN size | `5632` |
| Runtime modes | prompt `prefill` + autoregressive `decode` |
| GEMM precision | `INT8 x INT8 -> INT32` |
| Higher-precision blocks | RMSNorm, RoPE, softmax, final RMSNorm |
| KV cache | symmetric `INT8` in HBM, per-layer/per-KV-head scales |
| Token selection | FPGA-side greedy argmax |
| U55C memory model | `16 GB` HBM, `32` pseudo-channels |
| Production compute width | `512` INT8 lanes |

The authoritative implementation spec is [docs/design_decisions.txt](docs/design_decisions.txt).

---

## Repository Structure

```text
Project/
  README.md
  docs/
    README.md
    design_decisions.txt
    modules.md
    implementation_checklist.md
    golden_trace_plan.md
    u55c_bringup_checklist.md
    real_inference_closure_plan.md
    Installation_guide_ug1468-alveo-u55c.pdf
    parallelism_tradeoffs.md
    block_diagram.drawio
    block_diagram.md
    block_diagram.png
    theory.md
  model/
    README.md
    tinyllama.py
    tinyllama_gemm_int8.py
    export_fpga_vectors.py
    model.py
    gen_test_vectors.py
  rtl/
    README.md
    common/
      tinyllama_pkg.sv
      tinyllama_bus_pkg.sv
      stream_fifo.sv
      skid_buffer.sv
      descriptor_fifo.sv
    control/
      axi_lite_ctrl_slave.sv
      kernel_reg_file.sv
      host_cmd_status_mgr.sv
      prefill_decode_controller.sv
      layer_controller.sv
      stop_condition_unit.sv
    memory/
      hbm_port_router.sv
      prompt_token_reader.sv
      generated_token_writer.sv
      weight_dma_reader.sv
      embedding_lmhead_dma_reader.sv
      kv_cache_dma_reader.sv
      kv_cache_dma_writer.sv
      debug_dma_writer.sv
      scale_metadata_store.sv
      tile_buffer_bank.sv
      kv_cache_manager.sv
    compute/
      embedding_lookup.sv
      embedding_quantizer.sv
      mac_lane.sv
      accumulator_bank.sv
      requantize_unit.sv
      shared_gemm_engine.sv
      gemm_operand_router.sv
      gemm_result_router.sv
      gemm_op_scheduler.sv
      rope_lut_rom.sv
      rope_unit.sv
      gqa_router.sv
      causal_mask_unit.sv
      residual_add.sv
      elementwise_mul.sv
      lm_head_controller.sv
      argmax_reduction.sv
      debug_capture_mux.sv
    nonlinear/
      rmsnorm_core_hls_ip.sv
      rmsnorm_wrapper.sv
      softmax_wrapper.sv
      silu_wrapper.sv
    top/
      runtime_embedding_frontend.sv
      runtime_decoder_datapath.sv
      runtime_final_rmsnorm_tail.sv
      runtime_lm_head_tail.sv
      tinyllama_u55c_kernel_top.sv
      tinyllama_u55c_shell_wrapper.sv
    tb/
      README.md
      tb_stream_fifo.sv
      tb_descriptor_fifo.sv
      tb_axi_lite_ctrl_slave.sv
      tb_host_cmd_status_mgr.sv
      tb_prefill_decode_controller.sv
      tb_hbm_port_router.sv
      tb_tile_buffer_bank.sv
      tb_prompt_token_reader.sv
      tb_generated_token_writer.sv
      tb_scale_metadata_store.sv
      tb_kv_cache_manager.sv
      tb_weight_dma_reader.sv
      tb_kv_cache_dma_reader.sv
      tb_kv_cache_dma_writer.sv
      tb_embedding_lmhead_dma_reader.sv
      tb_mac_lane.sv
      tb_accumulator_bank.sv
      tb_requantize_unit.sv
      tb_shared_gemm_engine.sv
      tb_gemm_operand_router.sv
      tb_gemm_result_router.sv
      tb_gemm_op_scheduler.sv
      tb_rope_unit.sv
      tb_gqa_router.sv
      tb_causal_mask_unit.sv
      tb_rmsnorm_wrapper.sv
      tb_softmax_wrapper.sv
      tb_silu_wrapper.sv
      tb_elementwise_mul.sv
      tb_lm_head_controller.sv
      tb_argmax_reduction.sv
      tb_debug_capture_mux.sv
      tb_decoder_layer_smoke.sv
      tb_prefill_decode_smoke.sv
      tb_runtime_embedding_frontend.sv
      tb_runtime_decoder_datapath.sv
      tb_runtime_final_rmsnorm_tail.sv
      tb_runtime_lm_head_tail.sv
      tb_kernel_top_smoke.sv
      tb_kernel_top_acceptance.sv
      tb_shell_wrapper_smoke.sv
    control_fsm.sv
    top.sv
    mac_unit.sv
    mac_array.sv
    tb_control_fsm.sv
    tb_top.sv
    tb_mac_array.sv
  hls/
    README.md
    common/
      fixed_types.hpp
      stream_utils.hpp
      nonlinear_math.hpp
      test_memh.hpp
    rmsnorm/
      rmsnorm_core_hls.hpp
      rmsnorm_core_hls.cpp
      tb_rmsnorm.cpp
    softmax/
      softmax_core_hls.hpp
      softmax_core_hls.cpp
      tb_softmax.cpp
    silu/
      silu_core_hls.hpp
      silu_core_hls.cpp
      tb_silu.cpp
  sim/
```

---

## Where To Read Next

- [docs/design_decisions.txt](docs/design_decisions.txt)
  Final implementation choices for quantization, HBM allocation, controller behavior, and runtime ownership.

- [docs/block_diagram.drawio](docs/block_diagram.drawio)
  Editable architecture diagram for the full TinyLlama accelerator.

- [docs/block_diagram.md](docs/block_diagram.md)
  Text explanation of the full post-Phase-1 architecture plus the legacy GEMM validation core.

- [docs/modules.md](docs/modules.md)
  Physical module inventory for the U55C implementation, including the RTL/HLS split, bus plan, and required blocks.

- [docs/implementation_checklist.md](docs/implementation_checklist.md)
  File-by-file coding plan for the new TinyLlama RTL/HLS implementation, including dependencies, stub order, and first verification targets.

- [docs/golden_trace_plan.md](docs/golden_trace_plan.md)
  Real-model trace export and verification policy, including when trace-backed tests become mandatory.

- [docs/u55c_bringup_checklist.md](docs/u55c_bringup_checklist.md)
  Step-by-step Linux/Vivado/Vitis bring-up guide for the current Phase 9
  runtime core, including the first concrete Vivado synthesis checklist.

- [docs/real_inference_closure_plan.md](docs/real_inference_closure_plan.md)
  Concrete closure plan for replacing the current top-level runtime stubs with
  a true TinyLlama token-generating inference path.

- [docs/parallelism_tradeoffs.md](docs/parallelism_tradeoffs.md)
  Design rationale for where the current architecture is using FPGA
  parallelism aggressively and where it is intentionally conservative.

- [model/README.md](model/README.md)
  Software reference path, TinyLlama NumPy inference, and GEMM-only INT8 bridge.

- [rtl/README.md](rtl/README.md)
  Existing RTL validation core and simulation entry points.

---

## Current Implementation Status

- `model/` contains the TinyLlama software reference and the GEMM-only INT8 analysis/generation bridge.
- `rtl/` contains the existing validation core for the shared GEMM engine and control FSM.
- `rtl/common/`, `rtl/tb/`, and `hls/common/` now contain the verified Phase 0 production foundation.
- `rtl/control/` now contains the verified Phase 1 control-plane skeleton, now promoted through the Phase 8 runtime path for real command-aware launch sequencing and PC30 status writeback behavior.
- `rtl/memory/` now contains the hardened Phase 2 memory/DMA/buffer layer, including verified router, prompt I/O, generated-token I/O, multi-beat weight/KV readers, buffered KV writeback, scale-store, KV-address, and tile-buffer smoke tests.
- `rtl/compute/` now contains the hardened Phase 3 shared GEMM compute layer, including the MAC leaf, accumulator bank, bank-scaled requantizer, shared engine, operand/result routers, and deterministic GEMM scheduler.
- `rtl/compute/` now also contains the concrete Phase 4 attention-path leaves: the generated RoPE ROM, the real rotary datapath, the GQA router, and the pre-softmax causal-mask unit.
- `hls/rmsnorm/`, `hls/softmax/`, `hls/silu/`, and `rtl/nonlinear/` now contain the hardened Phase 5 nonlinear implementation:
  - fixed-point RMSNorm, softmax, and SiLU HLS kernels
  - verified host-side C++ smoke tests
  - verified RTL wrappers that consume exported Phase 5 trace fixtures
- `model/export_fpga_vectors.py` now provides the canonical golden-trace export entry point for both:
  - Phase 3 arithmetic verification
  - Phase 4 attention-path verification
  - Phase 5 nonlinear verification
  - full Phase 6 embedding/FFN/LM-head trace-backed verification
- `rtl/compute/` now also contains the complete verified Phase 6 set:
  - `embedding_lookup.sv`
  - `embedding_quantizer.sv`
  - `residual_add.sv`
  - `elementwise_mul.sv`
  - `lm_head_controller.sv`
  - `argmax_reduction.sv`
  - `debug_capture_mux.sv`
  - local smoke tests for all seven
- `rtl/control/layer_controller.sv`, `rtl/compute/gemm_op_scheduler.sv`,
  `rtl/compute/gemm_operand_router.sv`, and `rtl/compute/gemm_result_router.sv`
  now also include the concrete Phase 7 decoder-layer integration behavior:
  - block-level layer sequencing across the fixed 19-step decoder order
  - production block-driven GEMM scheduling
  - normalized active-mode operand/result routing
  - trace-backed decoder-layer smoke verification via
    `rtl/tb/tb_decoder_layer_smoke.sv`
- `model/export_fpga_vectors.py` writes canonical traces under `sim/golden_traces/`, emits packed `.memh` fixtures for the RTL benches, and regenerates the tracked RoPE ROM memh files under `rtl/compute/`.
- `model/export_fpga_vectors.py` now also exports Phase 7 decoder-layer
  schedule fixtures for prefill and decode under `sim/golden_traces/phase7/rtl/`.
- `rtl/top/tinyllama_u55c_kernel_top.sv` now provides the concrete Phase 8
  runtime-core top-level:
  - AXI-Lite launch/status
  - normalized shell DMA read/write boundary
  - PC30 command/status integration
  - prompt read and generated-token writeback plumbing
  - structural prompt-prefill/decode runtime bring-up
- `rtl/tb/tb_prefill_decode_smoke.sv` and `rtl/tb/tb_kernel_top_smoke.sv`
  now provide the Phase 8 runtime gates:
  - exported real-model prefill/decode runtime fixtures
  - controller-level prompt/layer/token sequencing checks
  - top-level AXI-Lite launch plus fake shell-DMA host-I/O smoke
- `model/export_fpga_vectors.py` now also exports Phase 8 runtime fixtures
  under `sim/golden_traces/phase8/rtl/`.
- `rtl/top/tinyllama_u55c_shell_wrapper.sv` now provides the first Phase 9
  platform-facing wrapper step around the runtime core:
  - same external normalized shell DMA boundary as Phase 8
  - elastic read buffering at the shell seam
  - coupled write-request buffering that preserves the router write contract
- `rtl/tb/tb_kernel_top_acceptance.sv` and `rtl/tb/tb_shell_wrapper_smoke.sv`
  now provide the first Phase 9 acceptance gates:
  - top-level abort/relaunch/status acceptance against exported Phase 9
    expectations
  - wrapper smoke under shell-side backpressure
- `model/export_fpga_vectors.py` now also exports Phase 9 runtime-acceptance
  fixtures under `sim/golden_traces/phase9/rtl/`.
- Post-Phase-9 real inference closure has started:
  - `rtl/top/runtime_embedding_frontend.sv` now fetches real embedding-output
    scale metadata plus real embedding rows during prefill and decode relaunch
- `rtl/top/runtime_decoder_datapath.sv` now consumes those embedding outputs,
    drives TinyLlama-scale block completion across all 22 layers, carries one
    tile coherently through `GATE -> UP -> SILU -> GLU_MUL -> DOWN`, now uses
    the real `shared_gemm_engine.sv` on `BLOCK_GATE`, `BLOCK_UP`, and
    `BLOCK_DOWN`, the real `silu_wrapper.sv` leaf on `BLOCK_SILU`, the real
    `elementwise_mul.sv` leaf on `BLOCK_GLU_MUL`, stages the synthetic
    weighted-sum output, routes `BLOCK_O` through the real
    `shared_gemm_engine.sv`, applies `BLOCK_RESIDUAL1` on that staged O tile,
    and emits the accumulated final-hidden stream at the
    `BLOCK_FINAL_RMSNORM` boundary
  - `rtl/top/runtime_final_rmsnorm_tail.sv` now fetches real final RMSNorm
    gamma, runs the real `rmsnorm_wrapper.sv`, and turns that stream into a
    post-RMSNorm runtime activation path
  - `rtl/top/runtime_lm_head_tail.sv` now consumes that post-RMSNorm stream,
    fetches real `TENSOR_LM_HEAD` weights, runs the reused
    `shared_gemm_engine.sv`, and then drives the real
    `lm_head_controller.sv` + `argmax_reduction.sv` path
  - `rtl/top/tinyllama_u55c_kernel_top.sv` now gates prefill completion on
    finished embedding ingress rather than only prompt-token fetch, and loops
    non-stopping emitted tokens back through the same ingress path before each
    decode pass
  - the current normalized-shell runtime harness now uses:
    - embedding rows at `0x0000_0000_1000_0000`
    - embedding-output scale metadata at `0x0000_0000_0400_0000`
    - final RMSNorm gamma at `0x0000_0000_0800_0000`
    - LM-head weights at `0x0000_0000_2000_0000`
  - `rtl/compute/embedding_quantizer.sv` now uses the current vendor-hardened
    microarchitecture: quantize one `N_TILE = 32` row-local feature slice per
    cycle during ingest, buffer INT8 feature tiles, then assemble the emitted
    512-lane batch tiles
- Current local regression frontier after that rework is green:
  - `tb_rmsnorm_wrapper.sv`
  - `tb_embedding_quantizer.sv`
  - `tb_elementwise_mul.sv`
  - `tb_lm_head_controller.sv`
  - `tb_argmax_reduction.sv`
  - `tb_runtime_embedding_frontend.sv`
  - `tb_runtime_decoder_datapath.sv`
  - `tb_runtime_final_rmsnorm_tail.sv`
  - `tb_runtime_lm_head_tail.sv`
  - `tb_kernel_top_smoke.sv`
  - `tb_kernel_top_acceptance.sv`
  - `tb_shell_wrapper_smoke.sv`
- the fresh top-level runtime proofs for the current heavier decoder slice are
  now captured under:
  - `sim/logs/xsim_tb_kernel_top_smoke_*`
  - `sim/logs/xsim_tb_kernel_top_acceptance_*`
  - `sim/logs/xsim_tb_shell_wrapper_smoke_*`
  because the added projection-GEMM work makes the full top-level benches
  impractically slow under Icarus
- Current Vivado synthesis checkpoints are also materially better:
  - `embedding_quantizer.sv` synthesizes cleanly as a leaf top
  - `runtime_embedding_frontend.sv` synthesizes cleanly as the next parent
  - `tinyllama_u55c_kernel_top.sv` synthesizes cleanly as the current runtime
    top
  - the new `runtime_final_rmsnorm_tail.sv` and `runtime_lm_head_tail.sv`
    runtime slices still need their own Vivado checkpoints
  - `rtl/nonlinear/rmsnorm_core_hls_ip.sv` is currently a repo-owned local
    simulation model for Icarus, not a new vendor-synthesis milestone
- `docs/` now describe the full TinyLlama prefill/decode accelerator that the project is building toward.

The repo now contains hardened production modules through the current
post-Phase-9 decoder-closure slice, including the decoder-datapath scaffold
that replaced the old one-cycle block-completion stub, routes the FFN-side
`GATE`, `UP`, `SILU`, `GLU_MUL`, and `DOWN` blocks through the real projection
and nonlinear leaves, and feeds the new runtime final-RMSNorm helper that sits
in front of token selection. The next milestone is replacing the remaining
deterministic attention-side and residual-side decoder scaffold with the actual
shared decoder tail that feeds the now-real runtime LM-head DMA/shared-GEMM
path. The raw `m_axi_pc00..pc31` wrapper and the full Vitis/platform packaging
flow remain the outer follow-on step after that.

One practical note: the production RTL we are writing is intended to be
synthesizable, but a passing Icarus smoke test is only the first gate. The
current synthesis-readiness checklist lives in [rtl/README.md](rtl/README.md)
and should be used alongside simulation results as the implementation grows.
