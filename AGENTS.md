# AGENTS.md - Codex Project Context

This file is the Codex-facing context and instruction file for this repository.
It should stay short enough to be useful, but complete enough that a new coding
session can quickly recover the current project state.

If this file ever conflicts with the detailed design docs, follow the precedence
rules in Section 1 and then update this file so it stops drifting.

Last updated: 2026-04-28

---

## 1. Source Of Truth Order

Use the repository docs in this order:

1. `docs/design_decisions.txt`
   Final implementation decisions for the TinyLlama U55C accelerator.
2. `docs/modules.md`
   Full module inventory, interface plan, RTL/HLS split, and frozen
   microarchitecture notes.
3. `docs/implementation_checklist.md`
   File-by-file coding order, planned source tree, dependencies, and first-pass
   verification plan.
4. `docs/golden_trace_plan.md`
   Trace-backed verification policy and export format for real TinyLlama cases.
5. `docs/block_diagram.drawio` and `docs/block_diagram.md`
   Visual and textual system architecture reference.
6. `model/README.md`
   Software reference path and quantized GEMM bridge behavior.
7. `README.md` and `docs/README.md`
   Project navigation and high-level overview.

If any of the above conflict, prefer the higher item in the list.

Historical note:

- `CLAUDE.md` exists, but it contains older project history and may lag the
  current spec. Do not treat it as the primary design authority.

---

## 2. Project Snapshot

This repository is building a mixed-precision FPGA inference accelerator for:

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Platform: Xilinx Alveo U55C
- Runtime: prompt `prefill` + autoregressive `decode`
- Control split:
  - host: tokenization, launch, prompt upload, generated-token readback
  - FPGA: embedding lookup, prefill, 22-layer execution, KV cache, final
    RMSNorm, LM head, greedy argmax, stop conditions

This is no longer a Pythia project and no longer a single-layer toy pipeline.
The target is full TinyLlama inference behavior on FPGA using one reused
decoder-layer engine and one reused shared GEMM engine.

---

## 3. Fixed Architecture Summary

These values are currently frozen:

- `N_LAYERS = 22`
- `D_MODEL = 2048`
- `D_FF = 5632`
- `N_Q_HEADS = 32`
- `N_KV_HEADS = 4`
- `HEAD_DIM = 64`
- `VOCAB_SIZE = 32000`
- `MAX_POS = 2048`
- `SEQ_TILE = 64`
- `GEMM_LANES (target) = 512`
- `GEMM_LANES (current synth/hw profile) = 64` (Vivado 2022.1 wide packed-struct synthesis limitation)

Decoder layer structure:

- RMSNorm
- Q / K / V projections
- RoPE on Q and K
- attention scores
- causal mask
- softmax
- weighted sum with V
- O projection
- residual add
- RMSNorm
- gate / up projections
- SiLU
- elementwise multiply
- down projection
- residual add

Reuse model:

- one physical decoder-layer engine reused across all 22 layers
- one shared GEMM engine reused across:
  - Q
  - K
  - V
  - score
  - weighted sum
  - O
  - gate
  - up
  - down
  - LM head

Attention model:

- TinyLlama GQA
- 32 query heads
- 4 KV heads
- 8 Q-head groups per KV head

---

## 4. Precision And Quantization Contract

Quantized GEMM-heavy blocks:

- embedding output path into the decoder datapath
- Q / K / V / O projections
- attention score computation
- weighted sum with V
- gate / up / down projections
- LM head projection

Quantized arithmetic:

- activations: INT8
- weights: INT8
- accumulation: INT32

Higher-precision blocks:

- RMSNorm
- RoPE
- softmax
- final RMSNorm

Other fixed points:

- residual accumulation stays INT32 before requantization
- quantization is symmetric INT8 with zero point 0
- weights use static per-output-channel scales
- activations use static per-tensor scales per block output
- external scale metadata buses and stored scale payloads are unsigned
- RTL-visible requantization scales use unsigned Q16.16 multipliers
- one scale entry applies to one 32-lane bank slice of the 512-lane datapath
- KV cache uses static per-layer, per-KV-head scales, separate for K and V

---

## 5. Memory And Runtime Contract

U55C memory model:

- 16 GB HBM
- 32 pseudo-channels

HBM allocation:

- `PC00-PC15`: decoder-layer weights
- `PC16-PC17`: token embeddings, final RMSNorm gamma, quantization metadata
- `PC18-PC21`: LM-head weights
- `PC22-PC25`: K cache
- `PC26-PC29`: V cache
- `PC30`: host command block, prompt token list, generated-token ring buffer,
  status block
- `PC31`: debug capture buffer

KV cache:

- symmetric INT8 in HBM
- layout: `[layer][kv_head][token][head_dim]`
- decode reads cached prefix and appends one K row and one V row per layer

Sampling:

- greedy argmax only in the initial implementation
- no top-k or top-p in the production-first bring-up

Control-register map:

- AXI-Lite is fixed to 32-bit words
- frozen word map:
  - 0: `CONTROL`
  - 1: `STATUS`
  - 2/3: `CMD_BASE_LO/HI`
  - 4/5: `STATUS_BASE_LO/HI`
  - 6/7: `DEBUG_BASE_LO/HI`
  - 8: `PROMPT_TOKEN_COUNT`
  - 9: `MAX_NEW_TOKENS`
  - 10: `EOS_TOKEN_ID`
  - 11: `DEBUG_CFG`
  - 12: `GENERATED_TOKEN_COUNT`
  - 13: `LAST_TOKEN_ID`
  - 14: `CURRENT_LAYER`
  - 15: `CURRENT_BLOCK`
  - 16: `RTL_VERSION`
- `CONTROL` bits:
  - bit 0: start pulse
  - bit 1: mode (`0=prefill launch`, `1=decode-only launch`)
  - bit 2: abort request
- `STATUS` bits:
  - bit 0: busy
  - bit 1: done sticky
  - bit 2: error sticky
  - bit 3: stop-valid sticky
  - bits `[6:4]`: stop reason
  - bits `[11:8]`: error code
- launch semantics:
  - mode 0 + `max_new_tokens=0` => prefill-only
  - mode 0 + `max_new_tokens>0` => prefill then decode loop
  - mode 1 => decode-only from an existing KV cache

PC30 command/status block layout:

- one 256-bit command beat at `cmd_base_addr`
- one 256-bit status beat at `status_base_addr`
- command words:
  - 0/1: prompt token list base lo/hi
  - 2/3: generated-token ring base lo/hi
  - 4: generated-token capacity
  - 5..7: reserved
- status words:
  - 0: STATUS bits using the same packing as the AXI-Lite status register
  - 1: generated token count
  - 2: last token id
  - 3: current layer
  - 4: current block
  - 5..6: reserved
  - 7: RTL version

Internal DMA/HBM contract:

- read path uses descriptor valid/ready followed by beat data valid/ready
- write path uses descriptor valid/ready plus write-data valid/ready
- shared `dma_desc_t` fields are:
  - `region`
  - `tensor_id`
  - `write_not_read`
  - `pseudo_channel`
  - `addr`
  - `burst_len`
  - `byte_count`
  - `layer_id`
  - `kv_head_id`
  - `tile_id`
- `dma_desc_t.byte_count` is fixed to 32 bits because KV-cache transfers can
  exceed 65535 bytes
- current Phase 8 user-RTL top-level boundary is:
  - AXI-Lite control/status
  - normalized shell DMA read descriptor/data handshake
  - normalized shell DMA write descriptor/data handshake
- the actual platform-facing `m_axi_pc00..31` shell binding is treated as a
  wrapper concern around that normalized shell DMA boundary

---

## 6. Frozen Microarchitecture Highlights

These are already frozen in `docs/modules.md` and should not be re-decided
during implementation unless the docs are deliberately revised.

- `DMA_BEAT_W = 256`
- `M_TILE (target) = 16`
- `M_TILE (current synth/hw profile) = 2`
- `N_TILE = 32`
- `K_TILE = 64`
- `SCORE_Q_TILE = 16`
- `SCORE_K_TILE = 64`
- `VOCAB_TILE (target) = 128`
- `VOCAB_TILE (current synth/hw profile) = 64`
- `HEAD_GROUP_PAR = 1`
- `ROPE_CHUNK_TOKENS = 8`
- `SCORE_ROWS_PER_CHUNK = 8`
- `SCORE_CHUNKS_PER_TILE = 2`
- `FIXED_POINT_FMT = Q16.16`
- `TILE_BUFFER_BANKS = 16`
- `STREAM_FIFO_DEPTH = 4`
- `SKID_BUFFER_DEPTH = 2`
- `DESC_FIFO_DEPTH = 8`
- `DEBUG_FIFO_DEPTH = 32`
- `MASK_NEG_INF = -1000000000`

Other important frozen choices:

- one-tile lookahead prefetch for weights, KV cache, and LM head
- ping/pong buffering is mandatory
- all major ready/valid boundaries are elastic
- debug capture must never stall compute
- token embeddings are stored in HBM as FP16 row-major vectors and quantized
  after fetch
- score-mask chunks are fixed to `8 query rows x 64 key columns`
- RoPE slices are fixed to `8 tokens x 64 dims`
- RoPE ROM microarchitecture is explicit `8 positions x 32 angle values`
  broadcast across the paired 64 dimensions
- score-chunk `elem_count` represents `query_row_count * 64`; tail key columns
  are represented by explicit mask fill, not ragged elem-count geometry
- RoPE preserves the incoming Q/K activation scale; no extra RoPE-only scale
  metadata path exists
- Q and K supplied to one RoPE invocation must share the same `token_base`
- shared GEMM lane packing is row-major over one `M_TILE x N_TILE` output tile:
  `lane = m_local * N_TILE + n_local`

---

## 7. RTL / HLS Split

SystemVerilog RTL:

- top-level kernel
- AXI-Lite control
- register file
- command/status manager
- prefill/decode controller
- layer controller
- stop-condition unit
- HBM router
- DMA readers/writers
- scale metadata store
- tile buffer bank
- KV cache manager
- embedding lookup / quantizer
- GEMM scheduler
- shared GEMM engine
- MAC lane
- accumulator bank
- operand/result routers
- RoPE LUT and datapath
- GQA router
- causal mask
- residual add
- requantize unit
- elementwise multiply
- LM-head controller
- argmax reduction
- debug capture mux
- utility FIFOs and buffers

HLS C++:

- RMSNorm core
- softmax core
- SiLU core

RTL wrappers:

- RMSNorm wrapper
- softmax wrapper
- SiLU wrapper

---

## 8. Planned Source Tree For New Production Code

The new TinyLlama implementation is planned under:

- `rtl/common/`
- `rtl/control/`
- `rtl/memory/`
- `rtl/compute/`
- `rtl/nonlinear/`
- `rtl/top/`
- `rtl/tb/`
- `hls/common/`
- `hls/rmsnorm/`
- `hls/softmax/`
- `hls/silu/`

Important:

- Do not repurpose the legacy flat `rtl/` demo files as the production
  TinyLlama runtime.
- Keep the old root-level files as validation infrastructure unless there is an
  explicit migration step.

Legacy validation files still present:

- `rtl/control_fsm.sv`
- `rtl/top.sv`
- `rtl/mac_unit.sv`
- `rtl/mac_array.sv`
- `rtl/tb_control_fsm.sv`
- `rtl/tb_top.sv`
- `rtl/tb_mac_array.sv`

These are useful references, but they are not the final TinyLlama runtime.

---

## 9. Current Implementation Status

Software side:

- `model/tinyllama.py` is the NumPy TinyLlama reference path
- `model/tinyllama_gemm_int8.py` is the GEMM-only quantized bridge with
  `analysis` and `generate` modes

Documentation side:

- system architecture, module inventory, microarchitecture, and file-by-file
  checklist are all now written

Hardware side:

- the old validation core exists
- the new production TinyLlama source tree has started
- Phase 0 foundation is now in place:
  - `rtl/common/tinyllama_pkg.sv`
  - `rtl/common/tinyllama_bus_pkg.sv`
  - `rtl/common/stream_fifo.sv`
  - `rtl/common/skid_buffer.sv`
  - `rtl/common/descriptor_fifo.sv`
  - `rtl/tb/tb_stream_fifo.sv`
  - `rtl/tb/tb_descriptor_fifo.sv`
  - `hls/common/fixed_types.hpp`
  - `hls/common/stream_utils.hpp`
- Phase 0 hardening is also in place:
  - `block_id_e` now includes explicit tags for KV-cache write and requantize
  - HLS common quantization helpers avoid explicit floating-point arithmetic
- external RTL scale metadata is unsigned by contract
- internal HLS accumulators may widen beyond `Q16.16` where numerically useful
- any Q16.16-to-INT8 quantization path divides by the stored unsigned Q16.16
  scale payload directly before round-to-nearest-even and clamp
  - `tb_stream_fifo.sv` covers occupancy, full backpressure, and simultaneous
    push/pop cases
- Phase 1 control skeleton is now in place:
  - `rtl/control/axi_lite_ctrl_slave.sv`
  - `rtl/control/kernel_reg_file.sv`
  - `rtl/control/host_cmd_status_mgr.sv`
  - `rtl/control/prefill_decode_controller.sv`
  - `rtl/control/layer_controller.sv`
  - `rtl/control/stop_condition_unit.sv`
  - `rtl/tb/tb_axi_lite_ctrl_slave.sv`
  - `rtl/tb/tb_host_cmd_status_mgr.sv`
  - `rtl/tb/tb_prefill_decode_controller.sv`
  - AXI/register, host-cmd/status, and controller smoke tests pass
  - Phase 1 hardening coverage now explicitly includes:
    - sticky `DONE/ERROR/STOP` clear-on-START behavior in
      `tb_axi_lite_ctrl_slave.sv`
    - end-to-end `STOP_REASON_MAX_TOKENS` in
      `tb_prefill_decode_controller.sv`
    - relaunch command refetch plus error-only terminal status write in
      `tb_host_cmd_status_mgr.sv`
- Phase 2 memory / DMA / buffer layer is now in place:
    - `rtl/memory/hbm_port_router.sv`
    - `rtl/memory/prompt_token_reader.sv`
    - `rtl/memory/generated_token_writer.sv`
    - `rtl/memory/weight_dma_reader.sv`
   - `rtl/memory/embedding_lmhead_dma_reader.sv`
   - `rtl/memory/kv_cache_dma_reader.sv`
   - `rtl/memory/kv_cache_dma_writer.sv`
   - `rtl/memory/debug_dma_writer.sv`
   - `rtl/memory/scale_metadata_store.sv`
   - `rtl/memory/tile_buffer_bank.sv`
   - `rtl/memory/kv_cache_manager.sv`
   - `rtl/tb/tb_hbm_port_router.sv`
   - `rtl/tb/tb_tile_buffer_bank.sv`
    - `rtl/tb/tb_prompt_token_reader.sv`
    - `rtl/tb/tb_generated_token_writer.sv`
    - `rtl/tb/tb_scale_metadata_store.sv`
    - `rtl/tb/tb_kv_cache_manager.sv`
    - `rtl/tb/tb_weight_dma_reader.sv`
    - `rtl/tb/tb_kv_cache_dma_reader.sv`
    - `rtl/tb/tb_kv_cache_dma_writer.sv`
    - `rtl/tb/tb_embedding_lmhead_dma_reader.sv`
    - Phase 2 hardening coverage now explicitly includes:
      - read/write arbitration and routing in `tb_hbm_port_router.sv`
      - ping/pong bank isolation in `tb_tile_buffer_bank.sv`
      - prompt token burst fetch and token stream emission in
        `tb_prompt_token_reader.sv`
      - generated-token write beat addressing and ring wrap in
        `tb_generated_token_writer.sv`
      - scale metadata write/readback in `tb_scale_metadata_store.sv`
      - KV descriptor address and pseudo-channel mapping in
        `tb_kv_cache_manager.sv`
      - streamed multi-beat weight reads plus tensor-to-block tag mapping in
        `tb_weight_dma_reader.sv`
      - streamed multi-beat K/V cache reads in `tb_kv_cache_dma_reader.sv`
      - buffered K/V write payload capture plus valid-independent ready in
        `tb_kv_cache_dma_writer.sv`
      - multi-beat embedding/gamma streaming, LM-head weight beats, and
        aggregated scale metadata in `tb_embedding_lmhead_dma_reader.sv`
      - all Phase 2 testbenches are now wired to real `clk` / `rst_n`
 - Phase 3 shared GEMM compute layer is now in place:
    - `rtl/compute/mac_lane.sv`
    - `rtl/compute/accumulator_bank.sv`
    - `rtl/compute/requantize_unit.sv`
    - `rtl/compute/shared_gemm_engine.sv`
    - `rtl/compute/gemm_operand_router.sv`
    - `rtl/compute/gemm_result_router.sv`
    - `rtl/compute/gemm_op_scheduler.sv`
    - `rtl/tb/tb_mac_lane.sv`
    - `rtl/tb/tb_accumulator_bank.sv`
    - `rtl/tb/tb_requantize_unit.sv`
    - `rtl/tb/tb_shared_gemm_engine.sv`
    - `rtl/tb/tb_gemm_operand_router.sv`
    - `rtl/tb/tb_gemm_result_router.sv`
    - `rtl/tb/tb_gemm_op_scheduler.sv`
    - Phase 3 hardening coverage now explicitly includes:
      - signed INT8xINT8->INT32 leaf arithmetic in `tb_mac_lane.sv`
      - 512-lane accumulator clear/load behavior in `tb_accumulator_bank.sv`
      - bank-scaled Q16.16 requantization in `tb_requantize_unit.sv`
      - two-cycle accumulation plus snapshot backpressure in
        `tb_shared_gemm_engine.sv`
      - multi-mode operand routing in `tb_gemm_operand_router.sv`
      - quantized-vs-raw result routing in `tb_gemm_result_router.sv`
      - full decoder-layer GEMM schedule counts plus LM-head-only schedule in
        `tb_gemm_op_scheduler.sv`
 - Phase 3 golden-trace export scaffolding is now in place:
    - `model/export_fpga_vectors.py`
    - `sim/golden_traces/manifest.json`
    - `sim/golden_traces/phase3/*.npz`
    - `sim/golden_traces/phase3/rtl/*.memh`
    - current export scope includes:
      - prefill and decode cases for layer 0
      - GEMM cases for Q, K, V, O, gate, up, and down
      - paired requantization cases for the same operations
    - exported GEMM traces include both:
      - canonical tile matrices
      - row-major 512-lane step-packed arrays for direct RTL consumption
    - current RTL consumption is in:
      - `rtl/tb/tb_requantize_unit.sv`
      - `rtl/tb/tb_shared_gemm_engine.sv`
    - `tb_shared_gemm_engine.sv` uses a real exported `K_TILE=64` q-projection
      smoke slice because a full 2048-step q-projection replay is too slow for
      the normal Icarus smoke-test loop
- Phase 4 attention-path layer is now in place:
    - `rtl/compute/rope_cos_rom.memh`
    - `rtl/compute/rope_sin_rom.memh`
    - `rtl/compute/rope_lut_rom.sv`
    - `rtl/compute/rope_unit.sv`
    - `rtl/compute/gqa_router.sv`
    - `rtl/compute/causal_mask_unit.sv`
    - `rtl/tb/tb_rope_unit.sv`
    - `rtl/tb/tb_gqa_router.sv`
    - `rtl/tb/tb_causal_mask_unit.sv`
    - `model/export_fpga_vectors.py` now also exports Phase 4 cases:
      - prefill RoPE
      - decode RoPE
      - prefill causal mask
      - decode causal mask
    - Phase 4 hardening coverage now explicitly includes:
      - generated Q16.16 RoPE ROM contents under `rtl/compute/*.memh`
      - trace-backed prefill/decode RoPE verification in `tb_rope_unit.sv`
      - grouped-query routing and KV-head validation in `tb_gqa_router.sv`
      - trace-backed prefill/decode score masking in `tb_causal_mask_unit.sv`
- Phase 5 nonlinear HLS path is now in place:
    - `hls/common/nonlinear_math.hpp`
    - `hls/common/test_memh.hpp`
    - `hls/rmsnorm/rmsnorm_core_hls.hpp`
    - `hls/rmsnorm/rmsnorm_core_hls.cpp`
    - `hls/rmsnorm/tb_rmsnorm.cpp`
    - `hls/softmax/softmax_core_hls.hpp`
    - `hls/softmax/softmax_core_hls.cpp`
    - `hls/softmax/tb_softmax.cpp`
    - `hls/silu/silu_core_hls.hpp`
    - `hls/silu/silu_core_hls.cpp`
    - `hls/silu/tb_silu.cpp`
    - `rtl/nonlinear/rmsnorm_wrapper.sv`
    - `rtl/nonlinear/softmax_wrapper.sv`
    - `rtl/nonlinear/silu_wrapper.sv`
    - `rtl/tb/tb_rmsnorm_wrapper.sv`
    - `rtl/tb/tb_softmax_wrapper.sv`
    - `rtl/tb/tb_silu_wrapper.sv`
    - Phase 5 hardening coverage now explicitly includes:
      - host-side fixed-point RMSNorm verification in `tb_rmsnorm.cpp`
      - host-side fixed-point softmax verification in `tb_softmax.cpp`
      - host-side fixed-point SiLU verification in `tb_silu.cpp`
      - trace-backed RMSNorm wrapper verification in `tb_rmsnorm_wrapper.sv`
      - trace-backed softmax wrapper verification in `tb_softmax_wrapper.sv`
      - trace-backed SiLU wrapper verification in `tb_silu_wrapper.sv`
    - frozen Phase 5 wrapper/HLS contracts now include:
      - RMSNorm consumes input activation scale and output quantization scale
      - SiLU consumes input activation scale and output quantization scale
      - softmax consumes input score scale and emits fixed probability scale
        `1/127`
      - final and per-layer gamma vectors remain FP16 in HBM and are unpacked
        into Q16.16 on the normalization path
- `model/export_fpga_vectors.py` now also exports Phase 5 cases:
    - prefill and decode RMSNorm traces
    - prefill and decode softmax traces
    - prefill and decode SiLU traces
    - packed `.memh` fixtures for the Phase 5 RTL wrapper benches
- Phase 6 compute/dataflow layer is now in place:
    - `rtl/compute/embedding_lookup.sv`
    - `rtl/compute/embedding_quantizer.sv`
    - `rtl/compute/residual_add.sv`
    - `rtl/compute/elementwise_mul.sv`
    - `rtl/compute/lm_head_controller.sv`
    - `rtl/compute/argmax_reduction.sv`
    - `rtl/compute/debug_capture_mux.sv`
    - `rtl/tb/tb_embedding_lookup.sv`
    - `rtl/tb/tb_embedding_quantizer.sv`
    - `rtl/tb/tb_residual_add.sv`
    - `rtl/tb/tb_elementwise_mul.sv`
    - `rtl/tb/tb_lm_head_controller.sv`
    - `rtl/tb/tb_argmax_reduction.sv`
    - `rtl/tb/tb_debug_capture_mux.sv`
    - Phase 6 hardening coverage now explicitly includes:
      - trace-backed embedding-row request and row-assembly verification in
        `tb_embedding_lookup.sv`
      - trace-backed FP16-to-INT8 embedding batch quantization in
        `tb_embedding_quantizer.sv`
      - trace-backed residual1/residual2 INT32 accumulation in
        `tb_residual_add.sv`
      - trace-backed SwiGLU multiply verification in `tb_elementwise_mul.sv`
      - outer vocab-tile loop verification in `tb_lm_head_controller.sv`
      - trace-backed greedy argmax verification in `tb_argmax_reduction.sv`
      - non-backpressuring debug-mux drop behavior in `tb_debug_capture_mux.sv`
- `model/export_fpga_vectors.py` now also exports Phase 6 cases:
    - prefill and decode embedding-lookup row traces
    - prefill and decode embedding-quantizer batch traces
    - prefill and decode residual-add traces
    - prefill and decode layer-0 SwiGLU multiply traces
    - prefill and decode final-logit argmax traces
    - packed `.memh` fixtures for:
      - `tb_embedding_lookup.sv`
      - `tb_embedding_quantizer.sv`
      - `tb_residual_add.sv`
      - `tb_elementwise_mul.sv`
      - `tb_argmax_reduction.sv`
- Phase 7 decoder-layer integration is now in place:
    - `rtl/control/layer_controller.sv`
    - `rtl/compute/gemm_op_scheduler.sv`
    - `rtl/compute/gemm_operand_router.sv`
    - `rtl/compute/gemm_result_router.sv`
    - `rtl/tb/tb_decoder_layer_smoke.sv`
    - Phase 7 hardening coverage now explicitly includes:
      - exact fixed decoder-layer block order for one real TinyLlama layer pass
      - repeated `score -> causal_mask -> softmax -> weighted_sum` sequencing
        across all 32 query heads
      - production block-driven GEMM tile scheduling with real TinyLlama tile
        counts
      - integrated operand/result routing checks during the decoder-layer smoke
        pass
      - compatibility of the new block-level `layer_controller` with
        `tb_prefill_decode_controller.sv`
- `model/export_fpga_vectors.py` now also exports Phase 7 cases:
    - prefill decoder-layer schedule fixtures
    - decode decoder-layer schedule fixtures
    - packed `.memh` fixtures for:
      - `tb_decoder_layer_smoke.sv`
- Phase 8 runtime-core integration is now in place:
    - `rtl/top/tinyllama_u55c_kernel_top.sv`
    - `rtl/tb/tb_prefill_decode_smoke.sv`
    - `rtl/tb/tb_kernel_top_smoke.sv`
    - `rtl/control/prefill_decode_controller.sv` is now command-aware and:
      - waits for `command_info_valid`
      - launches prompt read only for prefill
      - waits for `prompt_read_done`
      - starts the layer engine for prefill or decode
      - pulses `lm_head_start` and `argmax_start` together
    - `rtl/control/host_cmd_status_mgr.sv` is now runtime-visible and:
      - fetches the PC30 command beat on every launch
      - writes one launch/busy status beat on `START`
      - writes one terminal status beat on done/error/stop
    - `model/export_fpga_vectors.py` now also exports Phase 8 cases:
      - one real-model prefill+decode runtime case
      - packed `.memh` fixtures for:
        - `tb_prefill_decode_smoke.sv`
        - `tb_kernel_top_smoke.sv`
    - Phase 8 hardening coverage now explicitly includes:
      - exported runtime-control verification in `tb_prefill_decode_smoke.sv`
      - AXI-Lite launch, PC30 command fetch, prompt beat count, generated-token
        writeback, final status payload, and interrupt verification in
        `tb_kernel_top_smoke.sv`
- Phase 9 runtime acceptance and shell-wrapper closure are now in place:
    - `rtl/top/tinyllama_u55c_shell_wrapper.sv`
    - `rtl/tb/tb_kernel_top_acceptance.sv`
    - `rtl/tb/tb_shell_wrapper_smoke.sv`
    - `model/export_fpga_vectors.py` now also exports Phase 9 cases:
      - one runtime-acceptance fixture set under `sim/golden_traces/phase9/rtl/`
      - expected clean terminal status word for the nominal runtime case
      - expected host-abort terminal status word for the in-flight abort case
    - Phase 9 hardening coverage now explicitly includes:
      - top-level abort during `RUN_LAYERS`, relaunch, sticky-status clear, and
        integrated command/status visibility in `tb_kernel_top_acceptance.sv`
      - shell-side read/write backpressure against the new buffered wrapper seam
        in `tb_shell_wrapper_smoke.sv`
      - preservation of the coupled write-request contract across the
        shell-facing wrapper around `tinyllama_u55c_kernel_top.sv`
- Post-Phase-9 real inference closure now includes the first concrete
  embedding-plus-decoder-plus-final-RMSNorm-plus-tail scaffold slice:
    - `rtl/top/runtime_embedding_frontend.sv`
    - `rtl/top/runtime_decoder_datapath.sv`
    - `rtl/top/runtime_final_rmsnorm_tail.sv`
    - `rtl/top/runtime_lm_head_tail.sv`
    - `rtl/nonlinear/rmsnorm_core_hls_ip.sv`
    - `rtl/nonlinear/silu_core_hls_ip.sv`
    - `rtl/tb/tb_runtime_embedding_frontend.sv`
    - `rtl/tb/tb_runtime_decoder_datapath.sv`
    - `rtl/tb/tb_runtime_final_rmsnorm_tail.sv`
    - `rtl/tb/tb_runtime_lm_head_tail.sv`
    - `rtl/top/tinyllama_u55c_kernel_top.sv` now instantiates the runtime
      embedding frontend, the runtime decoder datapath, the runtime final
      RMSNorm helper, and the runtime LM-head tail helper, and routes real
      `TENSOR_SCALE_META` / `TENSOR_EMBED` / `TENSOR_FINAL_RMS_GAMMA` /
      `TENSOR_LM_HEAD` reads through `hbm_port_router.sv`
    - the current top-level now waits for completed embedding ingress before
      starting the layer engine in prefill mode and before restarting decode
      layers after each non-stopping emitted token
    - emitted decode tokens are now fed back through a held-valid runtime token
      mux into `runtime_embedding_frontend.sv`
    - `runtime_decoder_datapath.sv` now replaces the old one-cycle per-block
      completion stub with TinyLlama-scale deterministic block completion plus
      a coherent FFN tile chain:
      - real `shared_gemm_engine.sv` + `requantize_unit.sv` on
        `BLOCK_GATE` and `BLOCK_UP`
      - real `silu_wrapper.sv` on `BLOCK_SILU`
      - real `elementwise_mul.sv` on `BLOCK_GLU_MUL`
      - real `shared_gemm_engine.sv` on `BLOCK_DOWN`, followed by the real
        `residual_add.sv` + `requantize_unit.sv` hidden-state update path
      - a coherent attention-output chain where `BLOCK_WEIGHTED_SUM` now stages
        an attention tile, `BLOCK_O` routes that staged tile through the real
        `shared_gemm_engine.sv`, and `BLOCK_RESIDUAL1` applies the staged O
        projection back onto the hidden-state tile before the cursor advances
    - `runtime_final_rmsnorm_tail.sv` now fetches real final RMSNorm gamma,
      runs the real `rmsnorm_wrapper.sv`, and turns that deterministic
      final-hidden scaffold stream into a post-RMSNorm runtime stream for the
      token-selection tail
    - `runtime_lm_head_tail.sv` now wraps the real `lm_head_controller.sv` and
      `argmax_reduction.sv`, consumes the post-RMSNorm runtime stream, fetches
      real `TENSOR_LM_HEAD` weights through
      `embedding_lmhead_dma_reader.sv`, drives the reused
      `shared_gemm_engine.sv`, and replaces the old LM/token stub in
      `tinyllama_u55c_kernel_top.sv`
    - because `runtime_decoder_datapath.sv` still evolves hidden state through
      a deterministic block-driven scaffold instead of the full decoder math
      chain, the present integrated runtime token path remains deterministic
      even though the LM-head DMA/shared-GEMM/argmax tail is now real
    - the runtime final RMSNorm helper currently uses a stable dedicated
      output-scale source in `tinyllama_u55c_kernel_top.sv`
    - do not drive that helper output-scale contract from a live decoder bus;
      the long-term replacement is a configured final-RMSNorm output-scale path
    - current canonical normalized-shell addresses for this slice are:
      - embedding rows: `0x0000_0000_1000_0000`
      - embedding-output scale metadata: `0x0000_0000_0400_0000`
      - final RMSNorm gamma: `0x0000_0000_0800_0000`
      - LM-head weights: `0x0000_0000_2000_0000`
    - hardening coverage now explicitly includes:
      - trace-backed scale-fetch plus one real embedding-row integration in
        `tb_runtime_embedding_frontend.sv`
      - TinyLlama-scale 22-layer block-completion plus final-hidden emission in
        `tb_runtime_decoder_datapath.sv`
      - TinyLlama-scale final-gamma DMA plus real final-RMSNorm streaming in
        `tb_runtime_final_rmsnorm_tail.sv`
      - direct controller/reducer coverage for the real runtime LM-head tail in:
        - `tb_lm_head_controller.sv`
        - `tb_argmax_reduction.sv`
      - continued Phase 8/9 top-level regression under real embedding/meta read
        bursts, real final-RMSNorm gamma reads, real LM-head DMA/shared-GEMM
        tile sweeps, real controller/argmax handshakes, and decode-token
        embedding relaunches in:
        - `tb_kernel_top_smoke.sv`
        - `tb_kernel_top_acceptance.sv`
        - `tb_shell_wrapper_smoke.sv`
      - fresh local reruns of the current slice are now recorded for:
        - `tb_runtime_decoder_datapath.sv`
        - `tb_runtime_final_rmsnorm_tail.sv`
        - `tb_runtime_lm_head_tail.sv`
        - `tb_kernel_top_smoke.sv`
        - `tb_kernel_top_acceptance.sv`
        - `tb_shell_wrapper_smoke.sv`
      - the current decoder-step closure specifically proves the real FFN leaf
        chain now fires once per layer through:
        - `gate_gemm=22`
        - `up_gemm=22`
        - `down_gemm=22`
        - `o_gemm=22`
        - `silu_done=22`
        - `mul_done=22`
        in `tb_runtime_decoder_datapath.sv`
      - the fresh top-level reruns for this heavier slice are now recorded
        under:
        - `sim/logs/xsim_tb_kernel_top_smoke_*`
        - `sim/logs/xsim_tb_kernel_top_acceptance_*`
        - `sim/logs/xsim_tb_shell_wrapper_smoke_*`
        because the added projection-GEMM work makes the full runtime benches
        impractically slow under Icarus
    - current vendor-synthesis hardening note for this slice:
      - `embedding_quantizer.sv` now quantizes one `N_TILE = 32`-element
        feature slice per cycle while ingesting each row and buffers per-row
        INT8 feature tiles before 512-lane batch emission
      - do not reintroduce either a batch-global flattened FP16 store or a
        `512`-way divide/modulo fanout on the output path; both shapes drove
        Vivado into an unstable post-synthesis tail
      - `rtl/nonlinear/rmsnorm_core_hls_ip.sv` is currently a repo-owned Icarus
        simulation model for the HLS IP boundary and is not, by itself, a new
        Vivado synthesis checkpoint
      - `rtl/nonlinear/silu_core_hls_ip.sv` is now the repo-owned runtime
        simulation model for the SiLU HLS-IP boundary and likewise does not,
        by itself, imply a new Vivado synthesis checkpoint

Immediate next implementation step:

1. continue replacing the current deterministic block-driven decoder-state
   scaffold inside `runtime_decoder_datapath.sv` with the actual
   decoder/final-hidden compute chain, with the next focus on the remaining
   score / causal-mask / softmax / weighted-sum synthetic path and on moving
   `block_done_o` closer to real datapath completion
2. keep the raw `m_axi_pc00..31` wrapper and Vitis/platform packaging as the
   outer follow-on step after the top-level inference path is materially closer
   to final

Verification policy:

- directed smoke tests remain mandatory for control and plumbing modules
- golden traces are not required for Phase 0, Phase 1, or Phase 2
- golden traces are recommended for Phase 3 arithmetic blocks
- golden traces are required from Phase 4 onward for math/dataflow blocks
- golden traces are required for Phase 7-9 integration gates
- canonical export root is `sim/golden_traces/`
- detailed trace policy lives in `docs/golden_trace_plan.md`

---

## 10. Working Rules For Future Codex Sessions

When modifying this repository:

- follow `docs/design_decisions.txt` first
- follow `docs/modules.md` for interfaces and microarchitecture
- follow `docs/implementation_checklist.md` for coding order
- follow `docs/golden_trace_plan.md` for deciding when trace-backed verification
  is required
- do not silently change tile sizes, widths, HBM mapping, or precision policy
- do not introduce production TinyLlama RTL into the legacy flat `rtl/` files
- do not reintroduce Pythia-specific code or docs
- keep README files consistent with `.gitignore`
- if implementation forces a new concrete external contract such as a register
  map, launch semantic, or bus-level handshake rule, update the authoritative
  design docs and `AGENTS.md` in the same turn before building on it
- if architecture or implementation policy changes materially, update:
  - `docs/design_decisions.txt`
  - `docs/modules.md`
  - `docs/implementation_checklist.md`
  - `AGENTS.md`

If you create a new production module file, make sure:

- it matches the planned source tree
- its purpose matches `docs/modules.md`
- its creation order makes sense relative to `docs/implementation_checklist.md`
- it has a defined first verification target

---

## 11. When To Update This File

Update `AGENTS.md` whenever any of the following changes:

- target model
- control ownership
- quantization policy
- HBM allocation
- tile sizes or DMA width
- RTL/HLS partition
- planned source tree
- immediate next implementation milestone

This file should remain a high-signal session bootstrap file, not a full design
spec duplicate.
