# Module Inventory - TinyLlama U55C FPGA Accelerator

This file defines the implementation inventory for the TinyLlama 1.1B FPGA
accelerator on the Xilinx Alveo U55C.

The goal of this file is not to describe software behavior. The goal is to list
every physical block and logical stage that must exist in the hardware design so
that coding can begin in a structured way.

The source of truth for model dimensions and finalized architecture choices is
`design_decisions.txt`. This file translates those decisions into a hardware
module plan.

---

## 1. Fixed Architectural Constants

These values are treated as fixed for the first implementation:

- `MODEL_ID = TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `N_LAYERS = 22`
- `D_MODEL = 2048`
- `D_FF = 5632`
- `N_Q_HEADS = 32`
- `N_KV_HEADS = 4`
- `KV_GROUPS = 8`
- `HEAD_DIM = 64`
- `VOCAB_SIZE = 32000`
- `MAX_POS = 2048`
- `SEQ_TILE = 64`
- `ACT_W = 8`
- `WEIGHT_W = 8`
- `ACC_W = 32`
- `TOKEN_W = 32`
- `SCALE_W = 32`
- `GEMM_LANES = 512`
- `HBM_PC_COUNT = 32`
- `HBM_ADDR_W = 64`
- `HBM_SHELL_DATA_W = platform-defined AXI width`
- `DMA_BEAT_W = 256`
- `STREAM_FIFO_DEPTH = 4`
- `SKID_BUFFER_DEPTH = 2`
- `DESC_FIFO_DEPTH = 8`
- `DEBUG_FIFO_DEPTH = 32`
- `TILE_BUFFER_BANKS = 16`
- `BANK_SLICE_INT8 = 32`
- `BANK_SLICE_INT32 = 8`
- `M_TILE = 16`
- `N_TILE = 32`
- `K_TILE = 64`
- `SCORE_Q_TILE = 16`
- `SCORE_K_TILE = 64`
- `VOCAB_TILE = 128`
- `HEAD_GROUP_PAR = 1`
- `FIXED_POINT_FMT = Q16.16`

Notes:

- The accelerator runs in two modes: `prefill` and `decode`.
- All 22 decoder layers use the same physical decoder-layer engine.
- All GEMM-heavy stages reuse the same physical shared GEMM engine.
- Token selection is greedy argmax only in the initial implementation.

---

## 2. Interface And Bus Conventions

The module list below uses the following interface groups.

### 2.1 Global Signals

- `ap_clk`: main kernel clock
- `ap_rst_n`: active-low synchronous reset
- `interrupt`: optional completion interrupt to host

### 2.2 Host Control Bus

- `s_axi_control`: AXI4-Lite slave for kernel launch, base addresses, mode, and status

Expected control-register fields:

- `start`
- `mode_prefill_decode`
- `cmd_base_addr`
- `status_base_addr`
- `debug_base_addr`
- `prompt_token_count`
- `max_new_tokens`
- `eos_token_id`
- `debug_enable`
- `debug_layer_sel`
- `debug_step_sel`
- `done`
- `busy`
- `error_code`

### 2.3 HBM Memory Buses

- `m_axi_pc00` .. `m_axi_pc31`: AXI4 master ports to U55C HBM pseudo-channels

Fixed HBM allocation:

- `PC00-PC15`: decoder-layer weights
- `PC16-PC17`: token embeddings, final RMSNorm gamma, quantization metadata
- `PC18-PC21`: LM-head weights
- `PC22-PC25`: K cache
- `PC26-PC29`: V cache
- `PC30`: host command block, prompt token list, generated-token ring buffer, status block
- `PC31`: debug capture buffer

### 2.4 Internal Streaming Conventions

All internal data streams use a ready/valid handshake with a payload bus and
sideband tags.

Common fields:

- `*_data`
- `*_valid`
- `*_ready`
- `*_last`
- `*_tag`

Common payload types:

- `token_bus`: token IDs and token counts
- `act_bus`: activation tiles
- `wt_bus`: weight tiles
- `acc_bus`: INT32 accumulator tiles
- `scale_bus`: quantization scales and metadata
- `dbg_bus`: debug capture payloads

`scale_bus` carries 16 unsigned Q16.16 scale multipliers per tile. Each scale
entry applies to one 32-lane bank slice of the 512-lane datapath.

Common tag fields:

- `layer_id`
- `block_id`
- `tile_id`
- `token_base`
- `seq_count`
- `head_id`

### 2.5 Internal DMA / HBM Handshake Convention

All Phase 2 memory modules use a split descriptor/data handshake.

Read side:

- `rd_desc_valid`
- `rd_desc_ready`
- `rd_desc : dma_desc_t`
- `rd_data_valid`
- `rd_data_ready`
- `rd_data[DMA_BEAT_W-1:0]`

Write side:

- `wr_desc_valid`
- `wr_desc_ready`
- `wr_desc : dma_desc_t`
- `wr_data_valid`
- `wr_data_ready`
- `wr_data[DMA_BEAT_W-1:0]`

`dma_desc_t.byte_count` is 32 bits. This is required because full KV-cache
transfers can exceed 65535 bytes.

---

## 3. Crossbar / Switching Fabric Decision

Yes, switching fabric is required, but not as a fully generic all-to-all
packet crossbar.

The design uses two fixed-function routing layers:

1. `hbm_port_router.sv`
   This is a scheduled AXI arbitration and steering layer between internal DMA
   clients and the fixed HBM pseudo-channel groups. Because channels are
   statically assigned by function, this is a narrow router/arbiter, not a
   large dynamic memory crossbar.

2. `gemm_operand_router.sv` and `gemm_result_router.sv`
   These route activations, weights, scales, and results between shared tile
   buffers and the shared GEMM engine. This is the on-chip equivalent of a
   compute crossbar, but again it is mode-driven and scheduled, not a generic
   many-master switch.

There is no requirement for a generic NoC-style crossbar in the first
implementation.

---

## 4. Logical TinyLlama Stages To Physical Hardware Mapping

This table lists every logical stage in the TinyLlama datapath and the physical
hardware blocks that implement it.

| Logical stage | Physical module(s) used |
|---|---|
| Host launch and status | `axi_lite_ctrl_slave.sv`, `kernel_reg_file.sv`, `host_cmd_status_mgr.sv` |
| Prompt token fetch | `prompt_token_reader.sv`, `hbm_port_router.sv`, `tile_buffer_bank.sv` |
| Embedding lookup | `embedding_lookup.sv`, `embedding_lmhead_dma_reader.sv`, `tile_buffer_bank.sv` |
| Embedding output quantization | `embedding_quantizer.sv`, `requantize_unit.sv`, `scale_metadata_store.sv` |
| RMSNorm 1 / RMSNorm 2 / Final RMSNorm | `rmsnorm_wrapper.sv`, `rmsnorm_core_hls.cpp` |
| Q / K / V / O projections | `gemm_op_scheduler.sv`, `gemm_operand_router.sv`, `shared_gemm_engine.sv`, `gemm_result_router.sv` |
| RoPE | `rope_lut_rom.sv`, `rope_unit.sv` |
| GQA K/V routing | `gqa_router.sv` |
| Attention score GEMM | `gemm_op_scheduler.sv`, `shared_gemm_engine.sv`, `causal_mask_unit.sv`, `softmax_wrapper.sv`, `softmax_core_hls.cpp` |
| Weighted sum with V | `gemm_op_scheduler.sv`, `shared_gemm_engine.sv` |
| Residual add | `residual_add.sv`, `requantize_unit.sv` |
| gate / up / down projections | `gemm_op_scheduler.sv`, `shared_gemm_engine.sv` |
| SiLU and GLU multiply | `silu_wrapper.sv`, `silu_core_hls.cpp`, `elementwise_mul.sv` |
| KV-cache read / write | `kv_cache_manager.sv`, `kv_cache_dma_reader.sv`, `kv_cache_dma_writer.sv`, `tile_buffer_bank.sv` |
| LM head | `lm_head_controller.sv`, `shared_gemm_engine.sv`, `gemm_operand_router.sv`, `gemm_result_router.sv` |
| Greedy token selection | `argmax_reduction.sv` |
| Stop control | `stop_condition_unit.sv`, `prefill_decode_controller.sv` |
| Generated token writeback | `generated_token_writer.sv`, `host_cmd_status_mgr.sv` |
| Debug capture | `debug_capture_mux.sv`, `debug_dma_writer.sv` |

---

## 5. Physical Module Inventory

This section lists every physical module that needs to exist in the first
hardware implementation.

### 5.1 Top-Level And Host-Facing Control

#### M00. `tinyllama_u55c_kernel_top.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Top-level RTL kernel. Instantiates control, memory, buffering,
  shared compute, HLS wrappers, and debug.
- Inputs / buses:
  - `ap_clk`, `ap_rst_n`
  - `s_axi_control`
  - `m_axi_pc00` .. `m_axi_pc31`
- Outputs / buses:
  - `interrupt`
  - host-visible status through `s_axi_control`
- Parallelism:
  - structural only; all system-level parallelism is created by child modules
    and overlapped DMA/compute pipelines.

#### M01. `axi_lite_ctrl_slave.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: AXI4-Lite slave for scalar control and status.
- Inputs / buses:
  - `ap_clk`, `ap_rst_n`
  - `s_axi_control`
- Outputs / buses:
  - decoded register writes to `kernel_reg_file.sv`
  - register reads back to host
- Parallelism:
  - no arithmetic parallelism; single control-path module.

#### M02. `kernel_reg_file.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Holds launch parameters, base addresses, debug configuration, and
  completion/error status.
- Inputs / buses:
  - register write strobes and write data from `axi_lite_ctrl_slave.sv`
  - status updates from controller and DMA blocks
- Outputs / buses:
  - command fields to `host_cmd_status_mgr.sv` and `prefill_decode_controller.sv`
  - status fields back to `axi_lite_ctrl_slave.sv`
- Parallelism:
  - no arithmetic parallelism.

#### M03. `host_cmd_status_mgr.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Reads the command block from `PC30`, exposes the prompt token list
  base pointer and runtime parameters, and writes status/results metadata back
  to `PC30`.
- Inputs / buses:
  - `cmd_base_addr`, `status_base_addr`, `start`
  - one-beat command read descriptor and one-beat command data return from
    `hbm_port_router.sv`
- Outputs / buses:
  - parsed prompt-token-base, generated-token-ring-base, and generated-token
    capacity fields to downstream runtime blocks
  - one-beat status-write descriptor and one-beat status payload to
    `hbm_port_router.sv`
- Parallelism:
  - no arithmetic parallelism; pipelined command/status DMA.

#### M04. `prefill_decode_controller.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Owns the high-level runtime flow:
  `embedding -> prefill -> decode loop -> LM head -> argmax -> stop`.
- Inputs / buses:
  - command/config from `kernel_reg_file.sv` and `host_cmd_status_mgr.sv`
  - token-emission status from `argmax_reduction.sv`
  - completion/error flags from all datapath blocks
- Outputs / buses:
  - mode, layer-start, block-start, and loop-control signals
  - generated token count and stop control to `stop_condition_unit.sv`
- Parallelism:
  - control parallelism only; overlaps DMA launch with downstream compute.

#### M05. `layer_controller.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Iterates `layer_idx = 0..21`, selects layer-local weights and
  KV-cache regions, and sequences the reused decoder-layer engine.
- Inputs / buses:
  - mode and layer-start from `prefill_decode_controller.sv`
  - layer-done from `gemm_op_scheduler.sv` and HLS wrappers
- Outputs / buses:
  - `layer_id`
  - weight-region and KV-region selectors
  - per-layer control tags
- Parallelism:
  - sequential across layers; no inter-layer compute parallelism in the first
    implementation.

#### M06. `kv_cache_manager.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Tracks K/V cache base pointers, token positions, head offsets, and
  append locations for both prefill and decode.
- Inputs / buses:
  - `layer_id`, `mode`, `token_base`, `seq_count`
  - cache-read and cache-write requests from scheduler
- Outputs / buses:
  - read descriptors to `kv_cache_dma_reader.sv`
  - write descriptors to `kv_cache_dma_writer.sv`
- Parallelism:
  - address-generation parallelism for K and V sides.

#### M07. `gemm_op_scheduler.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Sequences all GEMM-backed logical operations:
  Q, K, V, score, weighted sum, O, gate, up, down, LM head.
- Inputs / buses:
  - block-start from controller
  - buffer-ready and DMA-ready signals
- Outputs / buses:
  - GEMM mode select
  - operand-route select
  - result-route select
  - tile loop counters
- Parallelism:
  - tile-level pipelining and overlap with DMA fetch.

#### M08. `stop_condition_unit.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Stops decode on `EOS` or `max_new_tokens`.
- Inputs / buses:
  - emitted token ID
  - `generated_token_count`
  - `eos_token_id`
  - `max_new_tokens`
- Outputs / buses:
  - `stop_now`
  - `stop_reason`
- Parallelism:
  - no arithmetic parallelism.

### 5.2 Memory Routing, DMA, And Buffers

#### M09. `hbm_port_router.sv`

- Language: SystemVerilog RTL
- Physical instances: 1 logical router, multiple port slices
- Purpose: Routes internal DMA clients to fixed HBM pseudo-channel groups and
  arbitrates conflicting requests inside each group.
- Inputs / buses:
  - DMA read/write requests from all DMA modules
  - `m_axi_pc00` .. `m_axi_pc31` ready/response channels
- Outputs / buses:
  - AXI requests to HBM ports
  - returned data / write responses to DMA clients
- Parallelism:
  - parallel across independently assigned pseudo-channel groups
  - arbitration only within a group.

#### M10. `prompt_token_reader.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Reads prompt token IDs from the command region in `PC30`.
- Inputs / buses:
  - prompt token list base address
  - token count
  - HBM read channel from `hbm_port_router.sv`
- Outputs / buses:
  - `token_bus` stream to `embedding_lookup.sv`
- Parallelism:
  - burst fetch of multiple token IDs; sequential consumption by prefill controller.

#### M11. `generated_token_writer.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Writes emitted token IDs into the generated-token ring buffer in `PC30`.
- Inputs / buses:
  - token IDs from `argmax_reduction.sv`
  - ring write pointer from `host_cmd_status_mgr.sv`
- Outputs / buses:
  - HBM write requests to `hbm_port_router.sv`
- Parallelism:
  - one emitted token per decode step; pipelined write path.

#### M12. `weight_dma_reader.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Reads decoder-layer weight tiles from `PC00-PC15`.
- Inputs / buses:
  - layer ID
  - tensor ID
  - output-channel tile index
  - input-channel tile index
  - HBM read channel from `hbm_port_router.sv`
- Outputs / buses:
  - `wt_bus` stream into `tile_buffer_bank.sv`
  - one streamed `wt_bus` beat per returned DMA beat, with `tag.is_last`
    marking the final beat of the descriptor
- Parallelism:
  - striped parallel reads across multiple weight pseudo-channels.

#### M13. `embedding_lmhead_dma_reader.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Reads embedding rows, final RMSNorm gamma, quantization metadata,
  and LM-head tiles from `PC16-PC21`.
- Inputs / buses:
  - row address / tile descriptors
  - HBM read channel from `hbm_port_router.sv`
- Outputs / buses:
  - embedding-row beat stream to `embedding_lookup.sv`, with an explicit
    beat-level `last` indication
  - RMSNorm-gamma beat stream to the normalization path, with an explicit
    beat-level `last` indication
  - LM-head `wt_bus` beat stream to `tile_buffer_bank.sv`
  - aggregated scale metadata entry to `scale_metadata_store.sv`
- Parallelism:
  - independent read streams for embedding path and LM-head path.

#### M14. `kv_cache_dma_reader.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Reads K and V cache rows from `PC22-PC29`.
- Inputs / buses:
  - read descriptors from `kv_cache_manager.sv`
  - HBM read channel from `hbm_port_router.sv`
- Outputs / buses:
  - K/V `act_bus` stream to `tile_buffer_bank.sv`
  - one streamed `act_bus` beat per returned DMA beat, with `tag.is_last`
    marking the final beat of the descriptor
- Parallelism:
  - K and V streams in parallel, each striped across assigned pseudo-channels.

#### M15. `kv_cache_dma_writer.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Writes updated K and V rows back to `PC22-PC29`.
- Inputs / buses:
  - quantized K/V rows from `gemm_result_router.sv`
  - write descriptors from `kv_cache_manager.sv`
- Outputs / buses:
  - HBM write requests to `hbm_port_router.sv`
- Parallelism:
  - K and V writeback in parallel when HBM arbitration allows.

#### M16. `debug_dma_writer.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Writes optional debug captures into `PC31`.
- Inputs / buses:
  - `dbg_bus` from `debug_capture_mux.sv`
  - debug enable and base address
- Outputs / buses:
  - HBM write requests to `hbm_port_router.sv`
- Parallelism:
  - burst write of captured windows; inactive in performance builds.

#### M17. `scale_metadata_store.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Stores quantization scales for activations, K/V cache, and LM-head
  tiles after fetch.
- Inputs / buses:
  - `scale_bus` from DMA readers and scheduler
- Outputs / buses:
  - scale values to `requantize_unit.sv`, `embedding_quantizer.sv`,
    `kv_cache_dma_writer.sv`, and `lm_head_controller.sv`
- Parallelism:
  - multi-port register / SRAM lookup.

#### M18. `tile_buffer_bank.sv`

- Language: SystemVerilog RTL
- Physical instances:
  - activation ping
  - activation pong
  - weight ping
  - weight pong
  - KV ping
  - KV pong
  - score ping
  - score pong
  - LM-head ping
  - LM-head pong
- Purpose: Generic banked on-chip buffer used for activations, weights, K/V,
  attention scores, and LM-head tiles.
- Inputs / buses:
  - write-side DMA or compute stream
  - buffer-select and bank-select control
- Outputs / buses:
  - read-side streams to GEMM, HLS wrappers, and result routers
- Parallelism:
  - banked parallel reads/writes
  - ping/pong overlap between DMA and compute.

### 5.3 Shared Datapath And Dedicated RTL Blocks

#### M19. `embedding_lookup.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Converts token IDs into embedding-row fetches and outputs the first
  hidden-state vectors for prefill and decode.
- Inputs / buses:
  - `token_bus` from `prompt_token_reader.sv` or controller
  - embedding rows from `embedding_lmhead_dma_reader.sv`
- Outputs / buses:
  - raw embedding activation rows to `embedding_quantizer.sv`
- Parallelism:
  - row burst fetch for a `SEQ_TILE=64` prefill tile
  - single-row fetch in decode mode.

#### M20. `embedding_quantizer.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Converts embedding rows into INT8 activation tiles for the shared
  decoder datapath.
- Inputs / buses:
  - raw embedding rows
  - embedding scale metadata
- Outputs / buses:
  - `act_bus` to activation buffers
- Parallelism:
  - elementwise vector quantization across one activation tile.

#### M21. `shared_gemm_engine.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Main INT8 compute engine reused for all GEMM-heavy operations.
- Inputs / buses:
  - `act_bus`
  - `wt_bus`
  - mode/control from `gemm_op_scheduler.sv`
  - `mac_valid`, `clear_acc`, tile-loop control
- Outputs / buses:
  - `acc_bus` INT32 partial or final results
- Parallelism:
  - `512` parallel INT8 lanes
  - output-stationary accumulation
  - tile-level pipelining.

#### M22. `mac_lane.sv`

- Language: SystemVerilog RTL
- Physical instances: `512`
- Purpose: Single INT8 x INT8 -> INT32 MAC lane.
- Inputs / buses:
  - activation element
  - weight element
  - accumulator input
  - `mac_valid`
- Outputs / buses:
  - accumulator output
- Parallelism:
  - full lane-level parallelism across the GEMM engine.

#### M23. `accumulator_bank.sv`

- Language: SystemVerilog RTL
- Physical instances: 1 banked structure
- Purpose: Holds INT32 accumulators for the shared GEMM engine.
- Inputs / buses:
  - partial sums from `mac_lane.sv`
  - `clear_acc`
  - writeback control
- Outputs / buses:
  - `acc_bus`
- Parallelism:
  - one accumulator per GEMM lane.

#### M24. `gemm_operand_router.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Routes the correct activation and weight tiles into the shared GEMM
  engine according to the active logical operation.
- Inputs / buses:
  - activation buffers
  - weight buffers
  - score buffers
  - KV buffers
  - mode select from `gemm_op_scheduler.sv`
- Outputs / buses:
  - `act_bus` and `wt_bus` to `shared_gemm_engine.sv`
- Parallelism:
  - one route active per GEMM invocation, but route selection is pipelined and
    supports double-buffer overlap.

#### M25. `gemm_result_router.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Routes GEMM results to the correct destination:
  Q/K/V buffers, O path, score buffer, weighted-sum path, FFN path, KV writeback,
  or LM-head argmax path.
- Inputs / buses:
  - `acc_bus` from `shared_gemm_engine.sv`
  - mode select from `gemm_op_scheduler.sv`
  - scale metadata from `scale_metadata_store.sv`
- Outputs / buses:
  - routed result streams to buffers, requantizer, and final reduction blocks
- Parallelism:
  - one destination active per GEMM invocation; result writeback is pipelined.

#### M26. `rope_lut_rom.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Stores precomputed sine/cosine tables for RoPE positions.
- Inputs / buses:
  - `token_base`
  - `token_count`
- Outputs / buses:
  - one 512-lane packed `cos` vector for an `8 x 64` RoPE slice
  - one 512-lane packed `sin` vector for an `8 x 64` RoPE slice
- Concrete contract:
  - lane packing is token-major:
    `lane = token_local * HEAD_DIM + dim_local`
  - ROM access is structured as explicit `8 positions x 32 unique angle values`
    per slice, then broadcast across the paired 64 dimensions
  - token positions outside `token_count` emit identity rotation values:
    `cos = 1.0`, `sin = 0.0`
  - ROM contents are stored as signed `Q16.16` values loaded from generated
    memh files
- Parallelism:
  - parallel ROM reads across the selected vector width.

#### M27. `rope_unit.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Applies RoPE to Q and K after projection and before score computation.
- Inputs / buses:
  - Q head slice as one `8 x 64` packed activation tile
  - K head slice as one `8 x 64` packed activation tile
  - sine/cosine values from `rope_lut_rom.sv`
- Outputs / buses:
  - rotated Q head slice
  - rotated K head slice
- Concrete contract:
  - lane packing is token-major:
    `lane = token_local * HEAD_DIM + dim_local`
  - the rotary pairing follows TinyLlama's rotate-half convention:
    dims `0..31` pair with dims `32..63`
  - Q and K slices supplied to one RoPE invocation must carry the same
    `token_base`; RTL checks this in simulation
  - the RoPE output preserves the incoming activation scale; no extra scale
    bus is created for the rotary stage
- Parallelism:
  - vector-lane parallel multiply/add across the active RoPE slice.

#### M28. `gqa_router.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Implements grouped-query attention routing from 4 KV heads to 32
  query-head groups without physically copying K/V tensors in HBM.
- Inputs / buses:
  - K slice stream
  - V slice stream
  - selected `q_head_id`
- Outputs / buses:
  - one routed K-or-V stream with rewritten tags for the selected query head
  - routing error flag if the supplied KV head does not match
    `q_head_id / KV_GROUPS`
- Concrete contract:
  - the selected K path emits tags for `BLOCK_SCORE` / `GEMM_SCORE`
  - the selected V path emits tags for `BLOCK_WEIGHTED_SUM` /
    `GEMM_WEIGHTED_SUM`
  - the router does not replicate payload data; it validates and rewrites tags
    only
- Parallelism:
  - parallel fan-out by group using address reuse / replicated read pointers.

#### M29. `causal_mask_unit.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Applies the pre-softmax causal mask for both prefill and decode.
- Inputs / buses:
  - one score chunk as `8 query rows x 64 key columns`
  - `query_pos_base`
  - `key_pos_base`
  - `query_row_count`
  - `key_col_count`
  - `mode`
- Outputs / buses:
  - masked score tile to `softmax_wrapper.sv`
- Concrete contract:
  - score-chunk lane packing is row-major:
    `lane = query_row_local * SCORE_K_TILE + key_col_local`
  - masked fill value is fixed to `MASK_NEG_INF = -1000000000`
  - output `elem_count` is `query_row_count * SCORE_K_TILE`
  - key columns beyond `key_col_count` are explicitly forced to
    `MASK_NEG_INF` and remain inside the packed 64-column row shape
- Parallelism:
  - elementwise compare-and-mask across one score tile.

#### M30. `residual_add.sv`

- Language: SystemVerilog RTL
- Physical instances: 1 reused block
- Purpose: Adds the residual stream after attention and after FFN.
- Inputs / buses:
  - residual INT32 or INT8 stream
  - update INT32 or INT8 stream
  - mode tag
- Outputs / buses:
  - summed stream to `requantize_unit.sv` or next block
- Parallelism:
  - vector-lane elementwise add across the active tile.

#### M31. `requantize_unit.sv`

- Language: SystemVerilog RTL
- Physical instances: 1 reused block
- Purpose: Requantizes INT32 outputs back to INT8 where the next block expects
  quantized activations or cache rows.
- Inputs / buses:
  - INT32 input stream
  - output scale
  - clamp/rounding mode
- Outputs / buses:
  - INT8 stream
- Parallelism:
  - vector-lane elementwise requantization.
- Numeric contract:
  - scale payload format is unsigned Q16.16
  - one scale entry applies to one 32-lane bank slice
  - rounding mode is round-to-nearest-even
  - signed clamp range is `[-127, 127]`

#### M32. `elementwise_mul.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Performs the `SiLU(gate) * up` multiply inside SwiGLU.
- Inputs / buses:
  - SiLU output vector
  - `up_proj` vector
- Outputs / buses:
  - INT32 hidden vector to `requantize_unit.sv`
- Parallelism:
  - vector-lane elementwise multiply across the active FFN tile.

#### M33. `lm_head_controller.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Sequences LM-head tiled GEMM across vocabulary tiles and drives the
  final reduction path.
- Inputs / buses:
  - final hidden state
  - LM-head weight tile descriptors
  - scale metadata
- Outputs / buses:
  - GEMM schedule to `gemm_op_scheduler.sv`
  - partial-logit streams to `argmax_reduction.sv`
- Parallelism:
  - tiled processing across vocab tiles
  - overlaps weight fetch with logit reduction when possible.

#### M34. `argmax_reduction.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Computes greedy argmax across the full vocabulary after the LM-head path.
- Inputs / buses:
  - partial logits by vocab tile
  - vocab tile IDs
- Outputs / buses:
  - winning token ID
  - winning logit value
  - end-of-vocab reduction done
- Parallelism:
  - parallel compare-reduce within each vocab tile
  - hierarchical reduction across tiles.

#### M35. `debug_capture_mux.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Selects the requested internal tensors for debug capture without
  permanently wiring every internal bus to memory.
- Inputs / buses:
  - candidate debug sources from datapath and HLS wrappers
  - debug configuration from `kernel_reg_file.sv`
- Outputs / buses:
  - `dbg_bus` to `debug_dma_writer.sv`
- Parallelism:
  - no arithmetic parallelism; selection/multiplexing only.

### 5.4 HLS Blocks And RTL Wrappers

#### M36. `rmsnorm_wrapper.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Adapts the shared RTL datapath handshake to the HLS RMSNorm core.
- Inputs / buses:
  - INT8 or INT32 activation tile
  - RMSNorm gamma vector
  - mode tag
- Outputs / buses:
  - normalized INT8 activation tile
  - output scale metadata
  - done/ready status
- Parallelism:
  - one invocation at a time; reused for pre-attention norm, pre-FFN norm, and final norm.

#### M37. `rmsnorm_core_hls.cpp`

- Language: HLS C++
- Physical instances: 1 compiled IP core
- Purpose: Computes RMSNorm with higher internal precision.
- Inputs / buses:
  - activation tile stream
  - gamma vector stream
  - epsilon constant
- Outputs / buses:
  - normalized tile stream
- Parallelism:
  - reduction tree over hidden dimension
  - vector-lane multiply pipeline.

#### M38. `softmax_wrapper.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Adapts masked score tiles to the HLS softmax core.
- Inputs / buses:
  - masked score tile
  - tile shape
  - mode
- Outputs / buses:
  - INT8 probability tile to weighted-sum path
- Parallelism:
  - one score-row tile at a time; pipelined handshake.

#### M39. `softmax_core_hls.cpp`

- Language: HLS C++
- Physical instances: 1 compiled IP core
- Purpose: Computes stable softmax over attention-score rows.
- Inputs / buses:
  - masked score tile
  - row length
- Outputs / buses:
  - normalized probability tile
- Parallelism:
  - reduction for max
  - reduction for exp sum
  - pipelined elementwise normalization.

#### M40. `silu_wrapper.sv`

- Language: SystemVerilog RTL
- Physical instances: 1
- Purpose: Adapts the gate-projection output stream to the HLS SiLU core.
- Inputs / buses:
  - gate vector tile
- Outputs / buses:
  - INT8 SiLU-transformed gate vector
- Parallelism:
  - one vector tile at a time; pipelined handshake.

#### M41. `silu_core_hls.cpp`

- Language: HLS C++
- Physical instances: 1 compiled IP core
- Purpose: Computes SiLU for the gate branch of SwiGLU.
- Inputs / buses:
  - gate vector tile
- Outputs / buses:
  - SiLU(gate) vector tile
- Parallelism:
  - elementwise pipelined activation evaluation across the active tile.

### 5.5 Utility Primitives

#### M42. `stream_fifo.sv`

- Language: SystemVerilog RTL
- Physical instances: many parameterized instances
- Purpose: Elastic ready/valid FIFO used on wide data streams between DMA,
  buffers, GEMM, and HLS wrappers.
- Inputs / buses:
  - `in_data`, `in_valid`, `out_ready`
  - clock/reset
  - depth parameter
- Outputs / buses:
  - `in_ready`, `out_data`, `out_valid`
- Parallelism:
  - no arithmetic parallelism; throughput-preserving decoupling buffer.

#### M43. `skid_buffer.sv`

- Language: SystemVerilog RTL
- Physical instances: many parameterized instances
- Purpose: Two-entry timing-isolation buffer for narrow control or sideband
  paths where a full FIFO is unnecessary.
- Inputs / buses:
  - ready/valid source interface
  - clock/reset
- Outputs / buses:
  - ready/valid sink interface
- Parallelism:
  - no arithmetic parallelism; timing-closure helper only.

#### M44. `descriptor_fifo.sv`

- Language: SystemVerilog RTL
- Physical instances: many parameterized instances
- Purpose: Small FIFO used for DMA descriptors, debug descriptors, and generated
  token write descriptors so control never stalls burst generation.
- Inputs / buses:
  - descriptor payload
  - descriptor valid/ready handshake
  - clock/reset
- Outputs / buses:
  - queued descriptor payload
  - descriptor valid/ready handshake
- Parallelism:
  - no arithmetic parallelism; control-path decoupling only.

---

## 6. Shared GEMM Engine Logical Modes

The following logical operations are not separate physical GEMM blocks. They are
all different scheduled modes of `shared_gemm_engine.sv`.

| GEMM mode ID | Logical operation | Input shape | Weight / other operand shape | Output shape |
|---|---|---|---|---|
| `GEMM_Q` | Q projection | `[seq_tile][2048]` | `[2048][2048]` | `[seq_tile][2048]` |
| `GEMM_K` | K projection | `[seq_tile][2048]` | `[2048][256]` | `[seq_tile][256]` |
| `GEMM_V` | V projection | `[seq_tile][2048]` | `[2048][256]` | `[seq_tile][256]` |
| `GEMM_SCORE` | attention scores | `[q_rows][64]` | `[64][k_rows]` | `[q_rows][k_rows]` |
| `GEMM_WEIGHTED_SUM` | weighted sum with V | `[q_rows][k_rows]` | `[k_rows][64]` | `[q_rows][64]` |
| `GEMM_O` | O projection | `[seq_tile][2048]` | `[2048][2048]` | `[seq_tile][2048]` |
| `GEMM_GATE` | gate projection | `[seq_tile][2048]` | `[2048][5632]` | `[seq_tile][5632]` |
| `GEMM_UP` | up projection | `[seq_tile][2048]` | `[2048][5632]` | `[seq_tile][5632]` |
| `GEMM_DOWN` | down projection | `[seq_tile][5632]` | `[5632][2048]` | `[seq_tile][2048]` |
| `GEMM_LM_HEAD` | LM head | `[1][2048]` or `[seq_tile][2048]` | `[2048][vocab_tile]` | `[1][vocab_tile]` or `[seq_tile][vocab_tile]` |

Notes:

- Prefill uses sequence tiles of 64 tokens.
- Decode uses query length 1 and key length `cache_len + 1`.
- All GEMM modes use the same lane array, accumulator bank, and tile buffers.

---

## 7. Parallelism Plan By Subsystem

### 7.1 Required Parallelism

- GEMM lane parallelism:
  - `512` INT8 lanes in `shared_gemm_engine.sv`
- Banking:
  - banked on-chip buffers for weights, activations, K/V, scores, and LM-head tiles
- HBM parallelism:
  - striped weight and cache traffic across dedicated pseudo-channel groups
- Pipeline overlap:
  - weight fetch overlaps with compute
  - KV-cache fetch/write overlaps with compute where scheduling allows
  - ping/pong buffering hides memory latency
- Reduction parallelism:
  - RMSNorm reduction tree
  - softmax reduction tree
  - argmax compare tree

### 7.2 Explicitly Not Used In The First Implementation

- full replication of all 22 decoder layers
- separate GEMM engine per projection
- top-k or top-p hardware sampling
- generic NoC-style dynamic crossbar

---

## 8. Microarchitecture Notes

This section freezes the implementation-level decisions that sit underneath the
high-level architecture. The architecture defines what blocks exist and how they
connect. The microarchitecture defines how those blocks are built internally:
buffer depth, tile shape, issue policy, interface adaptation, reduction order,
and timing-isolation strategy.

These notes affect the design immediately because they determine:

- which helper modules must exist
- how wide the internal buses are
- how the testbenches must drive and observe the design
- how much buffering is required between modules
- how scheduling and HBM access are coded from the first line of RTL/HLS

### 8.1 Clocking, Reset, And Clock-Domain Policy

- The first implementation uses a single synchronous user clock domain:
  `ap_clk`.
- There are no user-managed CDC FIFOs in the first implementation.
- All RTL modules, HLS wrappers, DMA paths, and scheduler logic are synchronous
  to `ap_clk`.
- `ap_rst_n` is synchronized to `ap_clk` before use in every module.
- Any clock-domain handling hidden inside vendor AXI/HBM infrastructure is
  treated as platform IP behavior, not user RTL.

### 8.2 AXI / HBM Normalization Policy

- The external `m_axi_pc00` .. `m_axi_pc31` interface width is inherited from
  the selected U55C shell and exposed as `HBM_SHELL_DATA_W`.
- The internal DMA datapath width is fixed to `DMA_BEAT_W = 256` bits.
- Width pack/unpack logic is implemented inside `hbm_port_router.sv`; there is
  no separate top-level width-adapter block.
- Burst policy is fixed as:
  - bulk weights / KV / LM-head tiles: 16-beat bursts
  - scales / gamma / metadata: 4-beat bursts
  - prompt tokens / status / debug headers: 1-beat bursts
- Outstanding AXI transactions are fixed as:
  - up to 16 reads in flight per HBM pseudo-channel group
  - up to 8 writes in flight per HBM pseudo-channel group
- Request completion is FIFO-ordered per internal DMA client stream.

### 8.3 FIFO, Skid-Buffer, And Descriptor-Queue Policy

- Every ready/valid boundary between major modules is elastic.
- `stream_fifo.sv` is required between:
  - DMA readers and `tile_buffer_bank.sv`
  - `tile_buffer_bank.sv` and `shared_gemm_engine.sv`
  - `tile_buffer_bank.sv` and HLS wrappers
  - `gemm_result_router.sv` and downstream writeback / reduction consumers
- The default `stream_fifo.sv` depth is `STREAM_FIFO_DEPTH = 4`.
- `skid_buffer.sv` with `SKID_BUFFER_DEPTH = 2` is used on narrow control,
  sideband, and status paths where timing isolation is required but full storage
  is not.
- `descriptor_fifo.sv` with `DESC_FIFO_DEPTH = 8` is mandatory for:
  - weight DMA descriptors
  - KV read descriptors
  - KV write descriptors
  - LM-head tile descriptors
  - debug write descriptors
  - generated-token write descriptors
- Debug capture uses a dedicated FIFO with `DEBUG_FIFO_DEPTH = 32`.
- Debug buffering is isolated from performance-critical DMA issue credit.

### 8.4 Tile Geometry And Edge-Tile Handling

- The shared GEMM engine tile tuple is fixed to:
  - `M_TILE = 16`
  - `N_TILE = 32`
  - `K_TILE = 64`
- Attention-score tiling is fixed to:
  - `SCORE_Q_TILE = 16`
  - `SCORE_K_TILE = 64`
  - `SCORE_ROWS_PER_CHUNK = 8`
  - `SCORE_CHUNKS_PER_TILE = 2`
- RoPE head-slice tiling is fixed to:
  - `ROPE_CHUNK_TOKENS = 8`
  - `HEAD_DIM = 64`
- LM-head vocabulary processing is fixed to:
  - `VOCAB_TILE = 128`
- Prefill input tiling is fixed to `SEQ_TILE = 64`.
- Decode uses query length `1`, but the same hardware path is used with masked
  partial tiles.
- Partial tiles at sequence tails, vocabulary tails, or channel tails are
  represented by valid-element masks carried in `*_tag`.
- Inactive lanes are forced idle inside `shared_gemm_engine.sv`, and masked
  writeback suppresses invalid output elements.
- Tile sizes are not runtime-programmable in the first implementation.
- Score chunks use the fixed row-major lane mapping:
  - `lane = query_row_local * SCORE_K_TILE + key_col_local`
- For score chunks, `elem_count` represents `query_row_count * SCORE_K_TILE`.
  Tail key columns are represented by explicit causal-mask fill, not by a
  ragged elem-count rectangle.

### 8.5 Buffer Banking And Interleaving

- Every logical ping/pong tile store is split into `TILE_BUFFER_BANKS = 16`
  physical banks.
- One bank transfer corresponds to:
  - `BANK_SLICE_INT8 = 32` INT8 elements, or
  - `BANK_SLICE_INT32 = 8` INT32 elements
- Lane-to-bank mapping is fixed:
  - bank 0 -> lanes 0..31
  - bank 1 -> lanes 32..63
  - bank 2 -> lanes 64..95
  - ...
  - bank 15 -> lanes 480..511
- A full 512-lane vector read is therefore one parallel access across all 16
  banks.
- Activation, weight, KV, score, and LM-head buffers use physically separate
  bank groups. They do not share banks.
- The scheduler never issues two accesses to the same bank of the same buffer
  set in one cycle.

### 8.5a GEMM Lane Packing Contract

- The shared GEMM engine uses a fixed row-major output-lane mapping for one
  `M_TILE x N_TILE` tile.
- Lane index is:
  - `lane = m_local * N_TILE + n_local`
- For the fixed first-implementation tile tuple:
  - `m_local` ranges `0..15`
  - `n_local` ranges `0..31`
  - lane `0` is output `(0, 0)`
  - lane `31` is output `(0, 31)`
  - lane `32` is output `(1, 0)`
  - lane `511` is output `(15, 31)`
- During the K loop, each lane accumulates one output element of the tile.
- At K-step `k_local`, the packed operand values presented to
  `shared_gemm_engine.sv` are:
  - `act_lane[lane] = act_tile[m_local, k_local]`
  - `wt_lane[lane]  = wt_tile[k_local, n_local]`
- This lane-packing rule is the contract used by:
  - `shared_gemm_engine.sv`
  - future trace-backed GEMM testbenches
  - `model/export_fpga_vectors.py`
- Partial tiles use the same mapping, but only the first
  `active_lane_count = m_count * n_count` lanes are valid.
- Inactive lanes are zero-filled in exported fixtures and are masked by
  `elem_count` in RTL.

### 8.6 Prefetch, Overlap, And Issue Order

- One-tile lookahead prefetch is mandatory for:
  - decoder-layer weight tiles
  - KV-cache tiles
  - LM-head vocabulary tiles
- While one ping/pong bank set is consumed by compute, the opposite bank set is
  filled by DMA.
- Bank-role swap occurs only on tile boundaries.
- `gemm_op_scheduler.sv` is allowed to issue the next DMA descriptor as soon as
  the next tile address and mode are known. It does not wait for current compute
  to fully finish before launching prefetch.
- LM-head reduction overlaps the fetch of the next vocabulary tile.
- Debug DMA requests are always lowest priority.

### 8.7 Head-Group And Attention Scheduling

- Q, K, and V projection GEMMs operate on full hidden-state tiles.
- The attention score, mask, softmax, and weighted-sum path processes one KV
  head group at a time: `HEAD_GROUP_PAR = 1`.
- For each KV head, the associated 8 query heads are processed sequentially in
  fixed head-major order.
- There is no parallel multi-head score engine in the first implementation.
- The attention output tile is assembled incrementally as head slices complete.
- `GEMM_O` begins only after the full 2048-wide attention output tile for the
  current sequence tile is assembled.

### 8.8 Numeric-Format Conversion Boundaries

- Quantized RTL buses carry:
  - INT8 activations
  - INT8 weights
  - INT32 accumulators
- All HLS nonlinear kernels operate on internal fixed-point format
  `FIXED_POINT_FMT = Q16.16`.
- HLS wrappers perform all format conversion between quantized RTL streams and
  fixed-point HLS streams.
- `rmsnorm_wrapper.sv` converts incoming INT8/INT32 tiles to `Q16.16`, invokes
  the HLS core, and emits INT8 tiles plus scale metadata for downstream stages.
- `rope_unit.sv` uses `Q16.16` internally for rotation math and emits INT8 Q/K
  tiles for the score GEMM path.
- RoPE preserves the existing Q or K activation scale, so the same static
  activation scale applies before and after rotation.
- `softmax_wrapper.sv` converts masked score tiles to `Q16.16`, invokes the HLS
  softmax core, and emits nonnegative INT8-compatible probability bytes in the
  range `[0, 127]` using fixed probability scale `1/127`.
- `silu_wrapper.sv` converts gate tiles to `Q16.16`, invokes the HLS SiLU core,
  and emits INT8 gate activations for `elementwise_mul.sv`.
- `elementwise_mul.sv` performs `INT8 x INT8 -> INT32`.
- `requantize_unit.sv` is always used after `elementwise_mul.sv` before
  `GEMM_DOWN`.
- Requantization policy is fixed to:
  - symmetric INT8 range `[-127, 127]`
  - zero point `0`
  - round-to-nearest-even
  - saturating clamp on overflow
- Softmax probability output is the one exception to signed symmetry: it is
  clamped to `[0, 127]` because probabilities are nonnegative.

### 8.9 Fixed Schedule Order Inside One Decoder Layer

- The reused decoder-layer engine executes the following strict order:
  1. pre-attention RMSNorm
  2. `GEMM_Q`
  3. `GEMM_K`
  4. `GEMM_V`
  5. RoPE on Q and K
  6. KV-cache write for the current token positions
  7. for each KV head group:
     Q-head slice selection -> score GEMM -> causal mask -> softmax ->
     weighted-sum GEMM -> write assembled attention slice
  8. `GEMM_O`
  9. residual add
  10. requantize
  11. pre-FFN RMSNorm
  12. `GEMM_GATE`
  13. `GEMM_UP`
  14. SiLU on gate path
  15. elementwise multiply with up path
  16. requantize
  17. `GEMM_DOWN`
  18. residual add
  19. requantize
- `GEMM_GATE` and `GEMM_UP` are serialized on the single shared GEMM engine.
- There is no gate/up parallel execution in the first implementation.

### 8.10 HLS Implementation Targets

- `rmsnorm_core_hls.cpp`, `softmax_core_hls.cpp`, and `silu_core_hls.cpp`
  target initiation interval `II = 1` on their inner vector-processing loops.
- The HLS vector chunk width is fixed to 32 elements per cycle so it matches
  the 16-bank buffer organization and 256-bit bank slice.
- HLS cores use fixed-point arithmetic only. There is no floating-point datapath
  in the production implementation.
- The default HLS stream element type is `ap_fixed<32,16>`.
- Wider local accumulators are allowed inside the HLS core implementation where
  mathematically required, but the wrapper-visible stream format remains
  `ap_fixed<32,16>`.
- HLS implementations do not allocate dynamic memory.

### 8.11 Debug, Performance Isolation, And Failure Policy

- Debug capture is disabled by default.
- When enabled, debug capture happens only on tile boundaries selected by
  `debug_layer_sel`, `debug_step_sel`, and `block_id`.
- Debug traffic uses `PC31` exclusively.
- Debug capture is not allowed to stall the compute datapath.
- If the debug FIFO is full, the requested capture is dropped and a sticky
  debug-overflow status bit is set in the status block.

### 8.12 Utility Primitives That Must Exist

- `stream_fifo.sv`
- `skid_buffer.sv`
- `descriptor_fifo.sv`
- width pack/unpack logic inside `hbm_port_router.sv`
- valid-mask generation and masked-writeback logic inside the scheduler / router
  path

There are no remaining open microarchitecture choices in this file. Module code
should follow the rules above unless this document is explicitly revised first.

---

## 9. Minimum Module Bring-Up Order

This is the recommended coding order after this file is accepted.

1. `tinyllama_u55c_kernel_top.sv`
2. `axi_lite_ctrl_slave.sv`
3. `kernel_reg_file.sv`
4. `prefill_decode_controller.sv`
5. `layer_controller.sv`
6. `hbm_port_router.sv`
7. `tile_buffer_bank.sv`
8. `weight_dma_reader.sv`
9. `prompt_token_reader.sv`
10. `embedding_lookup.sv`
11. `shared_gemm_engine.sv`
12. `gemm_operand_router.sv`
13. `gemm_result_router.sv`
14. `rope_lut_rom.sv`
15. `rope_unit.sv`
16. `gqa_router.sv`
17. `causal_mask_unit.sv`
18. `rmsnorm_wrapper.sv` + `rmsnorm_core_hls.cpp`
19. `softmax_wrapper.sv` + `softmax_core_hls.cpp`
20. `silu_wrapper.sv` + `silu_core_hls.cpp`
21. `residual_add.sv`
22. `requantize_unit.sv`
23. `elementwise_mul.sv`
24. `kv_cache_manager.sv`
25. `kv_cache_dma_reader.sv`
26. `kv_cache_dma_writer.sv`
27. `lm_head_controller.sv`
28. `argmax_reduction.sv`
29. `generated_token_writer.sv`
30. `debug_capture_mux.sv`
31. `debug_dma_writer.sv`
32. `host_cmd_status_mgr.sv`
33. `stream_fifo.sv`
34. `skid_buffer.sv`
35. `descriptor_fifo.sv`

---

## 10. Frozen Implementation Decisions That Were Previously Open

The following choices are now fixed and no longer treated as open items:

1. HBM interface width policy
   - The external HBM AXI data width is platform-defined and represented as
     `HBM_SHELL_DATA_W`.
   - The internal DMA datapath width is fixed to `DMA_BEAT_W = 256`.
   - Width conversion is implemented inside `hbm_port_router.sv`.

2. Shared GEMM tile tuple
   - `M_TILE = 16`
   - `N_TILE = 32`
   - `K_TILE = 64`

3. Embedding-table storage format
   - Token embeddings are stored in HBM as FP16 row-major vectors.
   - `embedding_lmhead_dma_reader.sv` fetches FP16 rows.
   - `embedding_quantizer.sv` converts fetched rows to INT8 activation tiles.
   - Embedding-output quantization uses the same static per-block-output scale
     policy defined in `design_decisions.txt`.

If any of the fixed choices above are intentionally changed, this file must be
updated before RTL or HLS code diverges from the documented contract.

---

## 11. Verification Trace Policy

Trace-backed verification follows `golden_trace_plan.md`.

The policy is fixed as:

1. Phase 0, Phase 1, and Phase 2
   - Directed smoke tests are mandatory.
   - Golden traces are not required.

2. Phase 3
   - Directed smoke tests remain mandatory.
   - Golden traces are recommended for arithmetic-heavy blocks such as
     `shared_gemm_engine.sv` and `requantize_unit.sv`.

3. Phase 4, Phase 5, and Phase 6
   - Every math or dataflow block must gain at least one trace-backed test.

4. Phase 7, Phase 8, and Phase 9
   - Integration must be checked against exported decoder-layer, prefill/decode,
     and LM-head traces before the phase is treated as hardened.
