# Implementation Checklist - TinyLlama U55C FPGA Accelerator

This document turns the frozen architecture and microarchitecture into a
practical coding plan.

Use the documents in this order:

1. `design_decisions.txt`
   System-level implementation contract.
2. `modules.md`
   Module inventory, interfaces, RTL/HLS split, and microarchitecture rules.
3. `implementation_checklist.md`
   File-by-file coding order, dependencies, stub strategy, and verification plan.

This file is intentionally execution-oriented. It answers:

- which source files should exist
- where each file should live
- what each file is responsible for
- what must be stubbed first
- what each file depends on
- how each file is first verified

Trace-backed verification policy is defined in `golden_trace_plan.md`.

---

## 1. Implementation Rules

- Do not overwrite the existing legacy validation files in `rtl/` during the
  first pass:
  - `rtl/control_fsm.sv`
  - `rtl/top.sv`
  - `rtl/mac_unit.sv`
  - `rtl/mac_array.sv`
  - `rtl/tb_control_fsm.sv`
  - `rtl/tb_top.sv`
  - `rtl/tb_mac_array.sv`
- The new TinyLlama implementation lives in new subdirectories under `rtl/`
  and `hls/`.
- Every RTL module must compile as a standalone leaf before it is integrated.
- Every HLS kernel must pass C simulation before wrapper integration.
- The first implementation pass is `stub-first, then functional, then optimized`.
- Each phase below has explicit entry and exit criteria.
- If a file name here conflicts with a later code change, `modules.md` and this
  checklist must be updated before implementation drifts.
- Directed smoke tests are mandatory for control and plumbing modules.
- Golden traces are not a gate for Phase 0, Phase 1, or Phase 2.
- Golden traces are recommended for Phase 3 arithmetic blocks.
- Golden traces are required from Phase 4 onward for math/dataflow blocks and
  for Phase 7-9 integration gates.

---

## 2. Planned Source Tree

The following tree is the planned implementation layout. The directories do not
all exist yet; this checklist defines them before code creation begins.

```text
rtl/
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
    gemm_op_scheduler.sv
    shared_gemm_engine.sv
    mac_lane.sv
    accumulator_bank.sv
    gemm_operand_router.sv
    gemm_result_router.sv
    rope_lut_rom.sv
    rope_unit.sv
    gqa_router.sv
    causal_mask_unit.sv
    residual_add.sv
    requantize_unit.sv
    elementwise_mul.sv
    lm_head_controller.sv
    argmax_reduction.sv
    debug_capture_mux.sv
  nonlinear/
    rmsnorm_wrapper.sv
    softmax_wrapper.sv
    silu_wrapper.sv
  top/
    tinyllama_u55c_kernel_top.sv
  tb/
    tb_stream_fifo.sv
    tb_descriptor_fifo.sv
    tb_axi_lite_ctrl_slave.sv
    tb_prefill_decode_controller.sv
    tb_hbm_port_router.sv
    tb_tile_buffer_bank.sv
    tb_shared_gemm_engine.sv
    tb_requantize_unit.sv
    tb_rope_unit.sv
    tb_causal_mask_unit.sv
    tb_argmax_reduction.sv
    tb_decoder_layer_smoke.sv
    tb_prefill_decode_smoke.sv
    tb_kernel_top_smoke.sv
hls/
  common/
    fixed_types.hpp
    stream_utils.hpp
  rmsnorm/
    rmsnorm_core_hls.cpp
    rmsnorm_core_hls.hpp
    tb_rmsnorm.cpp
  softmax/
    softmax_core_hls.cpp
    softmax_core_hls.hpp
    tb_softmax.cpp
  silu/
    silu_core_hls.cpp
    silu_core_hls.hpp
    tb_silu.cpp
model/
  export_fpga_vectors.py
```

---

## 3. Stub-First Strategy

The first pass for each module is not full functionality. The first pass should:

- define the final port list
- define the parameters or package imports
- implement legal ready/valid behavior
- return deterministic placeholder outputs
- compile cleanly with its immediate dependencies

Only after the stub phase passes do we add real datapath logic.

Required order for each major block:

1. Port-complete stub
2. Local unit test / smoke test
3. Functional implementation
4. Local verification with golden vectors
5. Integration into parent module

---

## 4. Phase Overview

| Phase | Goal |
|---|---|
| 0 | Create package files, directory structure, and common conventions |
| 1 | Build utility primitives and control plane skeleton |
| 2 | Build memory and buffer infrastructure |
| 3 | Build GEMM datapath core |
| 4 | Build attention-specific compute path |
| 5 | Build nonlinear wrappers and HLS kernels |
| 6 | Build FFN path, LM head, and token selection |
| 7 | Integrate full decoder-layer engine |
| 8 | Integrate prefill/decode runtime |
| 9 | Add debug, golden-vector export, and top-level smoke tests |

---

## 5. File-By-File Checklist

## Phase 0 - Packages, Typedefs, And Layout

Exit criteria:

- package files compile
- all later modules can import shared constants and bus types
- no new TinyLlama RTL file is placed in the legacy flat `rtl/` root

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 0.1 | `rtl/common/tinyllama_pkg.sv` | RTL package | Constants, enums, tile sizes, GEMM mode IDs, block IDs, stop reasons, HBM allocation IDs | none | define parameters and enums only | compile import from a dummy module |
| 0.2 | `rtl/common/tinyllama_bus_pkg.sv` | RTL package | Packed structs and typedefs for `token_bus`, `act_bus`, `wt_bus`, `acc_bus`, `scale_bus`, `dbg_bus`, and common tags | `tinyllama_pkg.sv` | define bus structs and helper typedefs | compile import from a dummy module |
| 0.3 | `hls/common/fixed_types.hpp` | HLS header | Shared fixed-point typedefs, scalar widths, helper aliases | none | declare `ap_fixed<32,16>` aliases | compile include from dummy HLS test |
| 0.4 | `hls/common/stream_utils.hpp` | HLS header | Shared stream helpers, saturate helpers, round-to-nearest-even helpers | `fixed_types.hpp` | helper signatures only | compile include from dummy HLS test |

## Phase 1 - Utility Primitives And Control Plane Skeleton

Exit criteria:

- all control files compile together
- ready/valid primitives are verified
- controller stubs can emit deterministic phase sequencing without datapath logic

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 1.1 | `rtl/common/stream_fifo.sv` | RTL | Elastic wide-data FIFO | package files | parameterized synchronous FIFO | `rtl/tb/tb_stream_fifo.sv` |
| 1.2 | `rtl/common/skid_buffer.sv` | RTL | Two-entry timing isolation buffer | package files | generic ready/valid skid buffer | simple self-check inside FIFO TB or dedicated smoke TB |
| 1.3 | `rtl/common/descriptor_fifo.sv` | RTL | FIFO for DMA and write descriptors | package files | fixed-width FIFO with depth parameter | `rtl/tb/tb_descriptor_fifo.sv` |
| 1.4 | `rtl/control/axi_lite_ctrl_slave.sv` | RTL | AXI4-Lite register access front-end | packages | register read/write stub | `rtl/tb/tb_axi_lite_ctrl_slave.sv` |
| 1.5 | `rtl/control/kernel_reg_file.sv` | RTL | Launch/status register storage | packages, AXI-Lite slave | register map only | compile and drive from AXI-Lite TB |
| 1.6 | `rtl/control/host_cmd_status_mgr.sv` | RTL | Command/status block DMA manager | packages, FIFOs | stub descriptor issue and stub status writeback | compile-level smoke with fake HBM responses |
| 1.7 | `rtl/control/prefill_decode_controller.sv` | RTL | Top runtime FSM | packages | state machine stub for `IDLE -> PREFILL -> DECODE -> DONE` | `rtl/tb/tb_prefill_decode_controller.sv` |
| 1.8 | `rtl/control/layer_controller.sv` | RTL | Iterates layer index and block schedule tags | packages, controller | stub layer loop `0..21` | small self-check testbench or integrated controller TB |
| 1.9 | `rtl/control/stop_condition_unit.sv` | RTL | EOS and max-token stopping | packages | pure combinational/registered stop check | compile with controller TB |

## Phase 2 - Memory, DMA, And Buffer Infrastructure

Exit criteria:

- DMA request/response paths compile
- banked tile storage can accept writes and produce deterministic reads
- no compute block directly accesses HBM without going through the router

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 2.1 | `rtl/memory/hbm_port_router.sv` | RTL | AXI arbitration, width conversion, DMA routing to fixed pseudo-channel groups | packages, FIFOs | stub router with one read and one write client path | `rtl/tb/tb_hbm_port_router.sv` |
| 2.2 | `rtl/memory/prompt_token_reader.sv` | RTL | Reads prompt token IDs from `PC30` | packages, descriptor FIFO, router | burst-read descriptor generator | compile with fake HBM stream |
| 2.3 | `rtl/memory/generated_token_writer.sv` | RTL | Writes generated token IDs back to `PC30` ring buffer | packages, descriptor FIFO, router | single-token write path | compile with fake HBM sink |
| 2.4 | `rtl/memory/weight_dma_reader.sv` | RTL | Reads layer weight tiles from `PC00-PC15` | packages, descriptor FIFO, router | descriptor-to-stream stub | fake HBM tile return smoke |
| 2.5 | `rtl/memory/embedding_lmhead_dma_reader.sv` | RTL | Reads FP16 embedding rows, LM-head tiles, final gamma, and quant metadata | packages, descriptor FIFO, router | address decode + demux stub | compile-level smoke |
| 2.6 | `rtl/memory/kv_cache_dma_reader.sv` | RTL | Reads K/V cache rows from `PC22-PC29` | packages, descriptor FIFO, router | read-descriptor stub | compile-level smoke |
| 2.7 | `rtl/memory/kv_cache_dma_writer.sv` | RTL | Writes quantized K/V rows back to HBM | packages, descriptor FIFO, router | write-descriptor stub | compile-level smoke |
| 2.8 | `rtl/memory/debug_dma_writer.sv` | RTL | Writes debug captures into `PC31` | packages, descriptor FIFO, router | low-priority write path stub | compile-level smoke |
| 2.9 | `rtl/memory/scale_metadata_store.sv` | RTL | Stores activation, KV, and LM-head scale metadata | packages | multi-port scale RAM/register file | direct write/read TB or compile smoke |
| 2.10 | `rtl/memory/tile_buffer_bank.sv` | RTL | Ping/pong banked tile storage for activations, weights, K/V, score, LM-head | packages, FIFOs | banked memory with one read and one write side | `rtl/tb/tb_tile_buffer_bank.sv` |
| 2.11 | `rtl/memory/kv_cache_manager.sv` | RTL | Generates layer/head/token-based K/V descriptors | packages, controller | address generator stub | compile with reader/writer stubs |

## Phase 3 - Shared GEMM Datapath Core

Exit criteria:

- GEMM core runs with real tile loops
- accumulator clear/writeback is verified
- operand and result routers can drive/consume the GEMM core in multiple modes
- `model/export_fpga_vectors.py` can export at least one real Phase 3 GEMM case
  and one real Phase 3 requantization case under `sim/golden_traces/`
- before Phase 4 begins, `shared_gemm_engine.sv` and `requantize_unit.sv`
  should have a path to consume exported Phase 3 trace data

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 3.1 | `rtl/compute/mac_lane.sv` | RTL | One INT8xINT8->INT32 MAC lane | packages | combinational MAC | direct leaf TB or reuse old MAC concepts |
| 3.2 | `rtl/compute/accumulator_bank.sv` | RTL | 512-lane INT32 accumulator storage | packages | clear/write/read bank stub | compile with synthetic lane inputs |
| 3.3 | `rtl/compute/shared_gemm_engine.sv` | RTL | Reused INT8 GEMM engine | packages, MAC lane, accumulator bank | output-stationary tiled stub with deterministic lane firing | `rtl/tb/tb_shared_gemm_engine.sv` |
| 3.4 | `rtl/compute/gemm_operand_router.sv` | RTL | Selects input tile sources for active GEMM mode | packages, tile buffer bank | mode-based mux stub | compile smoke with forced modes |
| 3.5 | `rtl/compute/gemm_result_router.sv` | RTL | Routes GEMM results to correct consumers | packages, scale store | mode-based demux stub | compile smoke with synthetic `acc_bus` |
| 3.6 | `rtl/compute/gemm_op_scheduler.sv` | RTL | Issues GEMM modes and tile loop counters | packages, controller | deterministic schedule stub for one layer | integrate with GEMM smoke |
| 3.7 | `rtl/compute/requantize_unit.sv` | RTL | INT32 to INT8 requantization | packages | real arithmetic leaf | `rtl/tb/tb_requantize_unit.sv` |

## Phase 4 - Attention-Specific Compute Path

Exit criteria:

- Q/K/V -> RoPE -> score -> mask path compiles
- score tiling and head-group scheduling are represented correctly
- attention assembly path exists even if softmax is still stubbed

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 4.1 | `rtl/compute/rope_lut_rom.sv` | RTL | Stores RoPE sine/cosine tables | packages | ROM stub with deterministic entries | compile with rope TB |
| 4.2 | `rtl/compute/rope_unit.sv` | RTL | Applies RoPE to Q/K tiles | packages, ROM | first pass can bypass or identity-rotate under control | `rtl/tb/tb_rope_unit.sv` |
| 4.3 | `rtl/compute/gqa_router.sv` | RTL | Head-group routing from 4 KV heads to 32 Q heads | packages | address/tag routing stub | compile smoke with synthetic tags |
| 4.4 | `rtl/compute/causal_mask_unit.sv` | RTL | Applies pre-softmax causal mask | packages | real compare-and-mask leaf | `rtl/tb/tb_causal_mask_unit.sv` |

## Phase 5 - Nonlinear Wrappers And HLS Kernels

Exit criteria:

- each HLS kernel passes C simulation
- each RTL wrapper compiles against its HLS signature
- data type boundaries are consistent with `modules.md`

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 5.1 | `hls/rmsnorm/rmsnorm_core_hls.hpp` | HLS header | Declares RMSNorm kernel signature | HLS common headers | signature only | compile from TB |
| 5.2 | `hls/rmsnorm/rmsnorm_core_hls.cpp` | HLS | RMSNorm fixed-point kernel | headers | first pass can be identity transform with correct streams | `hls/rmsnorm/tb_rmsnorm.cpp` |
| 5.3 | `rtl/nonlinear/rmsnorm_wrapper.sv` | RTL | Wraps HLS RMSNorm kernel into RTL stream contract | packages, FIFOs | wrapper stub with deterministic passthrough | compile smoke with fixed interface |
| 5.4 | `hls/softmax/softmax_core_hls.hpp` | HLS header | Declares softmax kernel signature | HLS common headers | signature only | compile from TB |
| 5.5 | `hls/softmax/softmax_core_hls.cpp` | HLS | Softmax fixed-point kernel | headers | first pass can normalize a tiny known vector | `hls/softmax/tb_softmax.cpp` |
| 5.6 | `rtl/nonlinear/softmax_wrapper.sv` | RTL | Wraps HLS softmax kernel | packages, FIFOs | wrapper stub | compile smoke with masked-score tiles |
| 5.7 | `hls/silu/silu_core_hls.hpp` | HLS header | Declares SiLU kernel signature | HLS common headers | signature only | compile from TB |
| 5.8 | `hls/silu/silu_core_hls.cpp` | HLS | SiLU fixed-point kernel | headers | first pass can approximate with pass-through sign-preserving transform | `hls/silu/tb_silu.cpp` |
| 5.9 | `rtl/nonlinear/silu_wrapper.sv` | RTL | Wraps HLS SiLU kernel | packages, FIFOs | wrapper stub | compile smoke |

## Phase 6 - FFN Path, LM Head, Argmax, And Debug

Exit criteria:

- FFN path is structurally complete
- LM-head tiled path is present
- argmax and debug infrastructure have local tests

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 6.1 | `rtl/compute/embedding_lookup.sv` | RTL | Token ID to FP16 embedding-row fetch control | packages, embedding DMA | row fetch stub | compile smoke with fake embedding rows |
| 6.2 | `rtl/compute/embedding_quantizer.sv` | RTL | FP16 embedding row to INT8 activation tile | packages, scale store | first pass can emit clipped deterministic pattern | compile smoke |
| 6.3 | `rtl/compute/residual_add.sv` | RTL | Elementwise residual accumulation | packages | real leaf arithmetic | integrate with requantize TB or local smoke |
| 6.4 | `rtl/compute/elementwise_mul.sv` | RTL | SwiGLU multiply | packages | real leaf arithmetic | compile smoke |
| 6.5 | `rtl/compute/lm_head_controller.sv` | RTL | Vocabulary-tile scheduling for LM head | packages, scheduler, scale store | tile-descriptor issue stub | compile with argmax smoke |
| 6.6 | `rtl/compute/argmax_reduction.sv` | RTL | Greedy argmax across vocab tiles | packages | real hierarchical reduction leaf | `rtl/tb/tb_argmax_reduction.sv` |
| 6.7 | `rtl/compute/debug_capture_mux.sv` | RTL | Selectable internal debug source mux | packages | source-select stub | compile smoke |

## Phase 7 - Decoder-Layer Integration

Exit criteria:

- one full decoder-layer pass is structurally wired
- data moves from pre-attention norm through FFN residual path
- smoke test can execute one synthetic layer step

Files in this phase are integration files rather than brand-new primitives.

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 7.1 | `rtl/tb/tb_decoder_layer_smoke.sv` | TB | Synthetic smoke test for one reused decoder-layer path | phases 1-6 | instantiate stubs and verify schedule order only | run smoke after full layer wiring |
| 7.2 | `rtl/control/layer_controller.sv` | RTL update | Promote from stub to real per-layer orchestrator | all layer-level blocks | real block sequencing | decoder-layer smoke test |
| 7.3 | `rtl/compute/gemm_op_scheduler.sv` | RTL update | Promote from stub to real operation/tile scheduler | GEMM, buffers, layer controller | real mode issue order | decoder-layer smoke test |
| 7.4 | `rtl/compute/gemm_operand_router.sv` | RTL update | Real source selection for Q/K/V/O/FFN/LM-head modes | buffers, scheduler | real routing | decoder-layer smoke test |
| 7.5 | `rtl/compute/gemm_result_router.sv` | RTL update | Real result routing into Q/K/V, score, FFN, KV writeback, LM-head | requantize, buffers, scheduler | real routing | decoder-layer smoke test |

## Phase 8 - Full Runtime Integration

Exit criteria:

- prompt prefill path exists
- decode loop exists
- FPGA-side stop condition closes the loop
- top-level kernel compiles structurally

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 8.1 | `rtl/top/tinyllama_u55c_kernel_top.sv` | RTL | Final kernel top-level | everything | structural wiring only | compile-only top-level build |
| 8.2 | `rtl/tb/tb_prefill_decode_smoke.sv` | TB | Smoke test for prompt prefill plus a few decode steps with stubs | top-level dependencies except HLS accuracy | deterministic token-flow smoke | run after top-level wiring |
| 8.3 | `rtl/tb/tb_kernel_top_smoke.sv` | TB | Top-level kernel smoke including AXI-Lite launch and fake HBM model | top-level, control, memory | end-to-end structural smoke | run after full compile |
| 8.4 | `rtl/control/prefill_decode_controller.sv` | RTL update | Promote from state stub to real runtime controller | host/status, layer controller, stop unit | real state sequencing | prefill/decode smoke TB |
| 8.5 | `rtl/control/host_cmd_status_mgr.sv` | RTL update | Promote from descriptor stub to real command/status path | router, FIFOs | real status updates | top-level smoke TB |

## Phase 9 - Verification Collateral And Golden Traces

Exit criteria:

- software can export deterministic traces for RTL/HLS tests
- debug capture contract is testable
- each major module has at least one directed test path
- trace-backed acceptance follows `golden_trace_plan.md`

| Order | File | Type | Purpose | Depends on | First pass | First verification |
|---|---|---|---|---|---|---|
| 9.1 | `model/export_fpga_vectors.py` | Python support | Exports deterministic TinyLlama golden traces for GEMM, RMSNorm, softmax, RoPE, one decoder layer, LM-head, and runtime smoke cases | existing TinyLlama model scripts, `golden_trace_plan.md` | prompt-independent trace export script | run locally and inspect generated files under `sim/golden_traces/` |
| 9.2 | `rtl/tb/tb_requantize_unit.sv` | TB | Directed arithmetic verification for requantization | requantize unit | compare against exported vectors | run local simulation |
| 9.3 | `rtl/tb/tb_shared_gemm_engine.sv` | TB | Directed GEMM tile verification | GEMM core | compare against exported vectors | run local simulation |
| 9.4 | `rtl/tb/tb_rope_unit.sv` | TB | Directed RoPE vector verification | rope unit | compare against exported vectors | run local simulation |
| 9.5 | `rtl/tb/tb_causal_mask_unit.sv` | TB | Prefill and decode mask verification | mask unit | directed cases | run local simulation |

---

## 6. Build Order Summary

This is the recommended implementation sequence in the actual coding sessions:

1. Phase 0 package files
2. Phase 1 utility FIFOs and control stubs
3. Phase 2 HBM router, DMA readers/writers, and tile buffer bank
4. Phase 3 GEMM core and requantizer
5. Minimal Phase 3 golden-trace export and arithmetic hardening
6. Phase 4 RoPE, GQA, and causal mask
7. Phase 5 HLS kernels and wrappers
8. Phase 6 embedding path, FFN leaf blocks, LM head, argmax, debug mux
9. Phase 7 decoder-layer integration
10. Phase 8 top-level runtime integration
11. Phase 9 expanded golden-trace export and directed verification

Do not start the final top-level kernel file before Phases 1-3 compile cleanly.

---

## 7. First Files To Create

The first five files to implement are:

1. `rtl/common/tinyllama_pkg.sv`
2. `rtl/common/tinyllama_bus_pkg.sv`
3. `rtl/common/stream_fifo.sv`
4. `rtl/common/skid_buffer.sv`
5. `rtl/common/descriptor_fifo.sv`

The reason is simple: every later module depends on stable constants, stable bus
types, and stable ready/valid buffering.

---

## 8. Legacy Validation Core Policy

The current root-level legacy RTL remains useful and should stay in place during
the early implementation phases:

- `rtl/control_fsm.sv`
- `rtl/top.sv`
- `rtl/mac_unit.sv`
- `rtl/mac_array.sv`
- `rtl/tb_control_fsm.sv`
- `rtl/tb_top.sv`
- `rtl/tb_mac_array.sv`

They are not the production TinyLlama runtime. They remain as:

- arithmetic sanity references
- simulation bring-up references
- simple fallback examples for lane-level debugging

Do not merge the new TinyLlama production modules into those files.

---

## 9. Definition Of Done For The Checklist

This checklist is considered complete enough to begin coding when:

- every file listed above has an agreed location and purpose
- every file has an identified dependency set
- every major file has a defined first-pass implementation target
- every major file has a first verification target
- no module creation order depends on undocumented assumptions

This condition is now met.
