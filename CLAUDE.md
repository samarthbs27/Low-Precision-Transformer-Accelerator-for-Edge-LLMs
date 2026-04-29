# CLAUDE.md — FPGA Transformer Accelerator Project

Auto-loaded by Claude Code. Tracks all decisions, progress, and context for this project.

---

## Who I Am

**Samarth** — responsible for **DATAFLOW, TILING, AND CONTROL** (Person 3 of 4).

Teammates:
- **Satyarth** — Model, Quantization, Ground Truth (`model/tinyllama.py`, `model/tinyllama_gemm_int8.py`, test data)
- **Rijul** — MAC Unit and Parallel Compute Core (`mac_unit.sv`, `mac_array.sv`, softmax, RMSNorm, RoPE, SiLU)
- **Om** — U55C Integration and Host Interface (XRT kernel, PCIe host program, host launch / readback)

---

## Project in One Line

> Run full TinyLlama 1.1B inference on a Xilinx Alveo U55C FPGA — prompt prefill + autoregressive decode, one reused decoder-layer engine (RMSNorm → QKV → RoPE → attention → SwiGLU FFN → residual adds), KV cache in HBM, FPGA-side token selection, weights streaming from HBM.

> **Pivoted (2026-04-06):** Professor feedback said a pure GEMM accelerator was too simple. Scope expanded to a full TinyLlama decoder layer with all dedicated blocks.
> **Clarified (2026-04-07):** Target model is TinyLlama 1.1B. Architecture is pre-norm (RMSNorm before each sub-block). FFN is SwiGLU (3 GEMMs + SiLU). One decoder layer instantiated on FPGA, reused 22× via host-driven loop.
> **Expanded (2026-04-10):** Full TinyLlama inference target. FPGA owns embedding lookup, 22-layer loop, KV-cache management, final RMSNorm, LM head, greedy argmax, and stop conditions. Host handles only tokenization, launch, and token readback.

---

## Current Status (as of 2026-04-28)

This file is primarily a project-history log. For the current source of truth,
follow `AGENTS.md` -> `docs/design_decisions.txt` -> `docs/modules.md`.

- Engine width:
  - Target architecture: `GEMM_LANES=512`, `M_TILE=16`, `VOCAB_TILE=128`.
  - Current synth/hardware profile: `GEMM_LANES=64`, `M_TILE=2`, `VOCAB_TILE=64`
    due to Vivado 2022.1 crashing on the 512-lane packed-struct widths.
- Hardware bring-up: U55C bitstreams are currently blocked by a platform shell
  mismatch; SOL provides an Alveo U280 target platform for real XRT runs.

---

## Platform & Architecture

| Property | Value |
|---|---|
| FPGA Board | Target: Xilinx Alveo U55C (PCIe accelerator). Current bring-up: Alveo U280 on SOL. |
| Host Interface | PCIe via XRT (x86 host) |
| Architecture Type | Full TinyLlama inference — prefill + autoregressive decode, one decoder-layer engine reused 22×, FPGA-owned control loop |
| Target Model | TinyLlama 1.1B (d_model=2048, d_ff=5632, 22 layers, 32Q/4KV GQA heads) |
| Compute Core | Parallel MAC Array, output-stationary, time-multiplexed across all 9 GEMMs |
| Lanes | Current synth/hw profile: 64 lanes; target: 512 lanes for TinyLlama throughput |
| Precision | INT8 × INT8 → INT32 accumulation; RMSNorm/softmax/SiLU in Q16.16 fixed-point (HLS); RoPE in Q16.16 (RTL LUT) |
| Normalization | RMSNorm (no mean subtraction — TinyLlama uses RMSNorm, not LayerNorm) |
| FFN | SwiGLU: gate_proj → up_proj (serialized on shared GEMM) → SiLU(gate) × up → down_proj |
| Positional Encoding | RoPE applied to Q and K (not additive, not to V) |
| Attention | Grouped-query attention: 32 Q heads, 4 KV heads, 8× repetition |
| Weight Storage | U55C: 16 GB HBM2 (target). U280: 8 GB HBM2 + 32 GB DDR4 (bring-up). One layer ~43.5 MB, all 22 layers ~957 MB. |
| Test Dimensions | N=64, K=64 (Phase 1); TinyLlama target: N/K up to 2048/5632 |

---

## My Deliverables (by week)

### Week 1
- [x] Block diagram (`docs/block_diagram.drawio` — visual, open with Draw.io VS Code extension)
- [x] Block diagram explanation with per-block signal tables (`docs/block_diagram.md`)
- [x] Tiling plan (documented in `docs/block_diagram.md`)
- [x] FSM state diagram with WRITE state (documented in `docs/block_diagram.md`)
- [x] Design review applied (all 4 issues + 1 addition fixed)
- [x] Re-export `docs/block_diagram.png` from updated `.drawio` (manual step)
- [x] Control FSM RTL (`rtl/control_fsm.sv`)
- [x] FSM testbench (`rtl/tb_control_fsm.sv`)

### Week 2
- [x] Pulled Rijul's `mac_unit.sv` and `mac_array.sv` — interface verified, compatible ✅
- [x] Top-level integration module (`rtl/top.sv`) wiring FSM → MAC array ✅
- [x] BRAM instantiation and address generation ✅
- [x] Satyarth's test vectors integrated — `tb_mac_array` passing all 7 checks ✅
- [x] Rewrote `docs/block_diagram.drawio` for full TinyLlama decoder layer (RMSNorm, RoPE, SwiGLU) ✅
- [x] `docs/design_decisions.txt` created — conservative decisions, open items, implemented decisions ✅
- [x] `docs/theory.md` rewritten for TinyLlama (RMSNorm, RoPE, GQA, SwiGLU, resource feasibility) ✅
- [x] `docs/block_diagram.md` updated — TinyLlama pre-norm pipeline, 9-GEMM host sequencer ✅

### Week 3
- [x] Phase 1 control skeleton implemented and verified (Codex) ✅ — see Session 8
- [x] `host_cmd_status_mgr.sv` (Phase 1.6) — HBM command/status DMA manager stub implemented and verified ✅
- [x] Phase 2 HBM memory subsystem implemented and verified (Codex) ✅ — see Session 10
- [x] Phase 3 compute engine implemented and verified (Codex) ✅ — see Session 11
- [x] Golden trace plan documented and wired into all project docs ✅ — see Session 11
- [x] `model/export_fpga_vectors.py` — Phase 3 real-model trace exporter (layer 0, prefill + decode, all 7 ops) ✅
- [x] `tb_requantize_unit.sv` and `tb_shared_gemm_engine.sv` consume exported golden traces ✅
- [x] Phase 4 attention-path blocks implemented and verified: `rope_lut_rom.sv`, `rope_unit.sv`, `gqa_router.sv`, `causal_mask_unit.sv` ✅
- [x] `docs/parallelism_tradeoffs.md` — U55C parallelism rationale documented ✅
- [x] Phase 5 nonlinear wrappers implemented and verified: `rmsnorm_wrapper.sv`, `softmax_wrapper.sv`, `silu_wrapper.sv` ✅
- [x] HLS kernels implemented: `rmsnorm_core_hls.cpp`, `softmax_core_hls.cpp`, `silu_core_hls.cpp` ✅
- [x] `export_fpga_vectors.py` expanded with Phase 4 (RoPE ROM, causal mask) and Phase 5 (rmsnorm/softmax/silu) export functions ✅
- [x] Residual adder RTL — `residual_add.sv`, one reused block (Samarth) ✅
- [x] Phase 6 embedding, output, and datapath completion verified: `embedding_lookup.sv`, `embedding_quantizer.sv`, `elementwise_mul.sv`, `residual_add.sv`, `argmax_reduction.sv`, `lm_head_controller.sv`, `debug_capture_mux.sv` ✅ — see Session 14
- [x] Phase 7 decoder-layer integration gate verified: `layer_controller.sv` rewrite, `gemm_op_scheduler.sv` dual-mode, router tag normalization, `tb_decoder_layer_smoke.sv` ✅ — see Session 15
- [x] Phase 8 runtime-core integration verified: `tinyllama_u55c_kernel_top.sv`, extended `prefill_decode_controller.sv`, `tb_prefill_decode_smoke.sv`, `tb_kernel_top_smoke.sv` — all 4 testbenches PASS ✅ — see Session 16
- [x] Phase 9 shell-seam closure: `tinyllama_u55c_shell_wrapper.sv`, `tb_kernel_top_acceptance.sv`, `tb_shell_wrapper_smoke.sv` — all 3 testbenches PASS ✅ — see Session 17
- [x] Phase 9b real inference closure (first slice): `runtime_embedding_frontend.sv`, updated `tinyllama_u55c_kernel_top.sv`, `tb_runtime_embedding_frontend.sv` — all 4 testbenches PASS ✅ — see Session 17
- [x] Phase 9c decoder datapath scaffold: `runtime_decoder_datapath.sv`, updated `tinyllama_u55c_kernel_top.sv`, `tb_runtime_decoder_datapath.sv` — all 7 testbenches PASS ✅ — see Session 19
- [x] Phase 9d LM head tail scaffold: `runtime_lm_head_tail.sv`, updated `tinyllama_u55c_kernel_top.sv`, `tb_runtime_lm_head_tail.sv` — all 10 testbenches PASS ✅ — see Session 19
- [x] Phase 9e final RMSNorm tail: `runtime_final_rmsnorm_tail.sv`, `rmsnorm_core_hls_ip.sv`, updated `tinyllama_u55c_kernel_top.sv`, `tb_runtime_final_rmsnorm_tail.sv` — all 8 testbenches PASS ✅ — see Sessions 20-21; `output_scale_i` timing bug fixed (static localparam 1.0); decode case + tag checks added
- [x] Phase 9f real LM head: `runtime_lm_head_tail.sv` rewritten with real TENSOR_LM_HEAD DMA + real `shared_gemm_engine` GEMM; `tb_runtime_lm_head_tail.sv` rewritten; updated `tinyllama_u55c_kernel_top.sv` (EMRD 3-way mux FRONTEND>FINAL>LM_HEAD); all 8 testbenches PASS ✅ — see Session 22; `target_token_i` dead code noted; `target_token_i` port + dead registers removed in hardening pass
- [x] Phase 9g decoder datapath tail: `runtime_decoder_datapath.sv` rewritten with two-pass deterministic tail (real `residual_add` + `requantize_unit`; DDP_TAIL_SEND/DDP_TAIL_WAIT states); `tb_runtime_decoder_datapath.sv` rewritten with exact output verification (rejects raw passthrough, mirrors tail math bit-exact); tb_runtime_decoder_datapath PASS at 172.375 µs ✅ — see Session 23; integration tests PASS ✅
- [x] Phase 9h decoder per-block update: `runtime_decoder_datapath.sv` rewritten: DDP_APPLY_SEND/DDP_APPLY_WAIT replace tail states; one tile updated per block via block-type-specific residual+requantize; odd-stride tile cursor guarantees all 64 tiles touched over 3212 blocks; `tb_runtime_decoder_datapath.sv` mirrors full DUT evolution (shadow tile state, cursor, signature tracking); all 10 testbenches PASS ✅ — see Session 24; unit PASS 234.055 µs, integration tests PASS ✅
- [x] Phase 9i FFN chain closure: `runtime_decoder_datapath.sv` expanded to 11-state FSM; DDP_SILU_APPLY (single-cycle) + DDP_MUL_SEND/MUL_WAIT (real `elementwise_mul` handshake); FFN anchor tile (`ffn_tile_anchor_q`) set at BLOCK_GATE, reused for UP/SILU/GLU_MUL/DOWN; FFN side buffers (gate/up/silu/mul tiles); `tb_runtime_decoder_datapath.sv` adds FFN shadow arrays + `mul_done_count==22` gate; all 10 testbenches PASS ✅ — see Session 25; unit PASS 233.835 µs (−0.22 µs from Ph9h: 22 SILU blocks × 1 saved cycle), integration tests PASS ✅
- [x] Phase 9j real SiLU path: `runtime_decoder_datapath.sv` rewritten: DDP_SILU_APPLY replaced by DDP_SILU_SEND/SCALE/ACT (3-state handshake through real `silu_wrapper`); `silu_core_hls_ip.sv` sim model added; `silu_done_w` wire exposed; `tb_runtime_decoder_datapath.sv` updated: `tb_silu_scalar` mirrors real FP SiLU math, `silu_done_count==22` gate added; all 7 freshly rerun testbenches PASS ✅ — see Session 26; unit PASS 234.935 µs (+1.1 µs from Ph9i: 22 SILU blocks × 5 extra state cycles × 10 ns), integration tests PASS ✅
- [x] Phase 9k real FFN projection GEMMs: `runtime_decoder_datapath.sv` expanded to 15-state FSM; DDP_GEMM_SEND/WAIT added for BLOCK_GATE/UP/DOWN via real `shared_gemm_engine`; BLOCK_DOWN two-phase (GEMM→APPLY): captures gemm result to `down_gemm_acc_q`, sets `down_apply_from_gemm_q`, then DDP_APPLY_SEND/WAIT runs real `residual_add` + `requantize_unit`; `gemm_acc_valid_w` exposed for testbench GATE/UP/DOWN count monitoring; `tb_runtime_decoder_datapath.sv` adds `tb_apply_projection_tile` + `tb_apply_down_update`; gate_gemm/up_gemm/down_gemm==22 gates added; all 5 freshly rerun testbenches PASS (XSIM for top-level, Icarus for unit) ✅ — see Session 27; unit PASS 235.375 µs (+0.44 µs from Ph9j: 22 DOWN blocks × 2 extra GEMM cycles × 10 ns), integration tests PASS ✅
- [x] Phase 9l attention-output subchain: `runtime_decoder_datapath.sv` adds `attn_weighted_tiles_q` + `attn_o_tiles_q` side buffers; BLOCK_WEIGHTED_SUM stages to `attn_weighted_tiles_q` (not hidden); BLOCK_O added to `is_ffn_projection_block()` — runs real `shared_gemm_engine` (reads weighted tiles, stores to `attn_o_tiles_q`); BLOCK_RESIDUAL1 uses `attn_o_tiles_q[anchor]` as update operand; `attn_o_tile_anchor_q`/`attn_o_stride_q` maintain chain tile coherence; `advance_tile_cursor_on_block()` updated (cursor advances at BLOCK_RESIDUAL1, not BLOCK_O); `tb_runtime_decoder_datapath.sv` adds `expected_weighted_tiles_q`, `expected_o_tiles_q`, `tb_apply_residual_stage_update`, `o_gemm_count==22` gate; all 6 testbenches PASS (Icarus unit + XSIM integration) ✅ — see Session 28; unit PASS 235.375 µs (same as Ph9k: DDP_APPLY_SEND/WAIT→DDP_GEMM_SEND/WAIT for BLOCK_O is zero net cycle delta), o_gemm=22 confirmed
- [x] Phase 9m attention score chain: `softmax_core_hls_ip.sv` sim model added (fast/slow path via NO_FAST_SOFTMAX); `runtime_decoder_datapath.sv` expanded to 20-state FSM; BLOCK_SCORE/WEIGHTED_SUM routed through DDP_GEMM_SEND/WAIT (real `shared_gemm_engine`); BLOCK_CAUSAL_MASK → DDP_MASK_APPLY (single-cycle `causal_mask_unit`); BLOCK_SOFTMAX → DDP_SOFTMAX_ARM/SCORE/SCALE/ACT (real `softmax_wrapper` handshake); two RTL bugs fixed (cursor advance in new completion states; duplicate BLOCK_CAUSAL_MASK in unique case); score_gemm=704 wsum_gemm=704 confirmed; tb_runtime_decoder_datapath PASS at 425.455 µs ✅ — see Session 29; Phase 9m regression hardening: TB `expected_final_tiles` end-of-sim assertion added; `expected_latest_prob_tile_q` single-latch fix for WEIGHTED_SUM shadow; `build_args.txt` + runtime banner proof mechanisms; attention validity tracking (`attn_score/masked/prob/weighted/o_valid_q`); `attn_prob_scale_q` for WEIGHTED_SUM requantize; dual-mode PASS fast=62/full=63 changed_tiles ✅ — see Session 30
- [ ] Double buffering (ping-pong) for HBM weight streaming (Samarth)
- [ ] Confirm handshake protocol with Om (host launch / FPGA-execute / host-readback)
- [ ] Rijul: RMSNorm unit, RoPE unit, SiLU+multiply unit, softmax unit
- [ ] Scale MAC array from 8 → ~512 lanes (Rijul, parameterised — FSM unchanged)
- [ ] Performance measurement hooks

### Week 4
- [ ] Phase 9: wire compute datapath into tinyllama_u55c_kernel_top (remove sim stubs, connect shared_gemm_engine + nonlinear wrappers + DMA readers/writers)
- [ ] Connect to Om's XRT kernel (host launch, PCIe token upload/readback)
- [ ] End-to-end simulation of full decoder layer with TinyLlama test vectors
- [ ] Verify INT8 path matches `model/tinyllama_gemm_int8.py` golden reference
- [ ] Prefill + decode integration: verify KV-cache correctness across decode steps
- [ ] CPU baseline (C++) for speedup comparison

---

## Key Design Decisions (with rationale)

### 1. Tile Size = T outputs per tile (T = MAC lane count)
- Phase 1: T=8, test dims N=64 → 8 tiles × 64 cycles = 512 MAC cycles
- TinyLlama target: T=512, N=2048 → 4 tiles × 2048 cycles per GEMM
- FSM is fully parameterised (N, K, T) — same RTL at both scales

### 2. Output-Stationary Accumulation
- Each lane keeps its accumulator across the full K loop
- Accumulators cleared at tile boundary (`clear_acc` signal)
- Avoids writing partial sums back to memory

### 3. FSM States: IDLE → LOAD → COMPUTE → WRITE → (LOAD | DONE)
- `LOAD`: assert BRAM enables, wait LOAD_LAT cycles; `mac_valid` LOW
- `COMPUTE`: assert `mac_valid` every cycle, increment `k_idx`
- `WRITE`: assert `wr_en` + `wr_addr` for exactly 1 cycle; `mac_valid` LOW
- On tile boundary: WRITE → LOAD, increment `tile_idx`, pulse `clear_acc` 1 cycle
- `DONE` only after last tile's WRITE completes

### 4. Weights stream from HBM (not stored on-chip)
- One layer's weights at INT8 ≈ 43.5 MB; total on-chip memory ≈ 43.6 MB (zero headroom)
- All 22 layers (957 MB) impossible on-chip
- Weights streamed from HBM (8 GB, ~460 GB/s) per GEMM step
- Activations (~2 KB per vector at INT8) stay on-chip in BRAM

### 5. One decoder-layer engine reused 22× via on-chip controller
- FPGA on-chip prefill/decode controller iterates layer_idx = 0..21
- Controller loads correct HBM weight bank, KV-cache region, and RMSNorm gamma per layer
- Host does not issue a PCIe round trip per GEMM or per layer in production

### 6. On-chip controller owns the full inference loop
- Controller sequence: embedding lookup → prefill (full prompt) → decode loop → final RMSNorm + LM head → greedy argmax → stop on EOS/max_new_tokens
- Inside each layer: controller schedules all 9 GEMMs + dedicated blocks + HBM memory moves
- Host-side per-GEMM sequencing exists only in simulation/testbench infrastructure, not in the deployed runtime

### 7. MAC array target: ~512 lanes for TinyLlama throughput
- 8 lanes: Q projection ~524K cycles → ~1.75 ms at 300 MHz → <1 token/sec
- 512 lanes: Q projection ~8K cycles → ~27 μs at 300 MHz → ~120 tokens/sec
- U55C has 9,024 DSPs; 512-lane INT8 array uses ~256 DSPs (3%)

---

## Signal Interface (agreed with team)

### My FSM → Rijul's MAC Array
| Signal | Width | Description |
|--------|-------|-------------|
| `mac_valid` | 1 | HIGH only during COMPUTE state. LOW during LOAD, WRITE, DONE. |
| `clear_acc` | 1 | Pulsed HIGH for exactly 1 cycle before COMPUTE begins each tile. |
| `x_k` | 8b | Current x[k] value broadcast from Input BRAM to all T lanes. |
| `w_jk[T]` | T×8b | W[j,k] for each lane j, read in parallel from T BRAM banks. |

### My FSM → Output Buffer
| Signal | Width | Description |
|--------|-------|-------------|
| `wr_en` | 1 | Asserted for exactly 1 cycle in WRITE state after k_idx == K-1. |
| `wr_addr` | 4b | = tile_idx. Selects output slot group (tile_idx*T .. tile_idx*T+T-1). |

### My FSM → Om's Host Interface
| Signal | Width | Description |
|--------|-------|-------------|
| `start` | 1 | Input from host — begins GEMM, FSM leaves IDLE. |
| `done` | 1 | Asserted after last tile's WRITE state. Output buffer valid. |

---

## Full Transformer Pipeline (TinyLlama decoder layer)

One layer executes in this order. The MAC array is time-multiplexed across all 9 GEMMs.

```
Input X  (hidden state, INT8)
  → RMSNorm 1 (pre-attention)         [dedicated HW — Rijul]
  → Q Projection  (× Wq)              [MAC array — 32 Q heads]
  → K Projection  (× Wk)              [MAC array — 4 KV heads]
  → V Projection  (× Wv)              [MAC array — 4 KV heads]
  → RoPE on Q and K                   [dedicated HW — Rijul]
  → KV cache write (post-RoPE K, post-proj V → HBM)  [kv_cache_manager — Samarth]
  → Attention Scores (Q_rot × K_rotᵀ / √d_k)  [MAC array]
  → Causal Mask + Softmax             [dedicated HW — Rijul]
  → Weighted Sum (weights × V)        [MAC array]
  → Output Projection (× W_O)        [MAC array]
  → Residual Add (post-attention)     [residual_add.sv reused — Samarth]
  → RMSNorm 2 (pre-FFN)              [dedicated HW — Rijul]
  → gate_proj (× W_gate)             [MAC array — serialized]
  → up_proj   (× W_up)               [MAC array — serialized]
  → SiLU(gate) × up                  [dedicated HW — Rijul]
  → down_proj (× W_down)             [MAC array]
  → Residual Add (post-FFN)          [residual_add.sv reused — Samarth]
  → Layer Output (→ next iteration or PCIe)
```

FPGA on-chip controller drives 22-layer loop, final RMSNorm, LM head, and greedy argmax. Host handles only tokenization, launch, and token readback.

## Dedicated Hardware Blocks

| Block | Owner | Status |
|---|---|---|
| Causal Mask + Softmax unit (LUT exp, running sum, normalize) | Rijul | In Progress |
| SiLU activation + elementwise multiply (SwiGLU) | Rijul | Not Started |
| RMSNorm ×2 (RMS only — no mean subtraction, Q16.16 fixed-point HLS internal) | Rijul | Not Started |
| RoPE unit (sin/cos BRAM, rotate Q and K only) | Rijul | Not Started |
| Residual add (`residual_add.sv`, one reused block, used twice per layer schedule) | Samarth | Done ✅ |

## What Is Out of Scope

- INT4 weights
- Multi-layer simultaneous execution (one decoder-layer engine reused 22×; not 22 instantiated in parallel)
- 2D systolic array
- Full multi-head in Phase 1 validation (single head for test; TinyLlama production target is 32Q/4KV GQA)
- Top-k / top-p sampling (greedy argmax only in first implementation)
- Higher-precision KV cache (INT8 symmetric only; no FP16/FP32 KV cache path in production bitstream)

> Note: KV cache, embedding lookup on FPGA, final RMSNorm on FPGA, LM head on FPGA, and greedy argmax are all **in scope** for the post-Phase-1 target architecture.

---

## File Structure

```
Project/
  CLAUDE.md                           ← this file (auto-loaded by Claude Code)
  README.md                           ← project overview (TinyLlama, team, 4-week plan)
  .gitignore                          ← excludes: CLAUDE.md, AGENTS.md, sim/*.vvp, sim/*.vcd,
                                         model/data/, docs internal files
  docs/
    block_diagram.drawio              ← full TinyLlama decoder layer architecture (Draw.io) ✅
    block_diagram.md                  ← signal tables: full pipeline (Section 2) + GEMM engine (Section 1) ✅
    block_diagram.png                 ← exported PNG (re-export after drawio edits — manual)
    design_decisions.txt              ← all conservative decisions, open items, implemented decisions ✅
    modules.md                        ← full module inventory, interfaces, microarchitecture rules ✅
    implementation_checklist.md       ← file-by-file coding order and verification plan (phases 0-9) ✅
    theory.md                         ← TinyLlama theory: RMSNorm, RoPE, GQA, SwiGLU, INT8, resource feasibility ✅
    meeting_notes.txt                 ← professor feedback (gitignored)
    Milestone2.pdf/.png               ← milestone 2 spec
    Milestone3_Progress_Report.docx   ← milestone 3 report (gitignored)
    golden_trace_plan.md              ← trace verification policy: Phase 0-2 directed; Phase 3 recommended; Phase 4+ required; Phase 7-9 mandatory ✅
    parallelism_tradeoffs.md          ← U55C parallelism rationale: exploited/conservative/revisit tradeoff table ✅
    README.md                         ← docs folder overview ✅
  rtl/
    common/
      tinyllama_pkg.sv                ← all architectural constants, enums (GEMM modes, block IDs, stop reasons, HBM regions) ✅
      tinyllama_bus_pkg.sv            ← packed bus structs: act_bus_t, wt_bus_t, acc_bus_t, scale_bus_t, dma_desc_t, tile_tag_t ✅
      stream_fifo.sv                  ← parameterized ready/valid FIFO ✅
      skid_buffer.sv                  ← 2-entry timing isolation buffer ✅
      descriptor_fifo.sv              ← DMA descriptor FIFO ✅
    control/
      axi_lite_ctrl_slave.sv          ← AXI4-Lite slave front-end (AW+W merge, B/R channels) ✅
      kernel_reg_file.sv              ← 17-word register file, sticky status bits, launch outputs ✅
      prefill_decode_controller.sv    ← IDLE→LAYERS→LMHEAD→TOKEN→DONE runtime FSM ✅
      layer_controller.sv             ← 22-step fixed decoder block sequence; 32-head attention loop (SCORE→MASK→SOFTMAX→WSUM per head); block_start_o/block_done_i handshake; run_done_o after layer 21 ✅
      stop_condition_unit.sv          ← combinational EOS/max-token/abort stop logic ✅
      host_cmd_status_mgr.sv          ← PC30 command/status DMA manager: fetches command beat on launch, writes status beat on terminal events ✅
    memory/
      hbm_port_router.sv              ← priority arbitration (5 rd / 4 wr clients), burst tracking, shell interface ✅
      tile_buffer_bank.sv             ← ping-pong banked SRAM (TILE_BUFFER_BANKS=16, parameterized DATA_W/DEPTH) ✅
      kv_cache_manager.sv             ← KV descriptor address generator: row_index formula, PC22-25 K / PC26-29 V ✅
      scale_metadata_store.sv         ← 4-port combinational read, 1 sync write; 1408-entry register file ✅
      prompt_token_reader.sv          ← multi-beat FSM, token unpacking from DMA beat, is_last sideband ✅
      generated_token_writer.sv       ← ring-buffer write, one token per DMA beat, capacity wrap ✅
      weight_dma_reader.sv            ← multi-beat burst, tensor→block_id/gemm_mode map, rd_desc_ready handshake ✅
      kv_cache_dma_reader.sv          ← multi-beat burst, KDR_ISSUE_DESC handshake, is_last tag ✅
      kv_cache_dma_writer.sv          ← registered data capture, ready not valid-dependent ✅
      embedding_lmhead_dma_reader.sv  ← scale beat accumulation, split EDR_STREAM_RAW/WT/SCALE states ✅
      debug_dma_writer.sv             ← sequential address, debug_enable gate, PC31 ✅
    compute/
      mac_lane.sv                     ← combinational INT8×INT8→INT32 MAC, holds acc when mac_valid=0 ✅
      accumulator_bank.sv             ← 512-element INT32 register file, load>clear priority, tag stored ✅
      requantize_unit.sv              ← INT32→INT8 requant, Q16.16 scale, banker's rounding, nonnegative_only mode ✅
      gemm_operand_router.sv          ← combinational operand mux: act+wt / act+kv / score+kv by gemm_mode; normalizes block_id/gemm_mode tag on both outputs ✅
      gemm_result_router.sv           ← routes acc to requantize_unit / score bypass / lmhead bypass by gemm_mode; normalizes tags on all three output paths ✅
      gemm_op_scheduler.sv            ← dual-mode: RUNMODE_LEGACY (9-GEMM full layer) + RUNMODE_BLOCK (per-block from layer_controller); GQA head loop; clear_acc/emit_acc per K tile ✅
      shared_gemm_engine.sv           ← 512-lane output-stationary GEMM, snapshot register, busy_o ✅
      rope_lut_rom.sv                 ← cos/sin ROM; 2048 positions × 32 dims Q16.16; 8-token broadcast to 512 lanes ✅
      rope_unit.sv                    ← applies RoPE to Q+K; lower/upper half rotate; banker's rounding; token_base assertion ✅
      gqa_router.sv                   ← fans Q to 32 heads, K/V to 4 KV heads; q_head→kv_head = q_head / KV_GROUPS ✅
      causal_mask_unit.sv             ← MASK_NEG_INF for out-of-window positions; zero for inactive rows ✅
      rope_cos_rom.memh               ← 2048 × 32 Q16.16 cos values (generated by export script) ✅
      rope_sin_rom.memh               ← 2048 × 32 Q16.16 sin values (generated by export script) ✅
      elementwise_mul.sv              ← INT8×INT8→INT32 SwiGLU lane product; interleaved capture; BLOCK_GLU_MUL; effective_elem_count masking ✅
      residual_add.sv                 ← INT32+INT32 per lane; block_id_i parameterized for RESIDUAL1/RESIDUAL2 reuse; no saturation ✅
      argmax_reduction.sv             ← streaming argmax; tile-by-tile; tiebreaker = lower token_id; reduce_active_q / have_best_q two-phase ✅
      lm_head_controller.sv           ← 4-state FSM; LMHEAD_TILE_COUNT=250; logit bypass with registered vocab_tile_idx; 3-way IDLE handshake ✅
      embedding_lookup.sv             ← EL_IDLE→EL_REQ→EL_RECV_ROW→EL_OUT; 128-beat DMA assembly; addr = base + token_id × EMBED_ROW_BYTES ✅
      embedding_quantizer.sv          ← EQ_IDLE→EQ_QUANTIZE→EQ_OUT_SCALE→EQ_OUT_ACT; serialized N_TILE=32 lanes/cycle over 64 cycles/row; INT8 tile buffer quant_tile_storage_q[M_TILE][FEATURE_TILE_COUNT]; EQ_OUT_ACT is pure MUX (no arithmetic); 32 parallel quantizers (was 512) ✅
      debug_capture_mux.sv            ← combinatorial priority mux; (layer_id, block_id) match; debug_enable gate; drop_pulse_o on backpressure ✅
    tb/
      tb_stream_fifo.sv               ← stream_fifo testbench (sequential, simultaneous push/pop, backpressure) ✅
      tb_descriptor_fifo.sv           ← descriptor_fifo testbench ✅
      tb_axi_lite_ctrl_slave.sv       ← AXI-Lite + register file smoke test (all 3 write modes, 17 regs, sticky bits, abort) ✅
      tb_host_cmd_status_mgr.sv       ← PC30 command fetch + status write smoke test ✅
      tb_prefill_decode_controller.sv ← control-path smoke test (prefill, 22-layer iter, EOS, decode-only, abort, zero-token error) ✅
      tb_hbm_port_router.sv           ← arbitration priority (prompt>weight, host_status>gen_token), data routing ✅
      tb_tile_buffer_bank.sv          ← ping/pong isolation, 1-cycle read latency ✅
      tb_kv_cache_manager.sv          ← K/V address formula, PC23/PC27, byte_count=192 ✅
      tb_scale_metadata_store.sv      ← write + read (layer=3, head=1) smoke test ✅
      tb_prompt_token_reader.sv       ← 10 tokens across 2 beats, second descriptor addr ✅
      tb_generated_token_writer.sv    ← ring buffer wrap at capacity=2 ✅
      tb_weight_dma_reader.sv         ← compile-only stub ✅
      tb_kv_cache_dma_reader.sv       ← compile-only stub ✅
      tb_kv_cache_dma_writer.sv       ← compile-only stub ✅
      tb_embedding_lmhead_dma_reader.sv ← compile-only stub ✅
      tb_mac_lane.sv                  ← combinational: act×wt+acc, valid=0 hold, negative values ✅
      tb_accumulator_bank.sv          ← load, clear, tag update ✅
      tb_requantize_unit.sv           ← identity/clamp, banker's rounding, nonnegative_only, per-bank scale, golden traces (prefill+decode q_proj) ✅
      tb_gemm_operand_router.sv       ← GEMM_Q, GEMM_SCORE, GEMM_WEIGHTED_SUM, stall logic ✅
      tb_gemm_result_router.sv        ← quant path, scale stall, score bypass, lmhead bypass ✅
      tb_gemm_op_scheduler.sv         ← full layer (seq=16, kv=64), lm_head_only (seq=1, kv=1), real TinyLlama step counts ✅
      tb_shared_gemm_engine.sv        ← directed smoke, backpressure, idle; golden trace: exported q_proj K_TILE=64 slice ✅
      tb_rope_lut_rom.sv              ← ROM output spot-check: cos/sin at 8 token positions, 32 dims, with and without partial token_count ✅
      tb_rope_unit.sv                 ← prefill (token_base=8, 8 tokens) and decode (token_base=15, 1 token) golden traces ✅
      tb_gqa_router.sv                ← Q/K/V fanout, head-mapping, q_head→kv_head modulo, KV_GROUPS=8 ✅
      tb_causal_mask_unit.sv          ← prefill (qb8, 8 active rows) and decode (qb15, 1 active row) golden traces ✅
      tb_rmsnorm_wrapper.sv           ← rmsnorm1 prefill+decode + rmsnorm2 cases; stub validates fp16→Q16 and dequant paths; output tag checks ✅
      tb_softmax_wrapper.sv           ← prefill (qb8) and decode (qb15) softmax cases; prob data, scale, and tag checks ✅
      tb_silu_wrapper.sv              ← prefill and decode silu_gate_m0 cases; dequant and requant path; output tag checks ✅
      tb_elementwise_mul.sv           ← directed 4-lane multiply (2×-6, -3×7, 4×8, -5×-9), lane-beyond-count=0; golden traces glu_mul_m0 prefill+decode ✅
      tb_residual_add.sv              ← directed 4-lane sum ([10+2, -5+8, 3-1, -7+4]); golden traces residual1 prefill + residual2 decode ✅
      tb_argmax_reduction.sv          ← directed (tile0=[10,55,55], tile1=[40,100] → token=VOCAB_TILE+1, logit=100); golden traces; 512-cycle timeout ✅
      tb_lm_head_controller.sv        ← all 250 vocab tiles; vocab_tile_idx per tile; argmax tag (BLOCK_LM_HEAD, GEMM_LM_HEAD, is_last); done_pulse ✅
      tb_embedding_lookup.sv          ← directed (token_id=3, addr=base+3×4096); 128 DMA beats; row_meta checks; golden traces prefill+decode ✅
      tb_embedding_quantizer.sv       ← directed fp16→INT8 smoke; all 64 act tiles data+tag (block_id, tile_id, token_base, seq_count, elem_count, is_partial, is_last); golden traces ✅
      tb_debug_capture_mux.sv         ← combinatorial via #1 delay; priority, layer mismatch, debug_enable gate, drop_pulse behavior ✅
      tb_decoder_layer_smoke.sv       ← Phase 7 integration gate: layer_controller + gemm_op_scheduler + both routers; all 146 blocks verified (block_id, q_head, kv_head, GEMM step counts, router paths); golden traces prefill+decode ✅
      tb_prefill_decode_smoke.sv      ← Phase 8 control-plane gate: prefill_decode_controller + layer_controller + stop_condition_unit; replays golden trace tokens; checks MAX_TOKENS stop, layer passes, token emit count, final mode ✅
      tb_kernel_top_smoke.sv          ← Phase 8/9b end-to-end gate: separate embed/scale/prompt DMA counters; checks scale_read=1, embed_read=prompt_count, prompt_read=prompt_beats; PASS 171.6 µs ✅
      tb_kernel_top_acceptance.sv     ← Phase 9 two-scenario gate: abort-during-RUN_LAYERS then relaunch; checks abort_status_payload, AXI STATUS, gen_count; PASS 214.3 µs ✅
      tb_shell_wrapper_smoke.sv       ← Phase 9 wrapper gate: backpressure on rd_desc/wr_desc/wr_data; all 4 stall_seen flags must fire; PASS 172.1 µs ✅
      tb_runtime_embedding_frontend.sv← Phase 9b unit gate: 2 DMA descriptors (scale then embed); scale bus tags; all 64 act tiles bit-exact vs golden; PASS 3.4 µs ✅
      tb_runtime_decoder_datapath.sv  ← Phase 9h integration gate (rewritten): layer_controller + runtime_decoder_datapath; drives 64 act tiles, runs 22 layers × 146 blocks = 3212 start/done; shadow-tracks full DUT tile state + cursor + signature per block_done; verifies all 64 tiles touched (expected_touched_tiles_q mask), all tile values bit-exact, changed_tiles=64; PASS 234.055 µs ✅
      tb_runtime_lm_head_tail.sv      ← Phase 9f unit gate (rewritten): reads phase6 decode golden traces; overwrites last row to make token_id=7 win; drives real DMA model (4 LMHEAD_GROUPS × 64 beats each); checks sched_start_count==1, rd_desc_count==4, first_tile_winner==7; PASS 6.1 µs ✅
      tb_runtime_final_rmsnorm_tail.sv← Phase 9e unit gate (hardened): prefill+decode cases; scale tag check (11 tag fields, payload separate) + per-tile act tag check (11 fields × 64 tiles); gamma DMA (128 beats) + hidden scale + 64 act tiles; output ±16 INT8 tolerance; integration assertions in kernel_top testbenches probe output_scale_q via hierarchy; PASS 29.8 µs ✅
      README.md                       ← tb directory guide ✅
    nonlinear/
      rmsnorm_wrapper.sv              ← 7-state FSM; interleaved gamma/act capture; fp16_to_q16_16; dequant→HLS→requant; EPSILON_Q16=32'd1 ✅
      softmax_wrapper.sv              ← 5-state FSM; PROB_SCALE_Q16=516 (1/127 Q16.16); dequant score→HLS→quantize_probability ✅
      silu_wrapper.sv                 ← 5-state FSM; partial-tile elem_count gating; lane masking for inactive elements ✅
    top/
      tinyllama_u55c_kernel_top.sv    ← Phase 9f top-level: EMRD 3-way mux (FRONTEND > FINAL > LM_HEAD) on shared embed HBM port; LM_HEAD_BASE_ADDR=0x2000_0000; runtime_lm_head_tail wired with real lmhead_base_addr_i + full DMA interface; tb_kernel_top_smoke constructs winner weights from real DUT hidden_last_row_tile_q to verify token_id=0 wins ✅
      tinyllama_u55c_shell_wrapper.sv ← Phase 9 shell seam: 3 skid_buffer instances (rd_desc, rd_data, coupled wr_req=desc+data packed); write coupling: in_valid=wr_desc&&wr_data, shell_wr_req_ready=desc_ready&&data_ready ✅
      runtime_embedding_frontend.sv   ← Phase 9b: 4-state FSM (UNARMED→SCALE_REQ→SCALE_WAIT→READY); fetches scale metadata on launch; gates token_ready until REF_READY; muxes DMA reader between scale and embed row requests; instantiates embedding_lmhead_dma_reader + embedding_lookup + embedding_quantizer ✅
      runtime_decoder_datapath.sv     ← Phase 9h: 8-state DDP FSM (DDP_IDLE/CAPTURE/READY/BLOCK/APPLY_SEND/APPLY_WAIT/OUT_SCALE/OUT_ACT); scale-first capture of 64 act tiles; 22-layer × 146-block countdown; per-block tile update via DDP_APPLY_SEND/WAIT: one tile per block selected by odd-stride tile_cursor_q (guarantees all 64 tiles touched over 3212 blocks); block-type-specific update arithmetic (8 cases: RMSNORM, GEMM, ROPE/KV, SCORE/WSUM, MASK/SOFTMAX, RESIDUAL/REQUANT, SILU/GLU, default); block_done_o fires in APPLY_WAIT after requantize; emits final scale (BLOCK_FINAL_RMSNORM, layer=21) + all 64 transformed tiles; instantiates layer_controller internally ✅
      runtime_lm_head_tail.sv         ← Phase 9f (rewritten): real TENSOR_LM_HEAD DMA; LMHEAD_GROUPS=4 × N_TILE=32 lanes × 64 beats per group; hidden last-row captured per tile; mac_valid_i=1 (safe: gated by operands_valid=lmhead_group_active&&wt_valid); clear at feature_idx=0, emit at feature 63; group result captured into tile_logits_q; 3-stage logits pipeline; instantiates real shared_gemm_engine + lm_head_controller + argmax_reduction; target_token_i port removed in Phase 9f hardening ✅
      runtime_final_rmsnorm_tail.sv   ← Phase 9e: issues gamma DMA on launch_i (gamma_req_valid_q); scale-first gate (hidden_scale_ready_o=!scale_seen_q, hidden_act_ready_o=scale_seen_q&&rms_act_ready); passes input_scale_q[data[0]] to rmsnorm_wrapper; helper_rst_n=ap_rst_n&&!abort_req_i for abort; instantiates embedding_lmhead_dma_reader + rmsnorm_wrapper ✅
    control_fsm.sv                    ← legacy: SystemVerilog FSM, tiling & dataflow control ✅
    tb_control_fsm.sv                 ← legacy: FSM testbench (7 checks, all passing, VCD disabled) ✅
    top.sv                            ← legacy: top-level integration: FSM + MAC array + BRAM stubs ✅
    tb_top.sv                         ← legacy: integration testbench (4 checks, all passing, VCD disabled) ✅
    mac_unit.sv                       ← Rijul: combinational INT8×INT8→INT32 MAC unit ✅
    mac_array.sv                      ← Rijul: T-lane parallel MAC array with accumulators ✅
    tb_mac_array.sv                   ← Rijul: testbench (7 checks, all passing) ✅
    README.md                         ← module descriptions + sim commands ✅
  hls/
    common/
      fixed_types.hpp                 ← Q16.16 fixed-point aliases, scalar types, vec_t helpers, TinyLlama constants; host fallback stubs when Xilinx headers absent ✅
      stream_utils.hpp                ← clamp, round-to-nearest-even (fixed_t), requantize_int8, quantize_probability, stream read/write helpers ✅
    rmsnorm/
      rmsnorm_core_hls.cpp            ← reads gamma-first then act feature-major; per-row inv_rms; outputs feature-major Q16.16 ✅
      tb_rmsnorm.cpp                  ← HLS C-sim testbench ✅
    softmax/
      softmax_core_hls.cpp            ← stable softmax (subtract row_max); zero-fills inactive rows; row-major I/O ✅
      tb_softmax.cpp                  ← HLS C-sim testbench ✅
    silu/
      silu_core_hls.cpp               ← x * sigmoid(x) per element; handles partial last chunk ✅
      tb_silu.cpp                     ← HLS C-sim testbench ✅
    README.md                         ← HLS directory guide and smoke-test instructions ✅
  model/
    tinyllama.py                      ← Satyarth: pure NumPy TinyLlama golden FP32 reference ✅
    tinyllama_gemm_int8.py            ← Satyarth: INT8 GEMM bridge (analysis + generate modes) ✅
    model.py                          ← early FFN quantization experiments (Satyarth)
    gen_test_vectors.py               ← generates sim/x.txt, w.txt, expected.txt
    export_fpga_vectors.py            ← Phase 3–6 golden trace exporter: layer 0 prefill+decode; Phase 3: 7 weight-projection ops; Phase 4: RoPE ROM tables, causal mask tiles; Phase 5: rmsnorm/softmax/silu nonlinear cases; Phase 6: embedding lookup/quantizer, GLU mul, residual adds, argmax ✅
    README.md                         ← explains tinyllama.py, tinyllama_gemm_int8.py, and export_fpga_vectors.py ✅
    data/
      tinyllama_weights.npz           ← TinyLlama 1.1B weights in float16 (gitignored — 2.1 GB)
  sim/
    x.txt / w.txt / expected.txt      ← test vectors for tb_mac_array (K=64, N=64) ✅
    test_k16/ test_k64/               ← FFN reference outputs from model.py
    int8_layer0/                      ← debug dump from tinyllama_gemm_int8.py layer 0 analysis
    golden_traces/                    ← exported real-model trace artifacts (gitignored); phase3/ holds 28 case files + manifest.json
```

---

## Tiling Loop (pseudocode reference)

```
for tile_idx = 0 to (N/T - 1):
    load W rows [tile_idx*T .. tile_idx*T+T-1] from HBM into on-chip buffer
    clear accumulators
    for k_idx = 0 to (K - 1):
        x_k  = InputBRAM[k_idx]
        w_jk = WeightBRAM[j * K + k_idx]  for j in 0..T-1
        all T lanes: acc[j] += w_jk[j] * x_k   (mac_valid=1)
    write acc[0..T-1] → OutputBuffer[tile_idx*T .. tile_idx*T+T-1]
done
```

---

## Session Log

### Session 1 (2026-03-30)
- Reviewed README, Milestone2, team roles
- Created `docs/block_diagram.drawio`, `docs/block_diagram.md` with FSM, signal tables, tiling
- Applied design review fixes (weight addressing, BRAM banking, WRITE state, `mac_valid` timing)
- Scaffolded `rtl/control_fsm.sv` and `rtl/tb_control_fsm.sv` (7/7 checks passing)
- Created `CLAUDE.md`

### Session 2 (2026-04-01)
- Pulled Rijul's `mac_unit.sv`, `mac_array.sv`, `tb_mac_array.sv`
- Verified interface compatibility with FSM

### Session 3 (2026-04-01)
- Wrote `rtl/top.sv` — FSM + MAC array integration with behavioral BRAM stubs
- Fixed Rijul's `mac_unit.sv` and `mac_array.sv` for Icarus compatibility
- Wrote `rtl/tb_top.sv` — 4/4 integration checks passing
- Created `rtl/README.md`, `docs/README.md`
- Updated `.gitignore`

### Session 4 (2026-04-06)
- **Major pivot:** Expanded scope to full on-chip TinyLlama decoder layer
- Rewrote `docs/block_diagram_full.drawio` (now `block_diagram.drawio`) for full pipeline
- Created `docs/theory.md` and `docs/design_decisions.txt`
- Updated `README.md`, `docs/README.md`, `CLAUDE.md`
- Integrated Satyarth's `model/` scripts; moved to `model/` subdirectory
- Fixed `tb_mac_array.sv` for Icarus (void cast, typedef struct, unpacked array port)
- Satyarth's test vectors now in `sim/` — `tb_mac_array` passes 7/7 checks
- Fixed `tb_control_fsm.sv` VCD path → `sim/`
- Disabled VCD dumps in all 3 testbenches (commented out — uncomment for waveform debug)

### Session 5 (2026-04-07)
- **Architecture corrections:** Updated all docs for TinyLlama specifics:
  - RMSNorm (not LayerNorm) — no mean subtraction
  - SwiGLU FFN (3 GEMMs: gate_proj ∥ up_proj → SiLU × up → down_proj)
  - SiLU activation (not ReLU or GELU)
  - RoPE on Q and K (dedicated block, BRAM sin/cos tables)
  - GQA: 32 Q heads / 4 KV heads
  - Pre-norm residual structure (RMSNorm before each sub-block)
- **Resource feasibility confirmed:** One layer ~43.5 MB INT8; U55C on-chip ~43.6 MB total
  → weights must live in HBM; one layer reused 22× via host loop
- **MAC array sizing:** 8 lanes (Phase 1 test) → target ~512 lanes for TinyLlama (~120 tokens/sec at 300 MHz)
- Rewrote `docs/theory.md` — full TinyLlama architecture, resource table, MAC sizing table
- Updated `docs/block_diagram.drawio` — complete restructure: RMSNorm 1, RoPE, RMSNorm 2, gate_proj, up_proj, SiLU+multiply, down_proj; removed old LayerNorm/ReLU blocks
- Updated `docs/block_diagram.md` — corrected pre-norm pipeline overview, added RoPE, fixed host sequencer pseudocode to 9-GEMM TinyLlama structure
- Updated `docs/design_decisions.txt` — A3 (TinyLlama dims), A7 (SiLU), A9 (RMSNorm), added A13/A14/A15 (HBM, layer reuse, 512-lane target)
- Updated `CLAUDE.md`, `README.md`, `docs/README.md` for TinyLlama accuracy
- Added `model/tinyllama_gemm_int8.py` (Satyarth) — INT8 GEMM bridge, analysis + generate modes
- Updated `model/README.md` to document both tinyllama.py and tinyllama_gemm_int8.py
- Removed `docs/block_diagram.drawio` from `.gitignore` — source file should be tracked
- Added `model/data/` to `.gitignore` — 2.1 GB weights archive must not be committed
- Cleaned `sim/` — removed stale `.vcd`, `.vvp`, and `tiny_random_llama_test.npz` files
- `sim/int8_layer0/summary.txt` confirmed: layer 0 MAE=0.021, same top-1 token as FP32

### Session 6 (2026-04-10)
- **Architecture expansion:** Codex updated docs to full post-Phase-1 TinyLlama inference target
- FPGA now owns: embedding lookup, prefill scheduling, 22-layer loop, KV-cache management, final RMSNorm, LM head, greedy argmax, stop conditions
- Host now owns only: tokenization, XRT launch, prompt token upload, generated token readback
- KV cache added: symmetric INT8 in HBM, per-layer/per-KV-head scales for K and V, cache layout [layer][kv_head][token][head_dim]
- Prefill tiling policy: 64-token sequence tiles, causal mask, writes K/V for every prompt position
- HBM channel allocation finalized: PC00-PC15 layer weights, PC16-17 embeddings/metadata, PC18-21 LM head, PC22-25 K cache, PC26-29 V cache, PC30 control, PC31 debug
- HLS/RTL partition: RTL for controller/GEMM/RoPE/residual/argmax; HLS for RMSNorm/softmax/SiLU
- Double buffering classified as mandatory (not optional optimization)
- `docs/design_decisions.txt` fully rewritten (Sections A, B, C) for post-Phase-1 architecture
- `docs/block_diagram.drawio` fully rewritten for post-Phase-1 system diagram
- `docs/block_diagram.md` Section 2 fully rewritten for post-Phase-1 architecture
- `docs/modules.md` created — full module inventory, interfaces, microarchitecture rules (Codex)
- `docs/implementation_checklist.md` created — file-by-file coding order, phases 0-9 (Codex)
- `docs/design_decisions.txt` A7 and B2 updated: post-RoPE K cached, post-projection V cached; decode reads cached K directly without reapplying RoPE
- `docs/array_architecture.txt` deleted (superseded by modules.md)
- `AGENTS.md` added to `.gitignore` (Codex context file, parallel to CLAUDE.md)
- `README.md` and `docs/README.md` updated to include modules.md and implementation_checklist.md
- `CLAUDE.md` updated: post-Phase-1 scope, corrected residual adder to one reused block, removed gate/up parallel notation, fixed Om's role, precision row updated to Q16.16, session log reordered

### Session 7 (2026-04-11)
- **Phase 0 implemented and verified** (Codex)
- `rtl/common/tinyllama_pkg.sv` — all constants, GEMM mode enum, block_id_e (26 entries following §8.9 schedule), tensor_id_e, hbm_region_e, stop_reason_e, error_code_e
- `rtl/common/tinyllama_bus_pkg.sv` — tile_tag_t, token_bus_t, act_bus_t, wt_bus_t, acc_bus_t, scale_bus_t, dbg_bus_t, dma_desc_t, token_write_desc_t
- `rtl/common/stream_fifo.sv` + `skid_buffer.sv` + `descriptor_fifo.sv` — Phase 1.1-1.3 utilities also implemented
- `rtl/tb/tb_stream_fifo.sv` + `tb_descriptor_fifo.sv` — all passing; stream_fifo TB covers sequential, simultaneous push/pop, backpressure, full-FIFO push-while-full
- `hls/common/fixed_types.hpp` + `stream_utils.hpp` — Q16.16 fixed-point types, no float arithmetic, host-side fallback stubs for non-Vivado builds
- Phase 0 exit criteria met: packages compile and import cleanly, HLS headers g++ smoke-test clean, no new TinyLlama RTL in legacy flat `rtl/` root
- Issues found and fixed: added `BLOCK_KV_CACHE_WRITE` and `BLOCK_REQUANTIZE` to `block_id_e`; removed float from `stream_utils.hpp`; scale_bus_t made unsigned; stream helpers made reference-consistent; testbench coverage expanded

### Session 8 (2026-04-12)
- **Phase 1 control skeleton verified** — all 5 modules and 2 testbenches reviewed and confirmed correct
- `rtl/control/axi_lite_ctrl_slave.sv` — AW+W channel merge correct; word address strip correct; B/R paths correct
- `rtl/control/kernel_reg_file.sv` — all 17 registers correct; sticky DONE/ERROR/STOP_VALID bits correct; byte-strobe apply_wstrb32 correct; START clears all sticky bits
- `rtl/control/prefill_decode_controller.sv` — 6-state FSM; launch semantics match B9 exactly; abort/error at every state; mode transition prefill→decode after first token; one minor dead-code guard (zero-prompt-count check unreachable; harmless)
- `rtl/control/layer_controller.sv` — 0..21 iteration correct; abort shortcircuit correct; weight/KV selectors match layer_id
- `rtl/control/stop_condition_unit.sv` — pure combinational; priority abort>EOS>max_tokens; max_token uses >= and guards on max_new_tokens!=0
- `rtl/tb/tb_axi_lite_ctrl_slave.sv` — PASS; covers all 3 AXI write orderings, all 17 registers, sticky status, launch mode, START pulse one-cycle, abort, version
- `rtl/tb/tb_prefill_decode_controller.sv` — PASS; covers prefill launch, 22-layer iteration with layer_id check, LM head, EOS stop, decode-only, host abort
- `tinyllama_pkg.sv` AXI-Lite constants verified against design_decisions.txt B9 — exact match
- **Gap flagged:** `rtl/control/host_cmd_status_mgr.sv` (Phase 1.6) not implemented; needed for HBM command/status DMA path before Om's XRT host side can wire up

### Session 9 (2026-04-12)
- **Phase 1.6 implemented and verified** (Codex) — `host_cmd_status_mgr.sv` stub and its testbench
- `rtl/control/host_cmd_status_mgr.sv` — reads B10 command block from PC30 on launch (one 256-bit beat, 8 words); writes B10 status block on terminal events; `HOST_IO_PC_ID=30`, `HOST_BLOCK_WORDS=8`, all word indices frozen
- `rtl/tb/tb_host_cmd_status_mgr.sv` — PASS; covers: cmd descriptor fields, cmd data parsing, command_info_valid outputs, status descriptor/data fields, handshake deassert
- `rtl/control/prefill_decode_controller.sv` — fixed: zero-prompt-count guard moved into CTRL_IDLE → now reachable; fires ERROR_BAD_DESCRIPTOR on prefill launch with prompt_token_count=0
- `rtl/tb/tb_prefill_decode_controller.sv` — extended with zero-token prefill error case; PASS
- `rtl/common/tinyllama_pkg.sv` — added B10 host block constants: HOST_IO_PC_ID, HOST_BLOCK_WORDS, HOST_BLOCK_BYTES, HOST_CMD_WORD_* (0–4), HOST_STATUS_WORD_* (0,1,2,3,4,7)
- `docs/design_decisions.txt` B10 added — PC30 command/status block layout frozen (one beat each); layout matches package constants exactly
- `docs/modules.md` M03 updated — host_cmd_status_mgr documented; stage-to-module table reflects it under "Host launch and status" and "Generated token writeback"
- `rtl/README.md` + `rtl/tb/README.md` — host_cmd_status_mgr and its testbench added to Phase 1 Control Skeleton table

### Session 10 (2026-04-13)
- **Phase 2 HBM memory subsystem implemented and verified** (Codex + review)
- All 11 `rtl/memory/` modules implemented and verified correct
- **Critical bug found and fixed:** all 6 Phase 2 testbenches had `ap_clk`/`ap_rst_n` in port connections while local signals were `clk`/`rst_n` — DUTs received undriven clock, all checks passed trivially via X-eval; fixed to `.ap_clk(clk), .ap_rst_n(rst_n)`
- **`kv_cache_dma_writer.sv` fixed:** `pending_data_q` was driven by `always_comb` (live pass-through); corrected to capture in `always_ff`; ready-depends-on-valid anti-pattern removed (`req_ready_o = !pending_q`)
- **`weight_dma_reader.sv` expanded:** single-beat stub → multi-beat FSM with `WDR_ISSUE_DESC` state, beat counter, `tensor_supported()` error path, `block_id_from_tensor()` / `gemm_mode_from_tensor()` lookup functions
- **`kv_cache_dma_reader.sv` expanded:** same multi-beat + `KDR_ISSUE_DESC` upgrade
- **`embedding_lmhead_dma_reader.sv` expanded:** multi-beat, scale beat accumulation into `scale_beats_q[]`, split stream states (`EDR_STREAM_RAW` / `EDR_STREAM_WT` / `EDR_STREAM_SCALE`)
- Deferred to Phase 3+: variable bit-select synthesis warning in `prompt_token_reader.sv` and `kv_cache_manager.sv`; fixed-priority starvation risk in `hbm_port_router.sv`

### Session 11 (2026-04-13)
- **Phase 3 compute engine verified** — all 7 RTL modules (`mac_lane`, `accumulator_bank`, `requantize_unit`, `gemm_operand_router`, `gemm_result_router`, `gemm_op_scheduler`, `shared_gemm_engine`) and all 7 testbenches confirmed correct; real TinyLlama step counts verified for all 9 GEMMs
- **Golden trace plan documented** — `docs/golden_trace_plan.md` created; policy: Phase 0-2 directed only, Phase 3 recommended, Phase 4+ required, Phase 7-9 mandatory integration gates; wired into `design_decisions.txt` (B12), `modules.md` (Section 11), `implementation_checklist.md`, `docs/README.md`, `README.md`
- **`model/export_fpga_vectors.py`** — canonical Phase 3 trace exporter: layer 0 prefill (16 tokens) + decode (1 token), all 7 weight-projection ops; writes `.npz` cases + packed `.memh` RTL fixtures under `sim/golden_traces/phase3/`; lane packing `lane = m_local * N_TILE + n_local` frozen in `modules.md`
- **`tb_requantize_unit.sv` updated** — loads exported prefill and decode `q_proj` requant fixtures, verifies bit-exact output against Python reference; adds directed per-bank scale case (bank 0 = 1.0, bank 1 = 0.5, 64 active lanes) to exercise `SCALE_IDX = lane / 32` path with banker's rounding at the bank boundary
- **`tb_shared_gemm_engine.sv` updated** — replays exported real-model `q_proj` K_TILE=64 smoke slice through the engine and checks accumulated result bit-exact against exported expected vector
- `sim/golden_traces/` added to `.gitignore`; trace coverage is partial by design (q_proj only in Phase 3 — other ops covered in Phase 7+)

### Session 12 (2026-04-13)
- **Phase 4 attention-path blocks verified** — all 4 RTL modules and testbenches confirmed correct
- `rtl/compute/rope_lut_rom.sv` — cos/sin ROM; 2048 positions × 32 dims Q16.16; restructured to two generate loops (8 token lookups first, then 512-lane broadcast) to avoid 512 independent ROM address ports at synthesis; verified correct
- `rtl/compute/rope_unit.sv` — applies RoPE to Q+K; lower/upper half rotate with banker's rounding; `ifndef SYNTHESIS assertion added for matching Q/K token_base tags
- `rtl/compute/gqa_router.sv` — Q fans to all 32 heads, K/V to 4 KV heads; q_head→kv_head = q_head / KV_GROUPS; verified GQA repetition logic
- `rtl/compute/causal_mask_unit.sv` — MASK_NEG_INF for positions outside causal window; zero for inactive rows; testbench covers prefill (qb8, 8 active rows) and decode (qb15, 1 active row)
- `rtl/compute/rope_cos_rom.memh` + `rope_sin_rom.memh` — generated by export script, 2048×32 Q16.16 values
- `docs/parallelism_tradeoffs.md` — created; U55C parallelism rationale: what's exploited (lane-level, pipeline, GQA sharing, double-buffer), conservative choices (fixed-priority arbiter, single HLS instance), and what to revisit post-demo
- **Phase 4 hardening fixes verified:** rope_lut_rom synthesis restructure (two generate loops) and rope_unit token_base assertion both confirmed correct after Codex applied them
- `model/export_fpga_vectors.py` expanded — Phase 4 export functions: `build_rope_rom_tables`, `build_rope_case`, `build_causal_mask_case`, `export_phase4_for_tokens`

### Session 13 (2026-04-13)
- **Phase 5 nonlinear wrappers verified** — all 3 RTL wrappers, 3 testbenches, and 3 HLS kernels confirmed correct
- `rtl/nonlinear/rmsnorm_wrapper.sv` — 7-state FSM; fp16_to_q16_16 (handles denormals/normals/inf with banker's rounding); interleaved gamma/act capture; EPSILON_Q16=32'd1 (nearest Q16.16 to TinyLlama's 1e-5); feature-major act→core ordering matches HLS
- `rtl/nonlinear/softmax_wrapper.sv` — PROB_SCALE_Q16=516 (1/127 in Q16.16); dequantize INT32 score → Q16.16 → HLS → quantize_probability (negatives clamped to 0); always sends full SCORE_K_TILE columns (causal mask pre-applied)
- `rtl/nonlinear/silu_wrapper.sv` — partial-tile elem_count gating; lane masking zeroes inactive elements on receive; scale broadcast as {SCALE_VECTOR_ELEMS{output_scale_q}}
- `hls/rmsnorm/rmsnorm_core_hls.cpp` — gamma-first stream, act feature-major, per-row inv_rms, Q16.16 throughout
- `hls/softmax/softmax_core_hls.cpp` — stable softmax with row_max subtraction; zero-fills rows >= row_count
- `hls/silu/silu_core_hls.cpp` — x * sigmoid(x) per element; partial last chunk handled
- Testbenches use inline HLS stub pattern: stubs validate wrapper data marshaling (fp16 conversion, dequant, requant) against real TinyLlama golden traces; serve precomputed output chunks
- **Phase 5 hardening (Codex + review):** `tb_rmsnorm_wrapper.sv` hardened — output tag checks added for all 64 tiles (block_id, gemm_mode, tile_id, elem_count, is_partial, is_last); RMSNORM2 test case added; Icarus flat-array workaround for captured tag fields; scale tag check added
- **Epsilon correction:** Original `EPSILON_Q16=32'd655` recommendation was wrong (655/65536 ≈ 1e-2, not 1e-5); correct nearest Q16.16 representation of 1e-5 is 32'd1 (1/65536 ≈ 1.53e-5); restored to 32'd1
- `model/export_fpga_vectors.py` expanded — Phase 5 export functions: `build_rmsnorm_case`, `build_softmax_case`, `collect_phase5_nonlinear_tensors`, `export_phase5_cases`

### Session 14 (2026-04-13)
- **Phase 6 embedding, output, and datapath completion verified** — all 7 RTL modules and 7 testbenches confirmed correct
- `rtl/compute/embedding_lookup.sv` — 4-state FSM (EL_IDLE→EL_REQ→EL_RECV_ROW→EL_OUT); D_MODEL=2048 FP16 → EMBED_ROW_BYTES=4096 → 128 DMA beats; address = base_addr + token_id × EMBED_ROW_BYTES; accumulates beats via variable bit-select; passes row_meta tag through
- `rtl/compute/embedding_quantizer.sv` — 4-state FSM (EQ_IDLE→EQ_COLLECT→EQ_OUT_SCALE→EQ_OUT_ACT); scale captured first (scale_ready_o = !scale_captured_q && EQ_IDLE); batches M_TILE=16 rows into 64 act tiles; fp16_to_q16_16 with round-half-up bias; quantize_fixed_lane without <<16 shift (key fix); scale_q.data[ROW_LOCAL] = one scale per 32-lane (N_TILE) row slice; batch_token_base_q increment uses pre-clear value via non-blocking assignment
- `rtl/compute/elementwise_mul.sv` — interleaved capture of silu_i and up_i (act_bus_t); INT8×INT8→INT32 per lane; tags_compatible excludes block_id/gemm_mode (silu=BLOCK_SILU, up=BLOCK_UP differ); effective_elem_count masking on output; BLOCK_GLU_MUL
- `rtl/compute/residual_add.sv` — interleaved capture; INT32+INT32 per lane, no saturation (matches INT32 contract; overflow not expected in operating range); block_id_i parameterized for RESIDUAL1/RESIDUAL2 reuse; effective_elem_count picks first non-zero
- `rtl/compute/argmax_reduction.sv` — reduce_active_q set by start_i, cleared when is_last tile consumed; token_valid_o = !reduce_active_q && have_best_q; tiebreaker = lower token_id; out_token_id_q captured at is_last; done_pulse on output handshake; token_base = tile_id × VOCAB_TILE
- `rtl/compute/lm_head_controller.sv` — 4-state FSM (LMH_IDLE→LMH_ISSUE→LMH_WAIT_SCHED→LMH_WAIT_LOGIT); LMHEAD_TILE_COUNT=250, LMHEAD_LAST_ELEMS=128; 3-way IDLE handshake (start + hidden_valid + hidden_scale_valid); sched_start_o combinatorial from (state == LMH_ISSUE); logit bypass with registered vocab_tile_idx for correct tag attachment
- `rtl/compute/debug_capture_mux.sv` — purely combinatorial priority mux; matches (layer_id, block_id) against debug_layer_sel_i and debug_step_sel_i; drop_pulse_o fires one cycle when match exists but !dbg_ready_i; debug_enable_i gate
- **Testbench quality:** tb_embedding_quantizer.sv is the most comprehensive — all 64 act tiles data + all tag fields (block_id, gemm_mode, tile_id, token_base, seq_count, elem_count, is_partial, is_last); scale tag fully checked; directed and golden trace cases; tb_lm_head_controller covers all 250 vocab tiles with per-tile tag checks and done_pulse
- **quantize_fixed_lane fix confirmed across all RTL:** `numerator_abs / denominator` without prior <<16 shift — correctly implemented in embedding_quantizer, rmsnorm_wrapper, silu_wrapper (fix applied in Phase 5 session)
- **Two fp16_to_q16_16 rounding variants acknowledged:** rmsnorm_wrapper uses banker's rounding; embedding_quantizer uses round-half-up via bias — agree for FP16 values exactly representable in Q16.16; can differ by 1 LSB otherwise; acceptable for student project
- `model/export_fpga_vectors.py` expanded — Phase 6 export functions: `build_embedding_lookup_case`, `build_embedding_quantizer_case`, `build_glu_mul_case`, `build_residual_case`, `build_argmax_case`, `export_phase6_cases`

### Session 15 (2026-04-13)
- **Phase 7 decoder-layer integration gate verified** — all 5 testbenches passing; simulation confirmed live
- `rtl/control/layer_controller.sv` — rewritten with concrete 22-step decoder block sequence; `layer_step_e` enum encodes all steps; 32-head GQA inner loop (SCORE→CAUSAL_MASK→SOFTMAX→WEIGHTED_SUM per head); q_head_id/kv_head_id driven only during attention steps; REQUANT3 triggers layer increment or run_done_o
- `rtl/compute/gemm_op_scheduler.sv` — dual-mode: RUNMODE_LEGACY (original full-layer 9-GEMM sequence, Phase 3 backward compatible) and RUNMODE_BLOCK (one GEMM block at a time, driven by layer_controller's block_start_o/block_done_i handshake); non-GEMM block_start_i signals silently bypassed via block_is_gemm() guard
- `rtl/compute/gemm_operand_router.sv` — tag normalization added: both act_o and wt_o have block_id/gemm_mode overwritten from gemm_mode_i regardless of upstream source tags
- `rtl/compute/gemm_result_router.sv` — matching tag normalization on all three output paths (quant, score bypass, lmhead bypass)
- `rtl/tb/tb_decoder_layer_smoke.sv` — Phase 7 mandatory integration gate; instantiates layer_controller + gemm_op_scheduler + gemm_operand_router + gemm_result_router; verifies all 146 blocks per decoder layer (block_id, q_head_id, kv_head_id, GEMM step counts, router data/valid paths); golden traces prefill (seq=16) + decode (seq=1, kv=16); layer-1 transition + abort verified
- `model/export_fpga_vectors.py` expanded — `build_phase7_block_schedule`, `build_phase7_layer_case`, `export_phase7_cases`; generates 12 .memh fixtures per mode under `sim/golden_traces/phase7/rtl/`
- **Step counts verified:** Q/O=2048, K/V=256, SCORE/head=1, WSUM/head=2, GATE/UP=5632, DOWN=5632, LMH=128 (seq=16, kv=64); all match RTL tile count formulas against TinyLlama constants
- **Simulation run:** `vvp sim/tb_decoder_layer_smoke.vvp` → PASS at 436 ms sim time; benign MAX_BLOCKS=160 vs 146-entry trace warnings (loop bounded by meta_mem[2])

### Session 16 (2026-04-13)
- **Phase 8 runtime-core integration verified** — all 4 testbenches passing; simulation confirmed live
- `rtl/top/tinyllama_u55c_kernel_top.sv` — new top-level integrating 9 submodules (axi_lite_ctrl_slave, kernel_reg_file, host_cmd_status_mgr, prefill_decode_controller, stop_condition_unit, prompt_token_reader, generated_token_writer, layer_controller, hbm_port_router); weight/kv/embed/debug DMA ports stub-tied; sim stubs: `sim_block_done_q <= block_start` (1-cycle), 3-step LM head pipeline (lm_pending→lm_done+token_pending→token_valid); `sim_token_id_q = 32'd1000 + ctrl_generated_token_count`
- `rtl/control/prefill_decode_controller.sv` — extended to 8-state FSM: added CTRL_WAIT_CMD (waits command_info_valid_i, fires token_writer_start_o), CTRL_WAIT_PROMPT (waits prompt_read_done_i, fires embedding_start_o + layer_start_o); decode path: CTRL_WAIT_CMD → CTRL_RUN_LAYERS skipping prompt read; CTRL_DONE transitions to CTRL_IDLE in one cycle; abort_req_i now handled directly in CTRL_RUN_LAYERS (before layer_pass_done_i) so lm_head_start_o is never fired after an in-flight abort
- `rtl/control/stop_condition_unit.sv` — confirmed: uses `next_token_count = generated_count + 1`; checks count+1 during the token_valid event so MAX_TOKENS fires on the Nth token, not Nth+1
- `rtl/tb/tb_prefill_decode_smoke.sv` — integration gate for prefill_decode_controller + layer_controller + stop_condition_unit; replays golden trace generated tokens; checks no error, MAX_TOKENS stop, generated_token_count, layer_pass_count, token_emit_count, final active_mode=DECODE
- `rtl/tb/tb_kernel_top_smoke.sv` — end-to-end control-plane test through tinyllama_u55c_kernel_top; simulates shell DMA (rd_pending pipeline, cmd vs prompt address dispatch); verifies generated token values (1000, 1001), AXI-Lite readbacks (STATUS bits, GENERATED_TOKEN_COUNT=2, LAST_TOKEN_ID=1001), interrupt, command_read_count=1, prompt_read_count=2, launch+final status DMA writes, final_status_payload fields
- `rtl/tb/tb_prefill_decode_controller.sv` — abort-from-CTRL_RUN_LAYERS case added; `wait_for_abort_done_from_run_layers()` task checks lm_head_start never fires between abort assertion and done_pulse
- `sim/golden_traces/phase8/rtl/` — 4 fixture files: meta (12 words: prompt=16, max_new=2, eos=0xffffffff, generated_capacity=8, layer_passes=2, prompt_reads=2, gen_tokens=2, final_status_w0=0x2a=42), command_words (8 words), prompt_tokens (16 IDs), generated_tokens_expected (2 real TinyLlama token IDs: 0x750e, 0x7525)
- **Simulation runs:** tb_host_cmd_status_mgr PASS (220 ns), tb_prefill_decode_controller PASS (96.7 µs), tb_prefill_decode_smoke PASS (128.7 µs), tb_kernel_top_smoke PASS (129.3 µs)
- **hw_error_valid note:** mixes ctrl_error_pulse (1-cycle) with prompt_error_valid (level) — when prompt-side errors go live the sustained level feeds the entire error response chain: kernel_reg_file sticky ERROR bit (re-latches each cycle), host_cmd_status_mgr status DMA (may retrigger), prefill_decode_controller CTRL_ERROR (FSM sees condition over multiple cycles), and the interrupt line (sustained not pulsed); low-impact today because prompt_token_reader ties error_valid_o = 1'b0; fix = pulse-qualify prompt_error_valid before mixing into hw_error_valid; harden before prompt-side error paths go live
- **Phase 8 scope:** control-plane integration only; Phase 3-6 compute modules are real and verified but not yet wired into the top-level; normalized shell DMA boundary (not raw m_axi_pc00..31); full datapath wiring is Phase 9

### Session 17 (2026-04-16)
- **Phase 9 shell-seam closure verified** — all 3 new testbenches passing
- `rtl/top/tinyllama_u55c_shell_wrapper.sv` — 3 skid_buffer instances: `u_shell_rd_desc_skid` (DATA_W=DMA_DESC_W, core→shell read desc), `u_shell_rd_data_skid` (DATA_W=DMA_BEAT_W, shell→core read data), `u_shell_wr_req_skid` (DATA_W=DMA_DESC_W+DMA_BEAT_W, coupled write); write coupling: `in_valid = core_wr_desc_valid && core_wr_data_valid`; `shell_wr_req_ready = shell_wr_desc_ready_i && shell_wr_data_ready_i`; `shell_wr_desc_valid_o = shell_wr_data_valid_o = shell_wr_req_valid`
- `rtl/tb/tb_kernel_top_acceptance.sv` — two-scenario gate: (1) abort-during-RUN_LAYERS using hierarchical `dut.layer_busy`, checks abort_status_word=0x3a, AXI STATUS=HOST_ABORT, gen_count=0; (2) relaunch, checks STATUS=BUSY only (sticky cleared), then MAX_TOKENS, gen_count=2, last_token=1001; `sim/golden_traces/phase9/rtl/` has 13-word meta (adds meta[12]=0x3a); PASS (214.3 µs)
- `rtl/tb/tb_shell_wrapper_smoke.sv` — DUT=tinyllama_u55c_shell_wrapper; backpressure patterns: `shell_rd_desc_ready=cycle[0]`, `shell_wr_desc_ready=cycle[0]`, `shell_wr_data_ready=cycle[1]`; all 4 stall_seen flags (rd_desc, wr_desc, wr_data, wr_req) must fire; timeout 60000 cycles; PASS (172.1 µs)
- **Phase 9b real inference closure (first slice) verified** — all 4 testbenches passing
- `rtl/top/runtime_embedding_frontend.sv` — 4-state FSM (REF_UNARMED→REF_SCALE_REQ→REF_SCALE_WAIT→REF_READY); `launch_i` triggers scale metadata fetch; `token_ready_o = (state_q==REF_READY) && lookup_token_ready`; muxes DMA reader between scale (TENSOR_SCALE_META at scale_meta_base_addr_i) and embed rows (TENSOR_EMBED at lookup_req_base_addr); `done_pulse_o = quant_done_pulse`; instantiates embedding_lmhead_dma_reader + embedding_lookup + embedding_quantizer
- `rtl/top/tinyllama_u55c_kernel_top.sv` updated: adds runtime_embedding_frontend; `launch_i=start_pulse` (scale fetch starts immediately, hiding latency behind command fetch); `embed_lm_rd_desc` channel live (was 1'b0); `prompt_read_done_i = embed_done_pulse`; `token_ready_i = prompt_token_ready`; `hw_current_block` checks `embed_busy || prompt_busy` for BLOCK_EMBED; `scale_ready_i`/`act_ready_i` tied 1'b1 (consumed by future datapath)
- `rtl/tb/tb_runtime_embedding_frontend.sv` — Phase 9b unit gate; loads Phase 6 golden traces; DMA model tracks rd_kind (RD_SCALE vs RD_EMBED), beat index, total beats; checks 2 descriptors (first=SCALE_META at SCALE_META_BASE_ADDR, second=TENSOR_EMBED at embedding_base_addr + token_id × EMBED_ROW_BYTES); scale bus tag fields; all 64 act tiles bit-exact vs golden; PASS (3.4 µs)
- `rtl/tb/tb_kernel_top_smoke.sv` updated: adds `embed_read_count`/`scale_read_count` counters; dispatches REGION_EMBED_META + TENSOR_SCALE_META → scale, REGION_EMBED_META + TENSOR_EMBED → embed; serves scale beats as Q16.16=1.0, embed beats as zeros; checks scale=1, embed=prompt_count, prompt_read=prompt_beats; PASS (171.6 µs, +42 µs from Phase 8 baseline = 2050 embedding DMA beats)
- `rtl/tb/tb_kernel_top_acceptance.sv` updated with same REGION_EMBED_META dispatch logic; PASS (214.3 µs)
- `rtl/tb/tb_shell_wrapper_smoke.sv` updated with REGION_EMBED_META dispatch; PASS (172.1 µs)
- **DMA descriptor region tagging confirmed:** `embedding_lmhead_dma_reader.sv:164` sets `region = REGION_EMBED_META` for non-LM-head requests; `prompt_token_reader.sv` uses `REGION_HOST_IO`; testbench dispatch is correct
- **embedding_start early-launch design:** `prefill_decode_controller.embedding_start_o` is captured in kernel_top but not forwarded to the frontend — embedding frontend receives `start_pulse` directly so scale metadata fetch begins in the same cycle as overall launch, hiding its latency behind the B10 command-fetch; intentional optimization, not a dead wire; signal reserved for future gating if the pipeline needs to delay embedding ingress
- **Sim time reference (Ph9b):** tb_kernel_top_smoke 171.6 µs, tb_kernel_top_acceptance 214.3 µs, tb_shell_wrapper_smoke 172.1 µs, tb_runtime_embedding_frontend 3.4 µs

### Session 18 (2026-04-21)
- **Phase 9b synthesis hardening: embedding_quantizer.sv rework** — Vivado OOM resolved; all 5 affected testbenches re-verified PASS
- **Root cause:** 512 parallel 64-bit integer dividers in `quantize_fixed_lane` generate block exceed FPGA LUT budget; OOM occurs post-synthesis (opt_design/place_design phase, not during synth_design itself)
- **Iteration 1 (per-row FP16 storage):** replaced batch-global flattened FP16 store with per-row `row_storage_q[0:M_TILE-1]`; ROW_LOCAL is compile-time constant eliminating dynamic M_TILE-way mux; still 512 parallel dividers; all 5 testbenches PASS
- **Iteration 2 (serialized N_TILE=32 quantization, final):** EQ_COLLECT → EQ_QUANTIZE state; processes one captured FP16 row at 32 lanes/cycle (N_TILE=32) over FEATURE_TILE_COUNT=64 cycles; stores INT8 results in `quant_tile_storage_q[0:M_TILE-1][0:FEATURE_TILE_COUNT-1]` (262,144 bits); EQ_OUT_ACT assembles act_bus_t from INT8 buffer using MUX only (no arithmetic on output path); generate block reduced from 512 → 32 parallel quantizers
- **row_ready_o change:** removed EQ_COLLECT — now only asserts in EQ_IDLE; `row_ready_o = scale_captured_q && (state_q == EQ_IDLE) && (batch_row_count_q < M_TILE)`
- `quant_tile_storage_q` exact resource class (BRAM/LUTRAM/FFs) depends on Vivado inference — well-shaped memory, much cheaper than predecessor; next Vivado run will confirm
- Variable bit-select on `current_row_fp16_q` in generate block: synthesizable; Vivado may warn but is not a correctness issue
- Throughput: 65 cycles/row (64 EQ_QUANTIZE + 1 EQ_IDLE) still faster than DMA rate (128+ cycles/row per embed row) — no bottleneck introduced
- **Sim times after rework:** tb_embedding_quantizer 14.5 µs (timeout widened for 1024-cycle quantization phase), tb_runtime_embedding_frontend 4.1 µs, tb_kernel_top_smoke 172.2 µs, tb_kernel_top_acceptance 215.6 µs, tb_shell_wrapper_smoke 172.8 µs
- **Vivado synthesis confirmed** (`synth_design completed successfully`, 0 errors, 0 critical warnings): LUTs 1,003 (0.08%), FFs 1,507 (0.06%), BRAM 0, DSPs 0; `quant_tile_storage_q` inferred as FFs (BRAM=0 confirmed); embedding_quantizer = 170 cells; runtime_embedding_frontend = 528 cells; peak memory 17,259 MB on Windows machine — OOM fully resolved; I/O overutilization (147%) expected for standalone kernel_top synthesis, resolves in shell integration

### Session 19 (2026-04-21)
- **Phase 9c decoder datapath scaffold verified** — all 7 testbenches confirmed PASS
- `rtl/top/runtime_decoder_datapath.sv` — 6-state DDP FSM (DDP_IDLE→DDP_CAPTURE→DDP_READY→DDP_BLOCK→DDP_OUT_SCALE→DDP_OUT_ACT); scale-first capture ordering enforced (`embed_scale_ready = !scale_seen_q`, `embed_act_ready = !context_ready_q`); all 64 act tiles tracked via `tile_seen_q[FEATURE_TILE_COUNT-1:0]` bitmask; block countdown driven by `block_latency(block_id, q_head_id, context_signature_q)` — signature makes latencies input-dependent; last layer last block (layer==21 && block_count==145) transitions to DDP_OUT_SCALE; final hidden = passthrough of ingested act tiles (Phase 9c scaffold — not real RMSNorm); `BLOCKS_PER_LAYER = 6 + (4 × N_Q_HEADS) + 12 = 146`; 22 × 146 = 3212 confirmed
- `rtl/top/tinyllama_u55c_kernel_top.sv` updated (Phase 9c): `frontend_launch = start_pulse || embedding_start`; `frontend_token_valid = decode_token_pending_q || prompt_token_valid`; `decode_token_pending_q` register feeds generated tokens back as next-step embedding input; `decoder_final_scale_ready/act_ready` still stub-tied `1'b1` (fixed in Phase 9d); `decoder_context_valid` declared but not consumed (dead wire, cosmetic)
- **Phase 9c follow-up items confirmed resolved in Phase 9d:** `decoder_final_scale_ready = 1'b1` and `decoder_final_act_ready = 1'b1` stubs replaced with real handshakes from `runtime_lm_head_tail.hidden_scale_ready_o` / `hidden_act_ready_o`
- **Phase 9d LM head tail scaffold verified** — all 10 testbenches confirmed PASS (8 targeted + 2 controller regression)
- `rtl/top/runtime_lm_head_tail.sv` — hidden_scale_ready_o = !scale_seen_q (accepts exactly one scale per launch); hidden_act_ready_o = !context_ready_q (accepts all tiles until hidden_done_pulse fires, safe because decoder emits all 64 tiles before firing done_pulse); only first act tile stored as hidden_seed_q (scaffold); context_signature_q XOR hash; core_start_w = start_pending_q && context_ready_q && lmh_hidden_ready && lmh_hidden_scale_ready && !lmh_logits_valid_q (combinatorial 1-cycle pulse — start_pending_q cleared on next posedge, lmh_hidden_ready goes low as controller leaves LMH_IDLE); core_start_w drives start_i = hidden_valid_i = hidden_scale_valid_i simultaneously to lm_head_controller (correct 3-way handshake); 2-cycle pipeline: sched_start → lmh_logits_bus_q captured + lmh_sched_done_q set; next cycle: lmh_sched_done_q → lmh_logits_valid_q set; deterministic logit shim: target_token_q gets TARGET_LOGIT_BASE=500000, all others get DEFAULT_LOGIT_BASE=-100000 (minus tile-dependent offset); instantiates real `lm_head_controller` + real `argmax_reduction`
- `rtl/top/tinyllama_u55c_kernel_top.sv` updated (Phase 9d): `runtime_lm_head_tail` instantiated; `start_i = lm_head_start || argmax_start` (both fire simultaneously from controller, OR is safe); `target_token_i = 1000 + ctrl_generated_token_count` (sim shim); `hw_current_block` extended to track BLOCK_FINAL_RMSNORM, BLOCK_LM_HEAD, BLOCK_ARGMAX phases; `runtime_token_fire_q` registered pipeline: handshake → fire_q → controller sees token_valid → stop check on same cycle
- **Remaining cosmetic issues (not fixed):** `decoder_context_valid` and `runtime_lm_context_valid` are dead wires; `runtime_lm_head_tail.busy_o = ()` left floating; `hw_error_valid` sustained-level mixing (Phase 8 carry-forward)
- **Next scaffold retirement needed:** runtime_decoder_datapath final hidden passthrough → real final RMSNorm; runtime_lm_head_tail logit shim → real shared-GEMM LM head GEMM datapath
- **Sim times (Phase 9d):** tb_runtime_lm_head_tail 8.3 µs, tb_runtime_decoder_datapath 169.8 µs, tb_kernel_top_smoke 400.8 µs (+228 µs from Ph9b; real lm_head 250 tiles + argmax runs twice for max_new_tokens=2), tb_kernel_top_acceptance 444.2 µs, tb_shell_wrapper_smoke 401.4 µs

### Session 20 (2026-04-21)
- **Phase 9e final RMSNorm tail verified** — all 9 testbenches confirmed PASS
- `rtl/top/runtime_final_rmsnorm_tail.sv` — issues gamma DMA read on `launch_i` (gamma_req_valid_q → embedding_lmhead_dma_reader with TENSOR_FINAL_RMS_GAMMA); scale-first gate: `hidden_scale_ready_o = !scale_seen_q`, `hidden_act_ready_o = scale_seen_q && rms_act_ready`; captures `input_scale_q <= hidden_scale_i.data[0]`; clears scale_seen_q after norm_done_pulse; `helper_rst_n = ap_rst_n && !abort_req_i` propagates abort to DMA reader and rmsnorm_wrapper; instantiates real `embedding_lmhead_dma_reader` + real `rmsnorm_wrapper`
- `rtl/nonlinear/rmsnorm_core_hls_ip.sv` — simulation model replacing HLS-synthesized IP; 5-state FSM (IDLE→GAMMA→ACT→COMPUTE→OUT); CORE_COMPUTE is single-cycle floating-point RMSNorm (sumsq → inv_rms → scale × gamma → banker's rounding); memory arrays are sim-only (128 KB FFs for act/out buffers) — not for synthesis
- `rtl/top/tinyllama_u55c_kernel_top.sv` updated (Phase 9e): `runtime_final_rmsnorm_tail` instantiated between decoder and lm_head_tail; `decoder_final_scale_ready/act_ready` now driven by `runtime_final_rmsnorm_tail.hidden_scale_ready_o/hidden_act_ready_o`; `runtime_lm_head_tail` inputs changed from decoder outputs to `final_rms_scale/act` outputs; `hidden_done_pulse_i = final_rms_done_pulse`; new `EMRD_FRONTEND/EMRD_FINAL` embed DMA mux arbitrates between embedding frontend and final_rms gamma read on the shared `embed_lm` HBM port (priority: frontend > final_rms); beat counter `embed_rd_beats_remaining_q` tracks burst completion; `output_scale_i = decoder_final_scale_bus.data[0]` wired as live bus signal
- `rtl/tb/tb_runtime_final_rmsnorm_tail.sv` — uses Phase 5 layer-0 rmsnorm1 prefill golden traces; drives gamma DMA model (128 beats) + hidden_scale + 64 act tiles; checks output within ±16 INT8 tolerance vs expected y_tiles; timeout 12000 cycles; PASS 24.5 µs
- **Known design issue — `output_scale_i` timing:** `rmsnorm_wrapper` captures `output_scale_q` in RN_IDLE every cycle; leaves IDLE when gamma starts (immediately after launch); but `decoder_final_scale_bus` is only valid much later (after decoder finishes 22 layers); so `output_scale_q = 0` in real integration — masked by: (a) unit test uses static `output_scale_i`, (b) integration tests use logit shim; fix before shim retirement: make `output_scale_i` a static config constant, not a live bus tap
- **Testbench coverage gaps:** only prefill case (row_count=16) tested — no decode case (row_count=1); no output tag checks on `captured_scale.tag`; golden traces are from layer-0 rmsnorm1 (not final rmsnorm), though math is identical
- **Sim times (Phase 9e):** tb_runtime_final_rmsnorm_tail 24.5 µs, tb_kernel_top_smoke 430.3 µs (+29 µs from Ph9d; real gamma DMA + rmsnorm_wrapper per forward pass), tb_kernel_top_acceptance 476.2 µs, tb_shell_wrapper_smoke 430.9 µs

### Session 21 (2026-04-21)
- **Phase 9e hardening verified** — all 8 testbenches confirmed PASS
- **`output_scale_i` bug fixed:** `tinyllama_u55c_kernel_top.sv` line 45 adds `localparam logic [SCALE_W-1:0] FINAL_RMS_OUTPUT_SCALE_Q16 = 32'h0001_0000` (= 1.0 Q16.16); line 477 connects `.output_scale_i(FINAL_RMS_OUTPUT_SCALE_Q16)` replacing the live bus tap `decoder_final_scale_bus.data[0]`; value 1.0 is exact match for decode golden trace output_scale; prefill golden trace has 0x9b4 so integration uses a different scale than prefill-case unit test — acceptable for scaffold since logit shim masks quality
- **`tb_runtime_final_rmsnorm_tail.sv` hardened:** decode case added (`DECODE_BASE = "sim/golden_traces/phase5/rtl/phase5_decode_layer0_rmsnorm1"`); both prefill (row_count=16, is_partial=0) and decode (row_count=1, is_partial=1) cases run sequentially; scale tag check added (11 tag fields: layer_id, block_id, gemm_mode, tile_id, token_base, seq_count, q_head_id, kv_head_id, elem_count, is_last, is_partial — payload value check is separate); per-tile act tag check added (11 fields × 64 tiles); per-tile capture arrays added (`captured_layer_id/block_id/gemm_mode/tile_id/token_base/seq_count/q_head_id/kv_head_id/elem_count/is_last/is_partial[0:FEATURE_TILE_COUNT-1]`); tag arrays reset between cases
- **Integration assertions added to all 3 top-level testbenches:** `tb_kernel_top_smoke.sv`, `tb_kernel_top_acceptance.sv`, `tb_shell_wrapper_smoke.sv` each add `final_rms_scale_count` counter and hierarchical probe `dut.u_runtime_final_rmsnorm_tail.u_final_rmsnorm_wrapper.output_scale_q != FINAL_RMS_OUTPUT_SCALE_Q16` on every `final_rms_scale_valid && final_rms_scale_ready` handshake; end-of-sim guard checks `final_rms_scale_count != 0`; shell_wrapper uses deeper path `dut.u_tinyllama_u55c_kernel_top.u_runtime_final_rmsnorm_tail...`
- **Sim times (Phase 9e hardening):** tb_runtime_final_rmsnorm_tail 29.8 µs (+5.3 µs from decode case), tb_runtime_decoder_datapath 169.8 µs (unchanged), tb_runtime_embedding_frontend 4.1 µs (unchanged), tb_runtime_lm_head_tail 8.3 µs (unchanged), tb_rmsnorm_wrapper 53.9 µs (unchanged), tb_kernel_top_smoke 430.3 µs, tb_kernel_top_acceptance 476.2 µs, tb_shell_wrapper_smoke 430.9 µs

### Session 22 (2026-04-22)
- **Phase 9f real LM head verified** — all 8 testbenches confirmed PASS
- `rtl/top/runtime_lm_head_tail.sv` completely rewritten: real TENSOR_LM_HEAD DMA; LMHEAD_GROUPS=4 groups of N_TILE=32 lanes; LMHEAD_GROUP_BYTES=2048 per group; LMHEAD_TILE_BYTES=8192 per vocab tile (= 4 groups × 32 lanes × INT8 × 2 channels); hidden last row captured per act tile into `hidden_last_row_tile_q[FEATURE_TILE_COUNT-1:0][N_TILE*ACT_W-1:0]`; `mac_valid_i = 1'b1` safe because accumulation gates on `operands_fire_d = operands_valid_i && !snapshot_valid_q` and `operands_valid_i = lmhead_group_active_q && lmhead_wt_valid`; clear_acc at feature_idx==0, emit_acc at feature==63 && last beat; group result captured into `tile_logits_q[(active_group_q × N_TILE + lane)]`; 3-stage logits pipeline: tile_complete_q → lmh_sched_done_q+lmh_logits_bus_q → lmh_logits_valid_q; instantiates real `shared_gemm_engine` + `lm_head_controller` + `argmax_reduction`; dead code: `target_token_i` captured into `target_token_q` but never used (shim leftover)
- `rtl/top/tinyllama_u55c_kernel_top.sv` updated (Phase 9f): EMRD 3-way mux (EMRD_FRONTEND > EMRD_FINAL > EMRD_LM_HEAD) on shared embed HBM port; `LM_HEAD_BASE_ADDR = 64'h0000_0000_2000_0000`; runtime_lm_head_tail DMA fully wired; `embed_lm_rd_desc_valid` mux extended to include lm_head_rd_desc_valid
- `rtl/tb/tb_runtime_lm_head_tail.sv` rewritten (Phase 9f): reads phase6 decode golden traces; overwrites last row with synthetic values to make target_token=7 win; `pack_lmhead_weight_beat` constructs sign-consistent weights; DMA model tracks LMHEAD_GROUPS=4 descriptor bursts (64 beats each); checks sched_start_count==1, rd_desc_count==4, first_tile_winner==7; PASS 6.1 µs
- `rtl/tb/tb_kernel_top_smoke.sv` updated (Phase 9f): `pack_lmhead_weight_beat` reads real DUT `hidden_last_row_tile_q` via hierarchy; constructs sign(hidden_val) weight for token 0, -sign for all others; guarantees token 0 wins; end check: token_id == 0; PASS 3,075 µs (+7× from Phase 9e; real GEMM 250 tiles × 4 groups × 64 beats = 64,000 beats per forward pass)
- **Snapshot timing verified:** `shared_gemm_engine` line 161: `snapshot_q.data <= bank_load_d` — `bank_load_d` is the combinatorial `acc_mac_d` including the final beat; no off-by-one on last-tile capture
- **Codex uncertainty assessment:** Claims of "not green locally" and "timed out" were environmental (stale VVP / Windows paths / slow machine), not design bugs; all 8 testbenches are PASS; "not fully proven end-to-end" is partially correct — `runtime_decoder_datapath.sv` still emits raw passthrough as final hidden, so real inference quality is not yet validated; the LM head RTL itself is correct
- **Remaining scaffold:** `runtime_decoder_datapath.sv` passthrough → needs real decoder output (shared_gemm_engine attention + FFN path) to complete the inference chain
- **Sim times (Phase 9f):** tb_runtime_lm_head_tail 6.1 µs, tb_runtime_decoder_datapath 169.8 µs (unchanged), tb_runtime_final_rmsnorm_tail 29.8 µs (unchanged), tb_kernel_top_smoke 3,075 µs, tb_kernel_top_acceptance 3,121 µs, tb_shell_wrapper_smoke 3,116 µs

### Session 23 (2026-04-22)
- **Phase 9f hardening verified** — `target_token_i` port fully removed from `runtime_lm_head_tail.sv` and all instantiations; dead registers (`target_token_q`, `act_seen_q`, `lmh_context_valid`) removed; `tb_runtime_lm_head_tail.sv` hardened with `file_exists()` fail-fast, `$fatal(1,...)` on missing fixtures, XSIM-compatible `always @(posedge clk)` DMA responder; `sim/run_tb_runtime_lm_head_tail.ps1` XSIM canonical run script added; confirmed no remaining `target_token` references in RTL
- **Phase 9g decoder datapath tail verified** — `runtime_decoder_datapath.sv` rewritten: 8-state DDP FSM adds DDP_TAIL_SEND + DDP_TAIL_WAIT; real `residual_add` + `requantize_unit` instantiated (`u_tail_residual_add`, `u_tail_requantize`); two-pass tail: pass 0 (update = neighbor>>1 + bias), pass 1 (update = current − neighbor>>1 + bias), both passes add current to update before requant; 64 tiles × 2 passes × 2 states = 256 tail cycles; `tail_signature_q` captures post-block-22×146 XOR hash; `tb_runtime_decoder_datapath.sv` rewritten with exact tile verification (calls `tb_apply_tail_pass` twice, checks `!== act_tiles_mem` passthrough rejection, reads `dut.tail_signature_q` directly)
- **No-deadlock confirmed:** `residual_add.sv` both `residual_ready_o = !residual_captured_q` and `update_ready_o = !update_captured_q` start high after reset → single-cycle dual-capture on first DDP_TAIL_SEND → DDP_TAIL_WAIT fires next cycle; `sum_ready_i=1'b1` clears flags immediately; no interleaved-only deadlock risk
- **Independent verification (unit test):** Fresh compile + `vvp sim/tb_runtime_decoder_datapath_indep.vvp`: PASS at 172375000 ps, starts=3212 dones=3212 run_done=1 — exact match with Codex's claimed result; 3212 = 22 × 146 confirmed
- **Integration test:** Fresh compile of `sim/tb_kernel_top_smoke_9g.vvp` (no errors); run confirmed PASS at 3080380000 ps (3,080.38 µs)
- **Sim times (Phase 9g):** tb_runtime_decoder_datapath 172.375 µs (+2.6 µs from Phase 9c; 256 tail cycles × 10 ns = 2.56 µs); tb_kernel_top_smoke 3,080.38 µs (matches Codex's claimed ~3,080 µs)

### Session 24 (2026-04-22)
- **Phase 9h per-block decoder update verified** — `runtime_decoder_datapath.sv` rewritten: DDP_APPLY_SEND/APPLY_WAIT replace DDP_TAIL_SEND/TAIL_WAIT; one tile updated per block (not per pass at end); `tile_cursor_q` advances by block-dependent odd stride (strides forced odd to avoid long dead zones, but full 64-tile coverage is empirically confirmed by the testbench's `expected_touched_tiles_q == all_1` assertion — not deduced from stride math alone); update arithmetic is block-type-specific across 8 cases; `apply_signature_q` captures post-block XOR at end of DDP_BLOCK (before APPLY_SEND), used for both the tile arithmetic and next stride; `context_signature_q` updated in DDP_APPLY_WAIT; `block_done_o` fires in DDP_APPLY_WAIT after requantize
- **Testbench redesign:** `tb_runtime_decoder_datapath.sv` rewritten: shadow-tracks entire tile state `expected_final_tiles[64]`, cursor `expected_tile_cursor_q`, signature `expected_signature_q` — all updated on each `block_done`; verifies `expected_touched_tiles_q == all_1` (all 64 tiles touched); verifies all tiles bit-exact vs shadow; reports `changed_tiles=64` (all differ from raw embedding)
- **Timing arithmetic confirmed:** Phase 9g 172.375 µs + (3212 × 2 APPLY cycles − 256 old tail cycles) × 10 ns = 172.375 + 61.68 = 234.055 µs — exact match
- **Independent verification (unit test):** Fresh compile + `vvp sim/tb_runtime_decoder_datapath_9h.vvp`: PASS at 234055000 ps, starts=3212 dones=3212 changed_tiles=64 run_done=1 — exact match with Codex's claimed result
- **Integration tests:** All 3 Codex log files confirmed real (UTF-16 LE, content verified): tb_kernel_top_smoke 3203.74 µs, tb_kernel_top_acceptance 3249.70 µs, tb_shell_wrapper_smoke 3244.40 µs; fresh independent compile + run of `tb_kernel_top_smoke_9h.vvp` PASS at 3203740000 ps — exact match; all three top-level gates have confirmed passing evidence
- **Sim times (Phase 9h):** tb_runtime_decoder_datapath 234.055 µs; tb_kernel_top_smoke 3,203.74 µs; tb_kernel_top_acceptance 3,249.70 µs; tb_shell_wrapper_smoke 3,244.40 µs

### Session 25 (2026-04-22)
- **Phase 9i FFN chain closure verified** — all 10 testbenches confirmed PASS
- `rtl/top/runtime_decoder_datapath.sv` expanded to 11-state 4-bit enum: adds DDP_SILU_APPLY (4'd6), DDP_MUL_SEND (4'd7), DDP_MUL_WAIT (4'd8); FFN side buffers: `ffn_gate_tiles_q`, `ffn_up_tiles_q`, `ffn_silu_tiles_q`, `ffn_mul_tiles_q` (64 tiles each); FFN anchor tracking: `ffn_tile_anchor_q` and `ffn_stride_q` set when `is_ffn_anchor_block(BLOCK_GATE)` is true, reused for all subsequent FFN chain blocks (UP, SILU, GLU_MUL, DOWN) without advancing the main tile cursor
- **FFN chain coherence:** BLOCK_GATE → `ffn_gate_tiles_q[anchor]`; BLOCK_UP → `ffn_up_tiles_q[anchor]`; BLOCK_SILU (single-cycle DDP_SILU_APPLY, combinatorial `silu_tile_d`) → `ffn_silu_tiles_q[anchor]`; BLOCK_GLU_MUL (real `elementwise_mul u_block_ffn_mul` handshake, DDP_MUL_SEND→MUL_WAIT) → `ffn_mul_tiles_q[anchor]`; BLOCK_DOWN reads `ffn_mul_tiles_q[anchor]` → `hidden_tiles_q[anchor]` (final output update); chain is coherent and data-correct
- **DDP_SILU_APPLY timing savings:** BLOCK_SILU uses a **single update state** (DDP_SILU_APPLY), not a one-cycle whole block — DDP_BLOCK countdown still precedes it; the savings is 1 cycle in the update-execution step vs 2 cycles (APPLY_SEND+APPLY_WAIT); 22 layers × 1 saved cycle = 22 cycles = 220 ns; unit sim time 233.835 µs vs 234.055 µs Phase 9h (−0.22 µs, exact match)
- **`mul_done_w` fires 22× (once per GLU_MUL block, once per layer):** `elementwise_mul.done_pulse_o` asserted at end of MUL_WAIT; testbench tracks `dut.mul_done_w` directly and asserts `mul_done_count == N_LAYERS = 22` at end of sim
- `rtl/tb/tb_runtime_decoder_datapath.sv` rewritten (Phase 9i): adds FFN shadow arrays (`expected_gate/up/silu/mul_tiles_q`), `expected_ffn_tile_anchor_q`, `expected_ffn_stride_q`; `tb_apply_silu_tile`, `tb_apply_glu_mul_tile`, `tb_apply_down_update` functions mirror DUT FFN math; `always_ff` shadow update dispatches BLOCK_GATE/UP/SILU/GLU_MUL/DOWN to separate shadow arrays then routes BLOCK_DOWN to `expected_final_tiles`; `mul_done_count` counter probes `dut.mul_done_w`; final `mul_done_count != N_LAYERS` guard added; `changed_tiles=64` and `mul_done=22` confirmed in PASS line
- **`apply_residual_flat_d` gating:** always_comb gates on `state_q == DDP_APPLY_SEND || DDP_APPLY_WAIT`; in DDP_SILU_APPLY, DDP_MUL_SEND, DDP_MUL_WAIT the combinatorial data is '0 but `residual_valid_i = (state_q == DDP_APPLY_SEND)` prevents erroneous captures — no stale-data risk
- **Independent compilation confirmed:** `iverilog -g2012` with `elementwise_mul.sv` added; only Icarus warnings (constant selects, unique case), no errors; `vvp` run from project root: PASS at 233835000 ps — exact match
- **6 testbenches freshly rerun for Phase 9i:** tb_runtime_decoder_datapath (29.755 µs), tb_runtime_final_rmsnorm_tail (233.835 µs), tb_runtime_lm_head_tail (6.075 µs), tb_kernel_top_smoke (3203.30 µs), tb_kernel_top_acceptance (3249.26 µs), tb_shell_wrapper_smoke (3243.96 µs); 4 carried forward from prior phases (tb_runtime_embedding_frontend, tb_rmsnorm_wrapper, tb_prefill_decode_smoke, tb_prefill_decode_controller)
- **Sim times (Phase 9i):** tb_runtime_decoder_datapath 233.835 µs (−0.22 µs from Ph9h); tb_kernel_top_smoke 3,203.30 µs; tb_kernel_top_acceptance 3,249.26 µs; tb_shell_wrapper_smoke 3,243.96 µs

### Session 26 (2026-04-23)
- **Phase 9j real SiLU path verified** — all 7 freshly rerun testbenches confirmed PASS
- `rtl/nonlinear/silu_core_hls_ip.sv` — **repo-owned simulation model** (not the final synthesized HLS IP); 3-state FSM (CORE_IDLE→CORE_INPUT→CORE_OUT); uses real-number math (`real` + `$exp`) to compute `x * sigmoid(x)` with Q16.16 banker's rounding; processes N_TILE=32 elements per chunk; `effective_elem_count` handles zero → ACT_VECTOR_ELEMS; for hardware deployment, this would be replaced by the Vivado HLS-synthesized bitstream
- `rtl/top/runtime_decoder_datapath.sv` rewritten (Phase 9j): FSM expanded from 11→13 states; DDP_SILU_APPLY (single update state) replaced by DDP_SILU_SEND (4'd6) + DDP_SILU_SCALE (4'd7) + DDP_SILU_ACT (4'd8); DDP_MUL_SEND/WAIT shift to 4'd9/10; OUT states to 4'd11/12; real `silu_wrapper u_block_silu` instantiated; `silu_done_w = silu_wrapper.done_pulse_o` exposed for testbench monitoring; `silu_input_scale_d = silu_output_scale_d = effective_scale(hidden_scale_q[0])`
- **DDP_SILU_SEND/SCALE/ACT handshake:** DDP_SILU_SEND fires `gate_valid_i`, waits for `silu_gate_ready_w` (silu_wrapper SI_IDLE) → transitions immediately; DDP_SILU_SCALE waits for `silu_scale_valid_w` (silu_wrapper SI_OUT_SCALE, ~4 cycles for HLS 1-chunk processing); DDP_SILU_ACT waits for `silu_act_valid_w` (silu_wrapper SI_OUT_ACT), captures `silu_act_w.data → ffn_silu_tiles_q[apply_tile_idx_q]`, fires `block_done_o`
- **`apply_residual_flat_d` BLOCK_SILU case is now dead code:** it's only evaluated when `state_q == DDP_APPLY_SEND || DDP_APPLY_WAIT`, but BLOCK_SILU now routes to DDP_SILU_SEND — harmless
- **Timing:** +1.1 µs per DDP run is **strongly consistent with** 22 SILU blocks × 5 extra state cycles × 10 ns (DDP_SILU_SEND [1] + HLS CORE_INPUT [1] + HLS CORE_OUT [1] + DDP_SILU_SCALE [1] + DDP_SILU_ACT [1] = 5 extra vs Phase 9i's single update state); not a formal proof — would require cycle-by-cycle trace dumps to verify exactly
- `rtl/tb/tb_runtime_decoder_datapath.sv` updated (Phase 9j): `tb_silu_scalar` function mirrors `silu_core_hls_ip` exactly (dequantize → `x * sigmoid(x)` FP → quantize with banker's rounding); `tb_apply_silu_tile` now takes `input_scale`/`output_scale` parameters; `silu_done_count` counter probes `dut.silu_done_w`; `silu_done_count != N_LAYERS` guard added; PASS line adds `silu_done=22`
- **Independent compilation confirmed:** `iverilog -g2012` with `silu_core_hls_ip.sv` + `silu_wrapper.sv` added; only Icarus warnings (real variable in always_ff), no errors; `vvp` run: PASS at 234935000 ps — exact match
- **7 testbenches freshly rerun for Phase 9j:** tb_silu_wrapper (0.525 µs), tb_runtime_decoder_datapath (234.935 µs), tb_runtime_final_rmsnorm_tail (29.755 µs), tb_runtime_lm_head_tail (6.075 µs, XSIM fixture fallback), tb_kernel_top_smoke (3212.10 µs), tb_kernel_top_acceptance (3258.06 µs), tb_shell_wrapper_smoke (3252.76 µs)
- **Sim times (Phase 9j):** tb_runtime_decoder_datapath 234.935 µs (+1.1 µs from Ph9i); tb_kernel_top_smoke 3,212.10 µs; tb_kernel_top_acceptance 3,258.06 µs; tb_shell_wrapper_smoke 3,252.76 µs

### Session 27 (2026-04-23)
- **Phase 9k real FFN projection GEMMs verified** — all 5 freshly rerun testbenches confirmed PASS
- `rtl/top/runtime_decoder_datapath.sv` expanded to 15-state FSM: adds DDP_GEMM_SEND (4'd6) + DDP_GEMM_WAIT (4'd7); SILU/MUL/OUT states shift up by 2; `is_ffn_projection_block()` returns true for BLOCK_GATE/UP/DOWN → routes to DDP_GEMM_SEND/WAIT instead of DDP_APPLY_SEND
- **Real `shared_gemm_engine u_block_projection_gemm` instantiated:** `projection_weight_scalar()` generates deterministic synthetic INT8 weights per (lane, block, layer, signature, tile); `clear_acc` + `mac_valid` + `emit_acc` all asserted in DDP_GEMM_SEND (single-beat GEMM); `acc_ready_i = (state_q == DDP_GEMM_WAIT)`
- **BLOCK_GATE/UP path:** DDP_GEMM_SEND → DDP_GEMM_WAIT; result stored in `ffn_gate_tiles_q[anchor]` or `ffn_up_tiles_q[anchor]`; `block_done_o` fires in DDP_GEMM_WAIT; tile cursor does NOT advance (FFN anchor mechanism, only BLOCK_DOWN advances cursor)
- **BLOCK_DOWN two-phase path:** DDP_GEMM_SEND → DDP_GEMM_WAIT (captures `gemm_acc_w → down_gemm_acc_q`, sets `down_apply_from_gemm_q=1`) → DDP_APPLY_SEND → DDP_APPLY_WAIT (real `residual_add` with `residual=hidden_tiles_q[anchor]`, `update=down_gemm_acc_q`, then `requantize_unit` → writes `hidden_tiles_q[anchor]`, clears `down_apply_from_gemm_q`); `block_done_o` fires after requantize; tile cursor advances at BLOCK_DOWN only
- **`gemm_acc_valid_w` exposed:** testbench probes `dut.gemm_acc_valid_w && dut.active_block_q == BLOCK_GATE/UP/DOWN` to count 22 per-layer GEMMs each; all three must equal N_LAYERS=22 at end of sim
- **Testbench `tb_apply_projection_tile`:** act_tile × `tb_projection_weight_scalar()` per lane → requantize (mirrors DUT for BLOCK_GATE/UP); stored in `expected_gate/up_tiles_q[anchor]`
- **Testbench `tb_apply_down_update`:** `current_hidden_acc + (mul_tile × projection_weight_scalar(BLOCK_DOWN))` → requantize (mirrors DUT BLOCK_DOWN residual_add+requantize); stored in `expected_final_tiles[anchor]`
- **Timing:** +0.44 µs per DDP run is **strongly consistent with** 22 BLOCK_DOWN blocks × 2 extra GEMM cycles × 10 ns; not a formal proof — would require cycle-by-cycle trace dumps to verify exactly
- **XSIM switch for top-level tests:** real projection GEMM made full kernel_top simulation impractically slow under Icarus; XSIM used for tb_kernel_top_smoke, tb_kernel_top_acceptance, tb_shell_wrapper_smoke; this establishes the practical tool split: **iverilog for focused unit/runtime-helper benches** (decoder unit, final_rmsnorm_tail, lm_head_tail, embedding_frontend) + **xsim for heavy top-level runtime benches** (kernel_top_smoke, kernel_top_acceptance, shell_wrapper_smoke)
- **5 freshly rerun testbenches for Phase 9k:** tb_runtime_decoder_datapath (235.375 µs, Icarus), tb_runtime_final_rmsnorm_tail (29.755 µs, Icarus), tb_kernel_top_smoke (3212.98 µs, XSIM), tb_kernel_top_acceptance (3258.94 µs, XSIM), tb_shell_wrapper_smoke (3253.64 µs, XSIM)
- **Sim times (Phase 9k):** tb_runtime_decoder_datapath 235.375 µs (+0.44 µs from Ph9j); tb_kernel_top_smoke 3,212.98 µs; tb_kernel_top_acceptance 3,258.94 µs; tb_shell_wrapper_smoke 3,253.64 µs

### Session 28 (2026-04-23)
- **Phase 9l attention-output subchain verified** — all 6 testbenches confirmed PASS
- `rtl/top/runtime_decoder_datapath.sv` updated: new side buffers `attn_weighted_tiles_q[FEATURE_TILE_COUNT-1:0]` (BLOCK_WEIGHTED_SUM staging) and `attn_o_tiles_q[FEATURE_TILE_COUNT-1:0]` (BLOCK_O result); new state variables `attn_o_tile_anchor_q` + `attn_o_stride_q` for chain tile coherence
- **BLOCK_WEIGHTED_SUM** (DDP_APPLY_WAIT): result now written to `attn_weighted_tiles_q[apply_tile_idx_q]` instead of `hidden_tiles_q`; attention weighted sum is staged and not directly committed to hidden state — correct because the O projection must be applied first
- **BLOCK_O** added to `is_ffn_projection_block()` → routes through DDP_GEMM_SEND/WAIT: act data read from `attn_weighted_tiles_q[apply_tile_idx_q]`; result stored to `attn_o_tiles_q[apply_tile_idx_q]`; GEMM mode = GEMM_O; `attn_o_tile_anchor_q` and `attn_o_stride_q` set when DDP_BLOCK sees `is_attn_output_anchor_block(BLOCK_O)`; cursor does NOT advance at BLOCK_O
- **BLOCK_RESIDUAL1** (DDP_APPLY_WAIT): update operand reads `attn_o_tiles_q[anchor]` (not the generic `(current >>> 1) + bias` formula used for other blocks); `residual_add` with `residual = hidden_tiles_q[anchor]`, `update = attn_o_tiles_q[anchor]`; result written to `hidden_tiles_q[anchor]`; cursor advances at BLOCK_RESIDUAL1 (chain terminator)
- **`advance_tile_cursor_on_block()` updated:** returns true when NOT in FFN chain AND NOT in attn output chain, OR == BLOCK_DOWN, OR == BLOCK_RESIDUAL1; ensures cursor advances only at chain terminators, not at BLOCK_O
- **New helper functions:** `is_attn_output_chain_block()` (BLOCK_O, BLOCK_RESIDUAL1), `is_attn_output_anchor_block()` (BLOCK_O only); `projection_gemm_mode()` extended with BLOCK_O → GEMM_O case
- **Timing:** unit sim time unchanged at 235.375 µs — BLOCK_O path changed from DDP_APPLY_SEND/WAIT (2 states) to DDP_GEMM_SEND/WAIT (2 states); zero net cycle delta vs Phase 9k
- `rtl/tb/tb_runtime_decoder_datapath.sv` updated (Phase 9l): `expected_weighted_tiles_q` and `expected_o_tiles_q` shadow arrays added; shadow dispatches BLOCK_WEIGHTED_SUM → `expected_weighted_tiles_q`, BLOCK_O → `tb_apply_projection_tile(expected_weighted_tiles_q[anchor])` → `expected_o_tiles_q`, BLOCK_RESIDUAL1 → `tb_apply_residual_stage_update(hidden, scale, expected_o_tiles_q[anchor])` → `expected_final_tiles`; `expected_attn_o_tile_anchor_q`/`expected_attn_o_stride_q` updated at BLOCK_O; `o_gemm_count` counter added (probes `dut.gemm_acc_valid_w && dut.active_block_q == BLOCK_O`); `o_gemm_count != N_LAYERS` end-of-sim guard added; PASS line adds `o_gemm=22`
- **6 testbenches confirmed PASS:** tb_runtime_decoder_datapath (235.375 µs, Icarus, o_gemm=22, starts=3212, dones=3212, changed_tiles=64), tb_runtime_final_rmsnorm_tail (29.755 µs, Icarus), tb_kernel_top_smoke (3212.98 µs, XSIM), tb_kernel_top_acceptance (3258.94 µs, XSIM), tb_shell_wrapper_smoke (3253.64 µs, XSIM); tb_runtime_lm_head_tail carried forward
- **Remaining scaffold in runtime_decoder_datapath.sv:** BLOCK_SCORE, BLOCK_CAUSAL_MASK, BLOCK_SOFTMAX attention compute sub-blocks and the BLOCK_WEIGHTED_SUM weight generation still use synthetic path; real scored-attention chain (QK^T GEMM → mask → softmax → weighted-V GEMM) is the next step
- **Sim times (Phase 9l):** tb_runtime_decoder_datapath 235.375 µs (same as Ph9k); tb_kernel_top_smoke 3,212.98 µs; tb_kernel_top_acceptance 3,258.94 µs; tb_shell_wrapper_smoke 3,253.64 µs

### Session 29 (2026-04-24)
- **Phase 9m attention score chain verified** — tb_runtime_decoder_datapath PASS at 425.455 µs under XSIM; starts=3212 dones=3212 changed_tiles=63 gate_gemm=22 up_gemm=22 down_gemm=22 o_gemm=22 score_gemm=704 wsum_gemm=704 silu_done=22 mul_done=22 run_done=1 ✅
- `rtl/nonlinear/softmax_core_hls_ip.sv` — simulation model for softmax HLS IP boundary; 4-state FSM (CORE_IDLE→CORE_INPUT→CORE_DIGEST→CORE_OUT); `NO_FAST_SOFTMAX` conditional: fast path (default) outputs uniform 1/key_col_count probability per element (skips per-element dequant; tractable at 704 softmax invocations/run); slow path (with `NO_FAST_SOFTMAX`) runs real stable softmax (per-row max subtraction, exp, normalize) with Q16.16 banker's rounding; `score_real_q` array gated by `NO_FAST_SOFTMAX`; `digest_uniform_prob_q16` scratch for fast path
- `rtl/top/runtime_decoder_datapath.sv` expanded to 20-state 5-bit enum: adds DDP_MASK_APPLY (5'd15), DDP_SOFTMAX_ARM (5'd16), DDP_SOFTMAX_SCORE (5'd17), DDP_SOFTMAX_SCALE (5'd18), DDP_SOFTMAX_ACT (5'd19); `is_ffn_projection_block()` extended to include BLOCK_SCORE and BLOCK_WEIGHTED_SUM → DDP_GEMM_SEND/WAIT; new state variables `attn_score_acc_q`, `attn_masked_acc_q`, `attn_prob_tile_q`; causal_mask_unit + softmax_wrapper instantiated
- **BLOCK_SCORE** (DDP_GEMM_SEND/WAIT): act = hidden_tiles_q[apply_tile_idx_q] (Q vector for QK^T); GEMM_SCORE mode; result stored as raw INT32 to `attn_score_acc_q` (not requantized); cursor advances; `block_done_o` fires
- **BLOCK_CAUSAL_MASK** (DDP_MASK_APPLY): single-cycle; `causal_mask_unit` applies NEG_INF mask to `attn_score_acc_q` → `attn_masked_acc_q`; cursor advances; `block_done_o` fires
- **BLOCK_SOFTMAX** (DDP_SOFTMAX_ARM→SCORE→SCALE→ACT): 4-state handshake through `softmax_wrapper`; ARM fires `sfw_start_i`; SCORE waits for `sfw_in_ready_w` then feeds `attn_masked_acc_q`; SCALE waits for `sfw_out_scale_valid_w`; ACT captures `sfw_out_act_w.data → attn_prob_tile_q`, fires `block_done_o`; cursor advances
- **BLOCK_WEIGHTED_SUM** (DDP_GEMM_SEND/WAIT): act = `attn_prob_tile_q` (softmax prob tile, not hidden); GEMM_WEIGHTED_SUM mode; result stored via `gemm_requant_w.data → attn_weighted_tiles_q[apply_tile_idx_q]`; cursor advances
- **Two RTL bugs found and fixed before PASS run:** (1) tile cursor advance missing in DDP_MASK_APPLY, DDP_SOFTMAX_ACT, and DDP_GEMM_WAIT BLOCK_SCORE/WEIGHTED_SUM cases — added `if (advance_tile_cursor_on_block(active_block_q)) tile_cursor_q <= next_tile_cursor(...)` in each; (2) duplicate BLOCK_CAUSAL_MASK in block_latency unique case (latency 2 and latency 1 groups) — removed from latency-2 group, now appears only in latency-1 group
- `rtl/tb/tb_runtime_decoder_datapath.sv` updated: shadow arrays `expected_score_acc_flat_q`, `expected_masked_acc_flat_q`, `expected_prob_tiles_q`; counters `score_gemm_count` (probes `dut.gemm_acc_valid_w && BLOCK_SCORE`), `wsum_gemm_count`; attention blocks mirror DUT state directly (not independently recomputed); `score_gemm != N_LAYERS*N_Q_HEADS` and `wsum_gemm != N_LAYERS*N_Q_HEADS` end-of-sim guards; PASS line adds `score_gemm=%0d wsum_gemm=%0d`
- `sim/decoder_filelist.txt` — 17-file compile list including softmax_core_hls_ip.sv; `sim/run_tb_runtime_decoder_datapath.ps1` — XSIM + iverilog canonical run script with fixture staging and PASS-string check
- **`changed_tiles=63` note:** hard assertion `expected_touched_tiles_q == all_1` (line 1487) confirmed all 64 tiles touched; one tile's final INT8 value coincidentally matches its initial embedding value through deterministic synthetic arithmetic — not a coverage gap
- **+190 µs timing:** 22 layers × 32 heads × 4 attention states (SCORE/CAUSAL_MASK/SOFTMAX/WEIGHTED_SUM) × softmax HLS pipeline ≈ 704 softmax invocations × ~270 ns each
- **Sim times (Phase 9m):** tb_runtime_decoder_datapath 425.455 µs (+190.08 µs from Ph9l; score_gemm=704 wsum_gemm=704 confirmed)

### Session 30 (2026-04-25)
- **Phase 9m regression hardening verified** — all three proof mechanisms confirmed present and correct; dual-mode PASS confirmed (fast: `changed_tiles=62`; full_softmax: `changed_tiles=63`)
- **TB coverage fixes (two gaps addressed):**
  - `expected_final_tiles` end-of-sim assertion added (`tb_runtime_decoder_datapath.sv` lines 1552–1557): loops all 64 tiles comparing `captured_final_tiles[t] !== expected_final_tiles[t]`; `$error` + `$finish` on mismatch; previously shadow model existed but was never checked at end-of-sim
  - `expected_latest_prob_tile_q` single-latch fix: BLOCK_WEIGHTED_SUM TB shadow now consumes `expected_latest_prob_tile_q` (written at BLOCK_SOFTMAX from `dut.softmax_prob_bus_w.data`) instead of `expected_prob_tiles_q[expected_apply_tile]`; mirrors DUT's unindexed `attn_prob_tile_q` register correctly
- **Mode provenance proof mechanisms (both verified in run logs):**
  - `softmax_core_hls_ip.sv` startup banner (`initial begin`): prints "fast path (default)" or "slow path (NO_FAST_SOFTMAX defined)" at sim start; visible in xsim.log; verified in `20260425_070828/xsim.log` (fast) and `20260425_070912/xsim.log` (full)
  - `sim/run_tb_runtime_decoder_datapath.ps1` writes `build_args.txt` to log folder before xvlog invocation; records simulator, mode, full xvlog command, library, snapshot, quant_base; verified in `20260425_065413/build_args.txt` (full_softmax, `--define NO_FAST_SOFTMAX` present) and `20260425_065320/build_args.txt` (fast_softmax, no define)
- **`changed_tiles=62` (fast) vs `changed_tiles=63` (full) is expected behavior:** fast path produces uniform 1/N probabilities → WEIGHTED_SUM outputs uniform V-weighted tiles → one tile coincidentally matches initial embedding value; real softmax breaks symmetry → different INT8 output at that tile; both modes PASS the hard `expected_touched_tiles_q == all_1` assertion (all 64 tiles touched in both modes)
- **Attention validity tracking additions to `runtime_decoder_datapath.sv` (Cursor AI) — verified correct:**
  - Five new validity registers: `attn_score_valid_q`, `attn_masked_valid_q`, `attn_prob_valid_q`, `attn_weighted_valid_q[FEATURE_TILE_COUNT]`, `attn_o_valid_q[FEATURE_TILE_COUNT]`; reset on reset/launch/abort; cleared per-head at BLOCK_SCORE start
  - `gemm_operands_valid_w` wire gates only WEIGHTED_SUM GEMM launch on `attn_prob_valid_q`; O/SCORE/GATE/UP/DOWN ungated (critical: gating these caused deadlock in earlier Cursor AI attempt because layer_controller guarantees their input data is already present)
  - `softmax_score_valid_w = (state_q == DDP_SOFTMAX_SCORE) && attn_masked_valid_q` — real protocol gate preventing score feed before mask has run
  - DDP_MASK_APPLY waits on `attn_score_valid_q` before applying mask — prevents mask from consuming uninitialized score (important at reset/abort boundary)
  - Validity lifecycle: SCORE sets score_valid → MASK consumes, sets masked_valid → SOFTMAX_SCORE consumes masked_valid → SOFTMAX_ACT sets prob_valid → WEIGHTED_SUM consumes prob_valid, sets weighted_valid[t] → O consumes weighted_valid[t], sets o_valid[t] → RESIDUAL1 consumes o_valid[t]
- **`attn_prob_scale_q` for WEIGHTED_SUM requantize:** genuine correctness improvement; WEIGHTED_SUM now uses scale captured from `softmax_wrapper` scale output (`PROB_SCALE_Q16 = 516 = 1/127 Q16.16`) instead of incorrect `hidden_scale_q` (embedding input scale); `attn_score_scale_q` also captured at BLOCK_SCORE completion for softmax input scaling
- **`~/.claude/settings.json` merged** — was two separate JSON objects (invalid); merged to single object: `{"effortLevel":"max","permissions":{"allow":["Bash(*)","PowerShell(*)"]}}` — covers all commands including xsim/iverilog
- **Key lesson on valid-gating in sequenced FSMs:** In a strictly-sequenced FSM (like DDP driven by `layer_controller`), only gate on validity flags for cross-chain or asynchronous data dependencies. Gates on sequential in-chain data (e.g. gating O on `attn_weighted_valid_q`) cause spurious stalls when the valid bit clears between the set and the consuming state; the layer_controller ordering guarantee makes such gates redundant and dangerous
- **Sim times (Phase 9m hardened):** tb_runtime_decoder_datapath: fast PASS 425.455 µs (`changed_tiles=62`), full PASS 425.455 µs (`changed_tiles=63`); both under XSIM

### Session 31 (2026-04-27)
- **First Vivado synthesis completed on SOL** — job 51905999; `synth_design` succeeded, 0 errors, 0 critical warnings; LUTs 91.71% (1,195,571 / 1,303,680), Registers 17.30%, DSPs 11.35% (1024 / 9024), BRAMs 0%; WNS −72.633 ns on ap_clk (300 MHz target, synthesis timing pessimistic without placement); 274,543 failing endpoints
- **Root cause of 0 BRAM / 91.71% LUTs identified:** for-loop bulk clears in `runtime_decoder_datapath.sv` reset and launch blocks write all 64 LUTRAM entries in one cycle — impossible in RAM primitives; Vivado falls back to FFs + mux trees; 7 large tile arrays (64 × 256-bit each) synthesized as FFs instead of LUTRAM
- **BRAM fix applied to `rtl/top/runtime_decoder_datapath.sv`:** added `(* ram_style = "distributed" *)` to all 7 tile array declarations (`hidden_tiles_q`, `ffn_gate/up/silu/mul_tiles_q`, `attn_weighted/o_tiles_q`); removed reset for-loop (was lines 964–972); removed launch for-loop (was lines 1011–1018); validity bits (`tile_seen_q`, `attn_weighted_valid_q`, `attn_o_valid_q`) guarantee write-before-read so clearing is functionally unnecessary
- **BRAM fix verified with XSIM** — `.\sim\run_tb_runtime_decoder_datapath.ps1` (default XSIM mode): PASS in 12 seconds wall-clock; `starts=3212 dones=3212 changed_tiles=62 score_gemm=704 wsum_gemm=704 silu_done=22 mul_done=22 run_done=1`; previously stuck 5 Icarus vvp processes running for >2 hours were killed
- **GEMM_LANES unified to 512:** removed `ifdef SYNTHESIS` / `else` / `endif` blocks from `tinyllama_pkg.sv` for `GEMM_LANES` (64→512), `M_TILE` (2→16), `VOCAB_TILE` (64→128); simulation and synthesis now use same values; 480G SLURM request rejected (`QOSMaxMemoryPerJob` — 256G is the public partition ceiling)
- **Synthesis parameters updated:** `--cpus-per-task` 4→8; `set_param synth.maxThreads` 4→8 (Vivado maximum); clock constraint 300 MHz→100 MHz (reduces failing endpoint count from 274K, lowers XDC constraints phase RAM and time); 256G memory limit retained (partition max)
- **`docs/sol_synthesis_guide.md` created** — reference for all SOL/SLURM/Vivado commands, file transfer, monitoring, milestone timing, common failures; updated with QOS 256G limit, 8-thread recommendation, 100 MHz clock note, LUTRAM inference failure entry
- **Job 51924560 submitted** — 512 lanes + BRAM fix + 8 threads + 100 MHz clock; outcome pending

### Session 32 (2026-04-28)
- **512-lane synthesis conclusively fails on Vivado 2022.1** — all five configurations attempted, all crash with `NDup::dupNameType(UAggType*)` signal 11 in `dupBaseMod → dupPorts → dupNameType`; this is a width-dependent Vivado 2022.1 tool bug: 64-lane packed struct bus widths (~2000 bits) work; 512-lane widths (ACC_BUS_W=16463 bits) crash regardless of settings
- **Five failed 512-lane jobs:**
  - Job 51924560: ram_style=distributed + port shims + flatten_hierarchy full → crash
  - Job 51941077: port shims removed + no ram_style + flatten_hierarchy full → crash
  - Job 51958926: same as 51941077 (re-run, clean RTL confirmed) → crash
  - Job 51961189: flatten_hierarchy rebuilt → crash
  - Job 51961647: flatten_hierarchy none → crash (even without any hierarchy collapsing)
- **Port shim removal:** Python script `strip_shims.py` removed all `ifdef SYNTHESIS` flat-port shim blocks from 35 RTL modules; struct-typed ports fully restored; no internal alias wires remain; change verified correct with XSIM (tb_runtime_decoder_datapath PASS, changed_tiles=62)
- **`(* ram_style = "distributed" *)` removed:** all 7 tile array declarations in `runtime_decoder_datapath.sv` reverted to plain `logic`; for-loop bulk clears not restored (validity bits guarantee write-before-read); XSIM still PASS
- **Milestone synthesis artifact:** job 51905999 reports in `synth/out/` — LUTs 91.71% (1,195,571 / 1,303,680), Registers 17.30%, DSPs 11.35% (1024 / 9024), BRAMs 0%, 0 errors, 0 critical warnings; 64-lane design; 64→512 lane change is purely parametric (GEMM_LANES in tinyllama_pkg.sv), design logic proven correct by simulation at 512 lanes
- **TCL reverted to proven-working settings:** `flatten_hierarchy full`, comment documents the 512-lane crash and the milestone artifact decision
- **For report/presentation:** "Synthesis verified at 64 lanes (job 51905999, clean); 512-lane synthesis blocked by Vivado 2022.1 tool bug (NDup::dupNameType on ACC_BUS_W≥16K-bit packed structs); design correctness at 512 lanes validated by simulation."

### Session 33 (2026-04-28)
- **GEMM_LANES reverted to 64:** `tinyllama_pkg.sv` changes: `GEMM_LANES=64`, `M_TILE=2` (was 16), `VOCAB_TILE=64` (was 128); comment updated; synthesis and simulation now coherent at same lane count
- **Simulation re-verified at 64 lanes:** `tb_runtime_decoder_datapath` XSIM PASS at 228.335 µs (down from 425 µs at 512 lanes — 64-lane GEMM engine completes each pass faster); `starts=3212 dones=3212 changed_tiles=62 gate_gemm=22 up_gemm=22 down_gemm=22 o_gemm=22 score_gemm=704 wsum_gemm=704 silu_done=22 mul_done=22 run_done=1`; all structural counts (FEATURE_TILE_COUNT=64, 22×146=3212 blocks) unchanged — these derive from N_TILE=32 and model architecture, not GEMM_LANES
- **Job 51963647: full RTL synthesis SUCCESS at 64 lanes** — phases 9c–9m RTL included (complete attention-score chain, FFN projections, SiLU, residual adds); `flatten_hierarchy full`, `maxThreads 8`, 8h wall clock; 0 errors, 0 critical warnings
  - CLB LUTs: **83.22%** (1,084,885 / 1,303,680) — down from 91.71% in job 51905999 (for-loop bulk-clear removal saved ~110K LUTs; phases 9g–9m added less logic than the for-loops wasted)
  - CLB Registers: **9.52%** (248,238 / 2,607,360)
  - Block RAM: **0%** — validity-bit write-before-read guarantee; no LUTRAM inference attempted
  - DSPs: **11.35%** (1,024 / 9,024) — unchanged vs job 51905999 (MAC lanes + requantize Q16.16 multipliers + elementwise arithmetic)
  - WNS: **−60.988 ns** at 100 MHz target (synthesis pessimism without placement; hold timing clean: WHS = +0.057 ns, 0 hold violations); 66,124 failing setup endpoints of 570,994 total
- **`NDup::dupNameType(UAggType*)` root cause confirmed:** `UAggType` = Vivado 2022.1 internal representation of SystemVerilog packed-struct aggregate types (not VHDL aliases as some community posts describe); crash occurs during **Global Synthesis** (`writeAGenome` → `dumpDesign` phase), not RTL elaboration; size threshold is between ACC_BUS_W=2127 bits (64 lanes, works) and ACC_BUS_W=16463 bits (512 lanes, crashes); all five flatten settings fail at 512 lanes; problem is internal Vivado tool limitation, not an RTL bug
- **U55C bring-up blocked — U280 pivot:** professor email confirmed real U55C hardware runs `xilinx_u55c_gen3x16_xdma_base_2` shell but xclbin was built for `xilinx_u55c_gen3x16_xdma_3_202210_1`; XRT rejects bitstream due to shell mismatch; professor suggested Alveo U280 on SOL as alternative
- **U280 on SOL confirmed available:** same package (FSVH2892), same DSP count (9,024); memory differs: U55C has 16 GB HBM2 while U280 has 8 GB HBM2 + 32 GB DDR4; platform string: `xilinx_u280_gen3x16_xdma_1_202211_1`; board files at `/data/sse/u280_details/board_files`; accessed via `interactive -p general -q public -Lxilinx` or SLURM `--licenses=xilinx`; Vitis 2022.1 (`module load xilinx/vitis-2022.1`) + `source /data/sse/fpga/amd/fpga_env.sh`; v++ workflow documented for HLS kernel → .xo → .xclbin → real hardware execution
- **HLS kernels ready for U280 hardware execution:** `hls/rmsnorm/rmsnorm_core_hls.cpp`, `hls/softmax/softmax_core_hls.cpp`, `hls/silu/silu_core_hls.cpp` are standalone HLS C++ kernels compatible with v++ compile flow; no teammate dependency; host.cpp + golden traces from `model/export_fpga_vectors.py` complete the end-to-end hardware verification loop
- **Next steps:** (1) run one HLS kernel (RMSNorm recommended) on real U280 hardware via v++ + XRT host.cpp; (2) CPU baseline for speedup comparison; (3) performance estimates (tokens/sec at 64 and 512 lanes); (4) retarget synthesis to U280 (`xcu280-fsvh2892-2L-e`) for U280-specific utilization report
