# Golden Trace Export Plan

This document defines how real TinyLlama reference traces are used to harden
the FPGA implementation.

It is a verification plan, not a replacement for `design_decisions.txt` or
`modules.md`.

---

## 1. Purpose

Golden traces are exported from the Python TinyLlama reference path and used to
check that RTL and HLS blocks match real model behavior on selected tiles,
heads, and decode states.

The goal is not to replay the entire model in every unit test.

The goal is to:

- keep control and plumbing tests simple
- use real model-derived data where arithmetic or quantization can drift
- add trace-backed integration tests before full end-to-end bring-up

---

## 2. What Golden Traces Are And Are Not

Golden traces are:

- deterministic exports from the TinyLlama software reference
- small, selected slices of real model computation
- tied to fixed layer, token, head, and tile coordinates
- used as expected input/output pairs for RTL and HLS verification

Golden traces are not:

- a requirement for every module in every phase
- full 22-layer, 2048-token waveform dumps
- a substitute for directed smoke tests on control or handshake logic

---

## 3. Phase Policy

### 3.1 Phase 0 To Phase 2

Golden traces are **not required** for:

- package files
- FIFOs and skid buffers
- AXI-Lite and register logic
- controllers and stop logic
- DMA handshake logic
- HBM routing
- buffer banks
- address generation

Why:

- these phases are dominated by protocol, storage, and control behavior
- directed tests are simpler, faster, and easier to debug
- real model data adds little value for these blocks

### 3.2 Phase 3

Golden traces are **recommended but not mandatory** for:

- `shared_gemm_engine.sv`
- `requantize_unit.sv`
- `gemm_operand_router.sv`
- `gemm_result_router.sv`

Why:

- Phase 3 is the first place where real quantized arithmetic becomes important
- but the modules are still local enough that directed arithmetic tests remain
  valuable and sufficient for first hardening

### 3.3 Phase 4 To Phase 6

Golden traces are **required** for math and dataflow blocks:

- `rope_unit.sv`
- `causal_mask_unit.sv`
- `rmsnorm_core_hls.cpp`
- `softmax_core_hls.cpp`
- `silu_core_hls.cpp`
- `residual_add.sv`
- `elementwise_mul.sv`
- `lm_head_controller.sv`
- `argmax_reduction.sv`

Why:

- these blocks depend directly on real TinyLlama tensor values, positions,
  masking, and quantization boundaries
- directed synthetic vectors alone are too weak to catch model-specific drift

### 3.4 Phase 7 To Phase 9

Golden traces are **mandatory** for integration:

- one decoder-layer case
- one prefill case
- one decode-step case with a real KV-cache prefix
- one LM-head tiled case
- one top-level smoke case with expected generated-token outputs

Why:

- this is where individually-correct blocks can still disagree on tags, tile
  ordering, quantization placement, or cache semantics

---

## 4. Output Location

All generated trace artifacts live under:

```text
sim/golden_traces/
```

The generator must never write trace artifacts into the repo root.

Planned layout:

```text
sim/golden_traces/
  manifest.json
  phase3/
  phase4/
  phase5/
  phase6/
  phase7/
  phase8/
```

---

## 5. Canonical Export Format

The canonical exported case format is:

- one `manifest.json`
- one `.npz` file per case

`manifest.json` records:

- trace format version
- model identifier
- source commit or source script version when available
- list of case IDs
- phase, block, and path for each case
- metadata fields required to interpret the case

Each case `.npz` stores numeric arrays only.

The manifest carries the descriptive metadata so the archive format stays simple.

---

## 6. Standard Case Metadata

Each manifest entry must include the fields that apply to that case:

- `case_id`
- `phase`
- `block`
- `path`
- `runtime_mode` (`prefill` or `decode` when applicable)
- `layer_id`
- `token_start`
- `token_count`
- `q_head_id`
- `kv_head_id`
- `m_tile_idx`
- `n_tile_idx`
- `k_tile_idx`
- `dtype_summary`
- `shape_summary`

Unused fields are omitted, not filled with dummy values.

---

## 7. Standard Array Payloads

The array names inside each `.npz` are fixed by block family.

### 7.1 GEMM Tile Cases

- `act`
- `wt`
- `acc_expected`

Optional:

- `acc_init`
- `act_steps`
- `wt_steps`
- `acc_expected_lane`
- `active_lane_count`

When present, the `*_steps` arrays use the fixed row-major `M_TILE x N_TILE`
lane packing defined in `modules.md`.

### 7.2 Requantization Cases

- `acc`
- `scale`
- `out_expected`

### 7.3 RoPE Cases

- `q_in`
- `k_in`
- `q_out_expected`
- `k_out_expected`
- `cos`
- `sin`

### 7.4 Causal Mask Cases

- `score_in`
- `score_out_expected`
- `query_pos`
- `key_pos`

### 7.5 RMSNorm Cases

- `x`
- `gamma`
- `y_expected`

### 7.6 Softmax Cases

- `score_in`
- `prob_expected`

### 7.7 SiLU Cases

- `x_in`
- `y_expected`

### 7.8 Residual / Elementwise Cases

- `lhs`
- `rhs`
- `out_expected`

### 7.9 LM-Head Cases

- `act`
- `wt`
- `logits_expected`
- `argmax_expected`

### 7.10 Decoder-Layer Cases

- `layer_in`
- `layer_out_expected`

Optional intermediate arrays may be included for debug:

- `q_expected`
- `k_expected`
- `v_expected`
- `score_expected`
- `prob_expected`
- `ffn_expected`

---

## 8. Consumption Rules

Directed tests remain the first gate for new modules.

When a block reaches trace-backed verification:

- HLS C++ tests may consume canonical `.npz` cases through a small helper path
- RTL testbenches should consume trace-derived fixtures, not parse `.npz`
  directly inside SystemVerilog

For RTL, the preferred pattern is:

- export canonical `.npz`
- derive a small text, `.memh`, or include-format fixture only for the selected case
- keep the derived fixture under `sim/golden_traces/`

The current Phase 3 to Phase 5 implementation uses packed `.memh` fixtures for
RTL consumption.

This keeps the Python export path authoritative while avoiding heavy file
parsing logic inside RTL testbenches.

---

## 9. Case Selection Rules

The first trace set must favor coverage over volume.

Required early cases:

- one full-width GEMM tile with no partial lanes
- one edge tile with partial `elem_count`
- one prefill RoPE case
- one decode RoPE case
- one prefill causal-mask case
- one decode causal-mask case
- one RMSNorm case
- one softmax case with nontrivial distribution
- one SiLU case with positive and negative values
- one LM-head vocab-tile case
- one decoder-layer case at layer 0
- one decoder-layer case at layer 21

Do not start by exporting the full model state for every token and every layer.

---

## 10. Generator Contract

The planned generator entry point is:

- `model/export_fpga_vectors.py`

The generator must support:

- fixed output root under `sim/golden_traces/`
- phase-scoped export selection
- case-scoped export selection
- deterministic file naming

Expected first-pass command shape:

```text
python model/export_fpga_vectors.py --phase phase4 --output-dir sim/golden_traces
```

Current implemented export scope:

- Phase 3:
  - prefill and decode GEMM cases for layer 0
  - prefill and decode requantization cases for layer 0
  - generated packed `.memh` fixtures for:
    - `tb_shared_gemm_engine.sv`
    - `tb_requantize_unit.sv`
- Phase 4:
  - one prefill RoPE case
  - one decode RoPE case
  - one prefill causal-mask case
  - one decode causal-mask case
  - generated packed `.memh` fixtures for:
    - `tb_rope_unit.sv`
    - `tb_causal_mask_unit.sv`
  - generated RoPE Q16.16 ROM contents under:
    - `rtl/compute/rope_cos_rom.memh`
    - `rtl/compute/rope_sin_rom.memh`
- Phase 5:
  - prefill and decode RMSNorm cases
  - prefill and decode softmax cases
  - prefill and decode SiLU cases
  - generated packed `.memh` fixtures for:
    - `tb_rmsnorm_wrapper.sv`
    - `tb_softmax_wrapper.sv`
    - `tb_silu_wrapper.sv`

Expected later command shape:

```text
python model/export_fpga_vectors.py --case-set integration --output-dir sim/golden_traces
```

The exact CLI can evolve, but the output location and case discipline above are
fixed.

---

## 11. Acceptance Gates

The project should use the following gates:

- Phase 0 to Phase 2:
  directed smoke tests only
- Phase 3:
  directed smoke tests mandatory, trace-backed arithmetic tests encouraged
- Phase 4 to Phase 6:
  every math/dataflow block must have at least one trace-backed test
- Phase 7:
  decoder-layer integration must pass at least one exported layer trace
- Phase 8:
  runtime integration must pass at least one exported prefill+decode trace
- Phase 9:
  debug capture and top-level smoke must line up with exported expected values

---

## 12. Answer To The Phase 0 Question

No, golden traces are not necessary from Phase 0.

Phase 0 exists to freeze:

- constants
- typedefs
- bus layouts
- common helpers

Those are better validated by:

- compilation
- type import checks
- small deterministic smoke tests

Golden traces become valuable once the hardware is performing real model math
or model-ordered dataflow. That starts becoming important in Phase 3 and becomes
mandatory from Phase 4 onward.
