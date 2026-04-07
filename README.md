# FPGA Transformer Decoder Accelerator

> An on-chip INT8 accelerator for a full transformer decoder layer on the Xilinx Alveo U55C — QKV projections, scaled dot-product attention, FFN, softmax, layer norm, and residual connections, all running on FPGA fabric.

---

## Project Overview

We are building an **on-chip transformer decoder layer accelerator** in INT8 on an FPGA. The full decoder layer pipeline runs end-to-end on the FPGA between token input and layer output — no PCIe round-trips mid-inference.

**Decoder layer pipeline:**
```
Input X → Q/K/V Projections → Attention Scores (Q×Kᵀ/√d_k) → Softmax
        → Weighted Sum (×V) → Output Projection (×W_O)
        → Residual Add #1 → RMSNorm 1
        → FFN: gate_proj ∥ up_proj → SiLU(gate) × up → down_proj
        → Residual Add #2 → RMSNorm 2 → Output
```

**Two categories of hardware:**
- **Shared MAC array** — time-multiplexed across every matrix multiply step (Q, K, V, attention scores, weighted sum, W_O, FFN1, FFN2)
- **Dedicated blocks** — softmax unit, activation unit, two layer norms, two residual adders

All compute uses INT8 weights and activations with INT32 accumulation.

---

## Platform

| Property | Value |
|---|---|
| FPGA Board | Xilinx Alveo U55C (PCIe accelerator) |
| Host Interface | PCIe via XRT (x86 host) |
| Architecture | On-chip accelerator — full decoder layer on FPGA fabric |
| Compute Core | Parallel MAC array, output-stationary, time-multiplexed (8 lanes Phase 1; ~512 target) |
| Precision | INT8 × INT8 → INT32 accumulation |
| Test Dimensions | N=64, K=64 |

---

## Team

| Person | Role |
|---|---|
| **Satyarth** | Model, Quantization & Ground Truth — `model/model.py`, INT8 test vectors |
| **Rijul** | MAC Array & Dedicated Hardware — `mac_unit.sv`, `mac_array.sv`, softmax, activation, layer norm |
| **Samarth** | Dataflow, Tiling & Control — `control_fsm.sv`, `top.sv`, BRAM interface, residual adders |
| **Om** | U55C Integration & Host Interface — XRT kernel, PCIe host program |

---

## Repository Structure

```
Project/
  README.md
  docs/
    block_diagram.md               ← GEMM engine: signal tables, FSM diagram, tiling
    block_diagram.png              ← exported full transformer decoder layer diagram
    theory.md                      ← transformer theory, INT8 quantization, tiling explained
    Milestone2.pdf / .png          ← milestone 2 specification
    Milestone3_Progress_Report.docx← milestone 3 progress report
  rtl/
    control_fsm.sv                 ← 5-state tiling FSM (IDLE→LOAD→COMPUTE→WRITE→DONE)
    tb_control_fsm.sv              ← FSM testbench (7 checks, all passing)
    top.sv                         ← top-level integration: FSM + MAC array + BRAM stubs
    tb_top.sv                      ← integration testbench (4 checks, all passing)
    mac_unit.sv                    ← Rijul: combinational INT8×INT8→INT32 MAC unit
    mac_array.sv                   ← Rijul: 8-lane parallel MAC array with accumulators
    tb_mac_array.sv                ← Rijul: testbench (7 checks, all passing) ✅
    README.md                      ← module descriptions and simulation commands
  model/
    model.py                       ← Satyarth: FFN reference model + INT8 quantization
    gen_test_vectors.py            ← Satyarth: generates sim/x.txt, w.txt, expected.txt
  sim/
    x.txt / w.txt / expected.txt   ← test vectors for tb_mac_array (K=64, N=64)
    test_k16/ test_k64/            ← full FFN reference outputs
```

---

## System Architecture

![Full Transformer Decoder Layer](docs/block_diagram.png)

Full pipeline — signal tables, FSM state diagram, tiling strategy, and per-block descriptions: [docs/block_diagram.md](docs/block_diagram.md)

---

## 4-Week Execution Plan

### Week 1 — Foundations ✅
- INT8 quantized model + test data (Satyarth)
- MAC unit + 8-lane array in SystemVerilog (Rijul)
- Block diagram, FSM design, tiling plan (Samarth)
- U55C environment setup (Om)

### Week 2 — FPGA Compute Core ✅
- MAC array integrated with control FSM (`top.sv`)
- Tiled matrix-vector multiplication verified in simulation
- BRAM stubs for input, weight, and output buffers

### Week 3 — Transformer Pipeline
- Softmax unit in hardware (Rijul)
- SiLU activation + elementwise multiply for SwiGLU (Rijul)
- RMSNorm units ×2 (Rijul)
- RoPE unit for Q and K (Rijul)
- Residual adder units ×2 (Samarth)
- Attention dataflow scheduler on host (Samarth)
- Double-buffered weight streaming from HBM (Samarth)
- XRT kernel integration (Om)

### Week 4 — Integration & Evaluation
- End-to-end decoder layer simulation with real test vectors
- PCIe host-to-FPGA data pipeline
- CPU baseline (C++) for speedup comparison
- Performance measurement: latency, throughput, resource utilization

---

## Success Criteria

- Full transformer decoder layer executing on FPGA fabric
- All operations (linear + nonlinear) verified in simulation
- Host → FPGA → Host data pipeline via PCIe/XRT
- Measured latency and throughput vs CPU baseline

---

## Scope

**In scope:** One TinyLlama decoder layer — QKV projections (GQA), RoPE, scaled dot-product attention, softmax, SwiGLU FFN (SiLU), residual connections, RMSNorm. Reused 22× by host loop.

**Out of scope:** Multi-layer simultaneous execution, KV cache, multi-head in Phase 1 (single head for test), INT4 weights, 2D systolic array.
