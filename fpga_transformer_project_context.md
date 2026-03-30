# FPGA Transformer Accelerator Project Context

## 🧠 Project Overview

We are building a **low-precision hardware accelerator for Transformer-based inference** on an FPGA.

Originally targeted platform:
- Xilinx Kria KV260 (Zynq UltraScale+)

Updated platform:
- **Xilinx Alveo U55C (PCIe accelerator card)**

This changes the system architecture from:
- Embedded SoC (ARM + FPGA)

To:
- **Host CPU + FPGA accelerator over PCIe**

---

## 🎯 Final Project Direction

We are **NOT implementing full TinyLlama**.

We are implementing:

> A **low-precision Transformer block accelerator**, focusing on the **feed-forward network (FFN)** and linear layers.

### Core Computation

We target:

y = W2 * activation(W1 * x)

Where:
- W1, W2 = weight matrices
- x = input vector
- activation = ReLU (Phase 1), SiLU (optional Phase 2)

---

## ❗ Key Clarifications

### TinyLlama
TinyLlama is:
- Used as a **reference architecture**
- Used to derive **realistic dimensions and workloads**

It is NOT:
- Fully deployed
- Fully executed end-to-end

---

### “Sparse” in Title
We are **NOT implementing sparsity**.

Decision:
- Remove “Sparse” from project title
- Focus on **low-precision + memory efficiency**

---

## 🏗 System Architecture (U55C)

### High-Level Flow

Host CPU
- Tokenization
- Control flow
- Baseline computation
        ↓
PCIe
        ↓
U55C FPGA
- Input buffers
- Tile scheduler
- MAC array (core compute)
- Output buffers
        ↓
Host CPU
- Validation
- Output processing

---

## ⚙️ Core Hardware Design

### Chosen Architecture

We are using:

> **1D Parallel MAC Array (Output-Stationary)**

NOT:
- Full 2D systolic array (too complex for 4 weeks)

---

### Why 1D MAC Array?

- Matches Transformer inference (matrix-vector, batch=1)
- Easier to implement and debug
- Works well with tiling
- Less routing complexity

---

### MAC Operation

Each lane computes:

acc += W[j,k] * x[k]

---

### MAC Array Design

- INT8 × INT8 multiply
- INT32 accumulation
- 8–16 parallel lanes
- Broadcast x[k] to all lanes
- Each lane reads its own weight

---

### Tiling Strategy

- Compute outputs in chunks (tiles)
- Example: 8 or 16 outputs at a time
- Loop over input dimension

---

## 🧠 What We Are Actually Building

We are NOT building:
- Full Transformer
- Attention (initially)
- KV cache (initially)
- Full LLM inference

We ARE building:

> A **high-performance dot-product / matrix multiplication engine**

---

## 🔢 Quantization

Phase 1:
- INT8 weights
- INT8 activations
- INT32 accumulation

Phase 2 (optional):
- INT4 weights
- Packed weight format

---

## 📅 4-Week Execution Plan

---

### Week 1 — Foundations

Goal:
- Working math model
- Verified MAC unit
- Basic host-FPGA communication

Deliverables:
- Python FFN model
- INT8 quantized data
- 1 MAC unit (Verilog)
- 8-lane MAC array (simulation)
- Initial U55C host setup

---

### Week 2 — FPGA Compute Integration

Goal:
- Run real compute on FPGA

Deliverables:
- MAC array integrated with control FSM
- Tiled matrix multiplication
- BRAM buffers for inputs/weights
- FPGA execution of linear layer
- Correctness vs Python

---

### Week 3 — Memory + Performance

Goal:
- Improve throughput

Deliverables:
- Tiled weight streaming
- Double buffering (ping/pong)
- Performance measurement
- Resource utilization

---

### Week 4 — Optimization + Finalization

Goal:
- Polish + evaluate

Deliverables:
- Optional SiLU LUT
- Optional INT4 support
- Performance comparison vs CPU
- Bottleneck analysis
- Final demo + report

---

## 👥 Team Responsibilities (Week 1)

---

### Person 1 — Model + Quantization

- Build FFN in Python
- Quantize to INT8
- Generate test data
- Export:
  - x
  - W1, W2
  - expected outputs

---

### Person 2 — MAC Unit

- Implement INT8 MAC
- Expand to 8-lane array
- Write testbench
- Verify against Python

---

### Person 3 — Dataflow + Control

- Design tiling strategy
- Define dataflow
- Write control FSM (loop over k)
- Define memory layout

---

### Person 4 — U55C Integration

- Set up U55C environment
- Write host program
- Send/receive data
- Verify communication (even dummy)

---

## 🚨 Key Constraints

### Time: 4 weeks

Therefore:
- Must prioritize working system early
- Avoid over-engineering
- Focus on vertical slice

---

## ❌ What We Are NOT Doing (Phase 1)

- Full TinyLlama
- Multi-layer Transformer
- Full attention mechanism
- KV cache scaling
- Perfect quantization
- Complex systolic arrays

---

## ✅ What Defines Success

Minimum success:

- Working INT8 MAC array on FPGA
- Host → FPGA → Host pipeline
- Accelerated linear layer
- Measured speedup vs CPU

---

## 🧠 Mental Model

Core idea:

> We are building a machine that computes dot products faster than a CPU.

Everything else (Transformer, TinyLlama, AI) is context.

---

## 📚 Learning Priorities

Focus on understanding:

1. Dot product
2. Matrix multiplication
3. Fixed-point arithmetic (INT8)
4. Parallel hardware (MAC arrays)
5. Basic FPGA memory (BRAM)
6. Host–FPGA communication

NOT required:
- Deep ML theory
- Training
- Advanced optimization

---

## 🏁 Final Conclusion

### Architecture Choice
- 1D Parallel MAC Array (output-stationary)

### Platform
- U55C (PCIe accelerator)

### Scope
- Transformer block components (FFN first)

### Strategy
- Build working system early
- Add complexity later

---

## 🔥 Core Project Statement

> Design and implement a low-precision FPGA accelerator for Transformer block computations using a parallel MAC array architecture, integrated with host control via PCIe, and demonstrate performance gains over CPU-based execution.
