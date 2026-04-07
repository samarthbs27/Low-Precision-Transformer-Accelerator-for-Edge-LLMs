# Docs

Design documentation for the FPGA Transformer Decoder Accelerator.

---

## Files

| File | Description |
|------|-------------|
| `block_diagram.md` | Full transformer pipeline signal tables + GEMM engine detail: FSM state diagram, per-block signal tables, tiling strategy, memory layout, cycle trace |
| `block_diagram.png` | Exported PNG of the full transformer decoder layer architecture |
| `design_decisions.txt` | All conservative design decisions made, open decisions needing team discussion, and implemented decisions |
| `theory.md` | Theory behind the hardware: transformer decoder layer math, INT8 quantization, tiling, output-stationary dataflow, full pipeline diagram |
| `ClassProject_RC19_Milestone3.pdf` | Milestone 3 specification sheet |
| `Milestone2.pdf` / `.png` | Milestone 2 specification sheet |

---

## Key Documents

### `block_diagram.png`
The main architecture reference. Shows the complete on-chip transformer decoder layer pipeline:
QKV projections → attention scores → softmax → weighted sum → W_O projection → residual add → layer norm → FFN → activation → FFN2 → residual add → layer norm 2.
Color-coded by block type (green = shared MAC array, red = dedicated new hardware, orange = shared compute resources).

### `theory.md`
Explains what the hardware computes and why it is designed the way it is. Covers the full decoder layer math, INT8 quantization, the tiling loop, output-stationary dataflow, and how the four team members' work fits together.

### `block_diagram.md`
Two-section document. Section 2 (full transformer pipeline): inter-block signal tables for all stages (input buffer, shared MAC array, re-quantize, softmax, SiLU+multiply, residual adders ×2, RMSNorm ×2, output buffer) using TinyLlama-correct block names (RMSNorm not LayerNorm, SwiGLU not 2-layer FFN). Section 1 (GEMM engine): FSM signal tables, BRAM addressing, tiling strategy, and cycle-by-cycle trace.

### `design_decisions.txt`
Three sections: (A) conservative decisions made to unblock RTL and documentation, (B) open decisions requiring team discussion, (C) implemented decisions. Updated when decisions change.
