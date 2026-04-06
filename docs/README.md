# Docs

Design documentation for the FPGA Transformer Decoder Accelerator.

---

## Files

| File | Description |
|------|-------------|
| `block_diagram.md` | GEMM engine detail: FSM state diagram, per-block signal tables, tiling strategy, memory layout, cycle trace |
| `block_diagram.png` | Exported PNG of the GEMM engine block diagram |
| `block_diagram.png` | Exported PNG of the full transformer decoder layer |
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
Low-level reference for the GEMM compute engine (the shared MAC array core). Contains signal tables, FSM transitions, BRAM addressing formulas, and a cycle-by-cycle trace. Still accurate — the GEMM engine is unchanged; it is now time-multiplexed across all linear ops in the decoder layer.
