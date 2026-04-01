# Docs

Design documentation for the Low-Precision Transformer Accelerator.

---

## Files

| File | Description |
|------|-------------|
| `block_diagram.md` | Full written documentation: FSM state diagram, per-block signal tables, tiling strategy, memory layout, and a step-by-step cycle trace |
| `block_diagram.png` | Exported PNG of the block diagram |
| `Milestone2.pdf` / `.png` | Project milestone spec sheet |

---

## Key Document: `block_diagram.md`

The main reference for understanding the hardware design. Contains:

- **FSM state diagram** — `IDLE → LOAD → COMPUTE → WRITE → DONE` with transition conditions and timing notes
- **Per-block signal tables** — every signal into and out of each block (name, width, description)
- **Tiling strategy** — pseudocode for the two-level loop (8 tiles × 64 cycles = 512 MAC cycles)
- **Memory layout** — Input BRAM, Weight BRAM (8 banks), Output Buffer addressing
- **Full cycle trace** — numbered step-by-step walkthrough of one complete run

## Viewing the Block Diagram

`block_diagram.png` is linked from the project root `README.md` and shows the full system architecture.  
For signal-level detail, refer to `block_diagram.md`.
