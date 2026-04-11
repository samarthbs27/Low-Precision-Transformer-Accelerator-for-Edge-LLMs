# TinyLlama U55C FPGA Inference Accelerator

> A mixed-precision FPGA implementation target for TinyLlama 1.1B on the Xilinx Alveo U55C, with prompt prefill, autoregressive decode, HBM-resident KV cache, and a reused INT8 decoder-layer engine.

---

## Project Overview

This repository is centered on implementing TinyLlama inference on the U55C.
The current architecture is no longer a single decoder-layer demo or a host-loop
prototype. The intended runtime is:

- host tokenizes the prompt and launches the FPGA
- FPGA performs embedding lookup, prefill, decode, KV-cache updates, final RMSNorm, LM head, and greedy argmax
- FPGA emits generated token ids until EOS or `max_new_tokens`
- host reads token ids back and detokenizes them for display

The implementation strategy is reuse-heavy:

- one TinyLlama decoder-layer engine reused for all 22 layers
- one shared INT8 GEMM engine reused across Q, K, V, attention scores, weighted sum, O, gate, up, down, and LM head
- HBM used as the backing store for weights, KV cache, and debug buffers

---

## Finalized Architecture

| Item | Decision |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Layers | `22` reused decoder layers |
| Attention | `32` Q heads, `4` KV heads, `head_dim = 64` |
| Hidden size | `2048` |
| FFN size | `5632` |
| Runtime modes | prompt `prefill` + autoregressive `decode` |
| GEMM precision | `INT8 x INT8 -> INT32` |
| Higher-precision blocks | RMSNorm, RoPE, softmax, final RMSNorm |
| KV cache | symmetric `INT8` in HBM, per-layer/per-KV-head scales |
| Token selection | FPGA-side greedy argmax |
| U55C memory model | `16 GB` HBM, `32` pseudo-channels |
| Production compute width | `512` INT8 lanes |

The authoritative implementation spec is [docs/design_decisions.txt](docs/design_decisions.txt).

---

## Repository Structure

```text
Project/
  README.md
  docs/
    README.md
    design_decisions.txt
    block_diagram.drawio
    block_diagram.md
    block_diagram.png
    theory.md
  model/
    README.md
    tinyllama.py
    tinyllama_gemm_int8.py
    model.py
    gen_test_vectors.py
  rtl/
    README.md
    control_fsm.sv
    top.sv
    mac_unit.sv
    mac_array.sv
    tb_control_fsm.sv
    tb_top.sv
    tb_mac_array.sv
  sim/
```

---

## Where To Read Next

- [docs/design_decisions.txt](docs/design_decisions.txt)
  Final implementation choices for quantization, HBM allocation, controller behavior, and runtime ownership.

- [docs/block_diagram.drawio](docs/block_diagram.drawio)
  Editable architecture diagram for the full TinyLlama accelerator.

- [docs/block_diagram.md](docs/block_diagram.md)
  Text explanation of the full post-Phase-1 architecture plus the legacy GEMM validation core.

- [model/README.md](model/README.md)
  Software reference path, TinyLlama NumPy inference, and GEMM-only INT8 bridge.

- [rtl/README.md](rtl/README.md)
  Existing RTL validation core and simulation entry points.

---

## Current Implementation Status

- `model/` contains the TinyLlama software reference and the GEMM-only INT8 analysis/generation bridge.
- `rtl/` contains the existing validation core for the shared GEMM engine and control FSM.
- `docs/` now describe the full TinyLlama prefill/decode accelerator that the project is building toward.

The current RTL is still validation infrastructure; the finalized system architecture is documented in `docs/`.
