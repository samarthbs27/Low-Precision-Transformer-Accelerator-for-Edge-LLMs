# Docs

Design documentation for the TinyLlama U55C FPGA inference accelerator.

---

## Read First

1. `design_decisions.txt`
   The implementation spec. This is the source of truth for model dimensions,
   quantization policy, control ownership, HBM allocation, and FPGA/host behavior.

2. `modules.md`
   The hardware implementation inventory. This lists every required module,
   whether it is RTL or HLS, what it does, what it connects to, and what
   parallelism it uses.

3. `implementation_checklist.md`
   The file-by-file execution plan. This lists the planned source tree,
   implementation order, dependencies, stub-first strategy, and first
   verification target for each RTL/HLS/support file.

4. `golden_trace_plan.md`
   The trace-backed verification policy. This defines when real TinyLlama
   reference traces are required, where they are exported, and how they are
   consumed by RTL/HLS tests.

5. `block_diagram.drawio`
   The editable system diagram. This shows the post-Phase-1 TinyLlama architecture:
   host launch, on-chip prefill/decode controller, reused decoder-layer engine,
   HBM weights + KV cache, final RMSNorm, LM head, and greedy argmax.

6. `block_diagram.md`
   The textual companion to the diagram. Section 2 explains the full TinyLlama
   prefill/decode architecture. Section 1 documents the existing GEMM validation core.

7. `parallelism_tradeoffs.md`
   Design rationale for where the current architecture is using FPGA
   parallelism aggressively and where it is intentionally conservative.

8. `u55c_bringup_checklist.md`
   Practical teammate-facing bring-up guide for Linux, XRT/U55C platform
   checks, the current Phase 9 smoke reruns, and the first concrete Vivado
   synthesis pass on the runtime core plus shell wrapper.

9. `real_inference_closure_plan.md`
   Concrete integration plan for replacing the current top-level runtime stubs
   with the true TinyLlama inference datapath through final RMSNorm, LM head,
   and argmax.

10. `theory.md`
   Mathematical background for the transformer blocks, INT8 quantization, and
   hardware mapping concepts.

---

## Files

| File | Description |
|------|-------------|
| `design_decisions.txt` | Finalized implementation decisions for the TinyLlama U55C accelerator |
| `modules.md` | Physical module inventory for the FPGA implementation, including RTL/HLS split and interface plan |
| `implementation_checklist.md` | File-by-file coding plan for the TinyLlama implementation, including dependencies and verification order |
| `golden_trace_plan.md` | Real-model golden-trace export and verification policy |
| `u55c_bringup_checklist.md` | Shareable Linux/Vivado/Vitis bring-up checklist for the current Phase 9 runtime core |
| `real_inference_closure_plan.md` | Concrete plan to close the remaining gap from runtime harness to true token-generating TinyLlama inference |
| `block_diagram.drawio` | Editable system architecture diagram; source of truth for the visual dataflow |
| `block_diagram.md` | System-level architecture explanation plus legacy GEMM-engine details |
| `parallelism_tradeoffs.md` | Design-rationale note on FPGA parallelism choices and deliberate first-pass limits |
| `block_diagram.png` | Exported image artifact of the block diagram; may lag behind the `.drawio` source |
| `theory.md` | Transformer and quantization theory reference |
| `ClassProject_RC19_Milestone3.pdf` | Milestone 3 specification sheet |
| `Milestone2.pdf` / `Milestone2.png` | Milestone 2 materials |

---

## Current Architecture Summary

- Model target: TinyLlama 1.1B
- Runtime: prompt prefill followed by autoregressive decode
- Control split: host launches and reads back results; FPGA executes embedding lookup, 22-layer inference, KV-cache management, final RMSNorm, LM head, and greedy argmax
- Quantization: INT8 GEMMs with INT32 accumulation; RMSNorm, RoPE, softmax, and final RMSNorm stay higher precision
- Memory model: U55C HBM stores weights, KV cache, and debug buffers; BRAM/URAM hold active tiles and partial sums
- Current verified implementation frontier: Phases 0 through 9, including
  runtime acceptance and the first shell-facing wrapper around the normalized
  DMA boundary

---

## Important Note

If this directory contains conflicting statements, `design_decisions.txt` is the authority to follow for implementation.
