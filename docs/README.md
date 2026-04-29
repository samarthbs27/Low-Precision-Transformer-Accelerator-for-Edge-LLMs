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
   checks, the current Phase 9/post-Phase-9 smoke reruns, and the current
   Vivado synthesis checkpoints on the hardened embedding/frontend/kernel-top
   hierarchy.

9. `real_inference_closure_plan.md`
   Concrete integration plan for replacing the current top-level runtime stubs
   with the true TinyLlama inference datapath through final RMSNorm, LM head,
   and argmax.

10. `Installation_guide_ug1468-alveo-u55c.pdf`
   AMD's board-installation guide for the Alveo U55C. This is a platform
   reference, not a design authority, but it is useful when we move from the
   normalized shell seam to real board bring-up.

11. `theory.md`
   Mathematical background for the transformer blocks, INT8 quantization, and
   hardware mapping concepts.

---

## Files

| File | Description |
|------|-------------|
| `sol_synthesis_guide.md` | SOL HPC cluster reference: file transfer, sbatch commands, monitoring, common failures, QOS limits |
| `design_decisions.txt` | Finalized implementation decisions for the TinyLlama U55C accelerator |
| `modules.md` | Physical module inventory for the FPGA implementation, including RTL/HLS split and interface plan |
| `implementation_checklist.md` | File-by-file coding plan for the TinyLlama implementation, including dependencies and verification order |
| `golden_trace_plan.md` | Real-model golden-trace export and verification policy |
| `u55c_bringup_checklist.md` | Shareable Linux/Vivado/Vitis bring-up checklist for the current Phase 9 runtime core |
| `real_inference_closure_plan.md` | Concrete plan to close the remaining gap from runtime harness to true token-generating TinyLlama inference |
| `Installation_guide_ug1468-alveo-u55c.pdf` | AMD Alveo U55C installation and platform bring-up guide |
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
- Current verified implementation frontier: Phases 0 through 9 plus the first
  post-Phase-9 real-inference closure slice, including the runtime decoder
  scaffold, runtime final-RMSNorm helper, and runtime LM-head tail
- Current Vivado synthesis frontier: full `tinyllama_u55c_kernel_top` (Phases
  0–9m, complete attention + FFN datapath) synthesized successfully on ASU SOL
  HPC cluster (job 51963647); LUTs 83.22% (1,084,885), Registers 9.52%,
  DSPs 11.35% (1,024/9,024), BRAMs 0%, 0 errors, 0 critical warnings at
  GEMM_LANES=64; WNS −60.988 ns @ 100 MHz (synthesis pessimism; hold timing
  clean); 512-lane synthesis blocked by Vivado 2022.1 NDup::dupNameType crash
  on ACC_BUS_W≥16K-bit packed structs (tool limitation, not RTL bug); design
  correctness at 512 lanes validated by simulation
- Hardware execution frontier: U55C blocked by shell mismatch
  (`xdma_base_2` vs `xdma_3`); U280 available on SOL via Vitis v++ flow
  (platform `xilinx_u280_gen3x16_xdma_1_202211_1`); HLS kernels
  (`rmsnorm_core_hls.cpp`, `softmax_core_hls.cpp`, `silu_core_hls.cpp`)
  ready for real U280 hardware execution

---

## Important Note

If this directory contains conflicting statements, `design_decisions.txt` is the authority to follow for implementation.
