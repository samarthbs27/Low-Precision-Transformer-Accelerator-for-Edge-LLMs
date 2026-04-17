# Real Inference Closure Plan

This document maps the remaining work from the current Phase 9 runtime harness
to a **true end-to-end TinyLlama inference path**.

It exists because the repo now has:

- strong leaf-block verification
- strong runtime-control verification
- a verified normalized shell DMA seam

but it does **not** yet have a top-level runtime that produces real TinyLlama
token IDs from the on-FPGA inference datapath.

---

## Current State

Today the runtime core in
[tinyllama_u55c_kernel_top.sv](../rtl/top/tinyllama_u55c_kernel_top.sv)
is structurally integrated enough for:

- AXI-Lite launch and status
- PC30 command/status traffic
- prompt token fetch
- generated-token writeback
- runtime stop conditions
- abort/relaunch acceptance
- shell-side DMA seam verification

What it is **not** doing yet is true token inference.

The current top-level still contains deterministic LM/token stub logic:

- `block_done` is synthesized from `block_start`
- LM-head completion is synthesized by a small local state machine
- emitted token IDs are generated as `1000 + generated_token_count`

So the current top-level is a **runtime harness**, not the final inference
datapath.

One important update: the first post-Phase-9 real-inference closure slice is
now in place. The runtime core now includes
[runtime_embedding_frontend.sv](../rtl/top/runtime_embedding_frontend.sv),
which fetches embedding-output scale metadata, issues real embedding-row DMA
requests, and emits real INT8 embedding tiles during prefill. The remaining gap
is the integrated decoder/final-RMS/LM-head/argmax path.

---

## Design Target

The target architecture is already frozen in the main design docs:

- after layer 21, the FPGA applies final RMSNorm
- the FPGA runs LM-head tiled GEMM
- the FPGA computes greedy argmax
- the FPGA emits real token IDs

Relevant source-of-truth docs:

- [design_decisions.txt](design_decisions.txt)
- [modules.md](modules.md)
- [implementation_checklist.md](implementation_checklist.md)

This plan is only about closing the remaining top-level integration gap.

---

## What Is Already Implemented But Not Yet Fully Integrated

These blocks already exist and have local verification:

- [embedding_lookup.sv](../rtl/compute/embedding_lookup.sv)
- [embedding_quantizer.sv](../rtl/compute/embedding_quantizer.sv)
- [rmsnorm_wrapper.sv](../rtl/nonlinear/rmsnorm_wrapper.sv)
- [shared_gemm_engine.sv](../rtl/compute/shared_gemm_engine.sv)
- [gemm_op_scheduler.sv](../rtl/compute/gemm_op_scheduler.sv)
- [gemm_operand_router.sv](../rtl/compute/gemm_operand_router.sv)
- [gemm_result_router.sv](../rtl/compute/gemm_result_router.sv)
- [rope_unit.sv](../rtl/compute/rope_unit.sv)
- [gqa_router.sv](../rtl/compute/gqa_router.sv)
- [causal_mask_unit.sv](../rtl/compute/causal_mask_unit.sv)
- [softmax_wrapper.sv](../rtl/nonlinear/softmax_wrapper.sv)
- [silu_wrapper.sv](../rtl/nonlinear/silu_wrapper.sv)
- [residual_add.sv](../rtl/compute/residual_add.sv)
- [elementwise_mul.sv](../rtl/compute/elementwise_mul.sv)
- [lm_head_controller.sv](../rtl/compute/lm_head_controller.sv)
- [argmax_reduction.sv](../rtl/compute/argmax_reduction.sv)

The gap is not that these modules do not exist. The gap is that the top-level
runtime does not yet route real data through them as one end-to-end inference
path.

---

## Missing Integration Work

### 1. Real Embedding Ingress

The runtime must convert incoming token IDs into real activation tiles:

1. fetch embedding rows from HBM with
   [embedding_lookup.sv](../rtl/compute/embedding_lookup.sv)
2. quantize them into activation tiles with
   [embedding_quantizer.sv](../rtl/compute/embedding_quantizer.sv)
3. pass those tiles into the decoder-layer datapath

This is required for both:

- prompt prefill
- decode-time next-token embedding

### 2. A Real Integrated Decoder Datapath Module

The repo currently has the leaves and the schedule logic, but not one single
runtime-integrated decoder datapath module that ties them together into:

`input hidden -> 22-layer datapath -> final hidden`

That integration module should own the real dataflow across:

- GEMM path
- RoPE
- KV-cache read/write
- causal mask
- softmax
- weighted sum
- residual path
- RMSNorm path
- FFN path

This should be a dedicated integration module rather than stuffing all of the
connections directly into
[tinyllama_u55c_kernel_top.sv](../rtl/top/tinyllama_u55c_kernel_top.sv).

### 3. Real Layer-Block Completion

The current runtime uses a stubbed block-done path.

That must be replaced with real block completion from the integrated decoder
datapath so:

- `layer_controller.sv` advances only when the active block actually finishes
- the runtime no longer assumes "one cycle per block"

### 4. Final RMSNorm After Layer 21

The architecture requires:

- final hidden-state selection
- final gamma fetch
- final RMSNorm application on FPGA

This final RMSNorm must happen before LM head.

For a normal prompt launch, the first emitted token after prefill must come from
the **prompt tail position**, not an arbitrary internal placeholder.

### 5. Real LM-Head Path

[lm_head_controller.sv](../rtl/compute/lm_head_controller.sv) already owns the
outer vocabulary loop. It now needs to be wired into the top-level runtime with
the real inputs:

- captured final hidden-state tile
- hidden-state scale metadata
- real LM-head weight tiles from HBM
- real scheduler completion and real logits return

### 6. Real Argmax Path

[argmax_reduction.sv](../rtl/compute/argmax_reduction.sv) must replace the
synthetic token path entirely.

That means:

- real vocab-tile partial logits arrive from LM head
- argmax reduces across the full `VOCAB_SIZE`
- the emitted token ID becomes the real runtime token stream

### 7. Reconnect Runtime Consumers To Real Tokens

Once the token stream is real, the following blocks must consume it unchanged:

- [prefill_decode_controller.sv](../rtl/control/prefill_decode_controller.sv)
- [generated_token_writer.sv](../rtl/memory/generated_token_writer.sv)
- [stop_condition_unit.sv](../rtl/control/stop_condition_unit.sv)

At that point the runtime loop becomes genuine inference rather than a control
simulation around synthetic token IDs.

---

## Recommended Implementation Order

The safest order is:

1. continue from the completed embedding-ingress slice by introducing a
   dedicated integrated decoder datapath module
2. replace synthetic `block_done` with real decoder-block completion
3. add final RMSNorm at the end of the 22-layer pass
4. wire real LM-head control and logits flow
5. wire real argmax reduction and token emission
6. remove the synthetic LM/token stub logic from
   [tinyllama_u55c_kernel_top.sv](../rtl/top/tinyllama_u55c_kernel_top.sv)
7. add a true token-generating top-level testbench

This order keeps the control plane stable while we progressively replace the
remaining synthetic datapath seams.

---

## Verification Plan For True Inference

### Before Hardware

We should add new top-level simulations that use exported real TinyLlama
fixtures and compare actual emitted token IDs to the Python reference.

Minimum simulation milestones:

1. **Prefill-only, one emitted token**
   - fixed prompt
   - verify first emitted token after prefill matches reference

2. **Prefill + 1 decode step**
   - fixed prompt
   - verify the first two emitted token IDs match reference

3. **Prefill + 2 decode steps**
   - fixed prompt
   - verify the first three emitted token IDs match reference

4. **Decode-only from prepared KV state**
   - verify token generation from an existing cache state

### After Synthesis / Platform Closure

Once the raw U55C wrapper and vendor flow are in place:

1. load a fixed prompt from the host
2. run `max_new_tokens = 1`
3. compare FPGA-emitted token ID against Python reference
4. then extend to `2`, `4`, and a few fixed prompts

The first meaningful hardware correctness milestone is:

- **one fixed prompt**
- **one generated token**
- **exact token-ID match with the TinyLlama Python reference**

---

## Exit Criteria For Real Inference Closure

We should treat the runtime as truly end to end only when all of these are true:

- the top-level no longer fabricates LM-head completion or token IDs
- prompt prefill feeds real embedding tiles into the decoder datapath
- the 22-layer path produces a real final hidden state
- final RMSNorm runs on FPGA
- LM head runs on FPGA
- argmax runs on FPGA
- generated token IDs match the TinyLlama reference for at least one fixed
  prompt in simulation
- generated token IDs match the TinyLlama reference for at least one fixed
  prompt on hardware

---

## Recommended Next Milestone

The next most valuable milestone is not "full long-text generation."

The next most valuable milestone is:

- **one fixed prompt**
- **prefill plus one generated token**
- **exact token match against the Python reference**

That is the smallest end-to-end proof that the runtime is performing real
TinyLlama inference instead of only structural control-plane verification.
