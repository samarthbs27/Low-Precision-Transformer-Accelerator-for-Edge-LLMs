# FPGA Parallelism Tradeoffs

This note explains where the current TinyLlama U55C design is using FPGA
advantages aggressively, where it is intentionally conservative, and why those
choices are reasonable for the first full implementation.

This is a rationale document, not a source-of-truth spec.
If anything here conflicts with:

1. `design_decisions.txt`
2. `modules.md`
3. `implementation_checklist.md`

then those files win.

---

## Summary

Yes, the design is using FPGA strengths judiciously.

We are leaning into:
- wide lane-level compute
- banked on-chip storage
- HBM channel parallelism
- overlapped DMA and compute
- elastic streaming between major blocks

We are intentionally **not** leaning into:
- multiple GEMM engines
- inter-layer replication
- parallel multi-head score engines
- duplicated FFN projection engines

That is not because the design missed those ideas. It is because, on U55C for a
first concrete TinyLlama bring-up, those choices would likely increase area,
HBM pressure, control complexity, and integration risk faster than they would
increase useful throughput.

---

## Tradeoff Table

| Area | We are using now | We are intentionally not using now | Why this is a good tradeoff |
|---|---|---|---|
| Shared GEMM datapath | `512` parallel INT8 lanes in one reused GEMM engine | multiple parallel GEMM engines | A single wide engine captures the main FPGA arithmetic advantage without exploding area, routing, and HBM bandwidth demand. |
| Buffer architecture | `16` bank on-chip tile buffers with fixed lane-to-bank mapping | monolithic unbanked tile RAMs | Banking lets one full 512-lane vector move in parallel and supports ping/pong overlap cleanly. |
| Memory/compute overlap | mandatory ping/pong buffering and one-tile lookahead prefetch | fetch-then-compute serialization | This is exactly the kind of latency hiding FPGAs do well and keeps the single GEMM engine fed. |
| HBM usage | pseudo-channel partitioning across weights, KV, LM-head, host I/O, and debug | one giant undifferentiated memory pool | Fixed HBM ownership reduces arbitration chaos and makes bandwidth behavior more predictable. |
| Inter-stage timing isolation | elastic ready/valid boundaries with FIFOs and skid buffers | direct tightly coupled stage wiring everywhere | Elastic buffering is a practical FPGA advantage for timing closure and system stability. |
| Reuse across layers | one decoder-layer engine reused across all `22` layers | multiple physical decoder-layer replicas | Layer replication would be very expensive in area and memory traffic for TinyLlama 1.1B on U55C. |
| Attention head scheduling | grouped-query reuse with logical KV reuse and one head-group at a time | physically replicated K/V tensors or parallel score engines for many heads | GQA already reduces K/V cost; serializing head groups avoids overbuilding a bandwidth-heavy attention engine. |
| FFN scheduling | `gate`, `up`, and `down` all reuse the same GEMM engine | separate FFN GEMM engines or parallel gate/up engines | This keeps the design concrete and resource-disciplined while still matching the model correctly. |
| RoPE datapath | one `8 x 64` rotary slice processed across 512 lanes, with ROM broadcast structure | full-sequence or multi-slice RoPE engines in parallel | The chunked slice matches the datapath width exactly and avoids a wider, more complex rotary fabric. |
| Score-mask datapath | one `8 x 64` score chunk processed across 512 lanes | full `16 x 64` score tile masking in one oversized structure | Chunking keeps the mask unit aligned with the existing packed-bus width and avoids a second score-width contract. |
| Score softmax path | one score chunk at a time | many-row or many-head softmax engines in parallel | Softmax is numerically sensitive and bandwidth-heavy; keeping it narrow reduces implementation risk. |
| LM-head path | tiled vocab processing with overlap between reduction and next-tile fetch | full-vocab parallel projection | Full-vocab parallelism is unrealistic; tiling is the right FPGA-friendly way to scale this block. |
| Debug capture | dedicated debug FIFO and non-stalling debug policy | fully synchronous debug capture in the critical datapath | This protects performance and timing while still giving us visibility. |

---

## Where We Are Strongly Exploiting FPGA Advantages

| Advantage | Current design choice | Why it matters |
|---|---|---|
| Fine-grain data parallelism | `512`-lane GEMM engine | This is the main throughput engine of the design. |
| Spatial banking | `16` fixed tile-buffer banks | Enables wide vector movement and avoids single-memory bottlenecks. |
| Pipeline overlap | ping/pong buffers + one-tile prefetch | Lets memory and compute proceed concurrently. |
| Streaming composition | FIFO-based ready/valid module boundaries | Makes a large design easier to integrate and time. |
| HBM concurrency | fixed pseudo-channel groups | Uses the U55C memory system intentionally instead of incidentally. |
| Fixed-function datapaths | dedicated leaves for RoPE, causal mask, requantization, routing | Good FPGA designs often win by specializing datapaths instead of over-generalizing them. |

---

## Where We Are Deliberately Conservative

| Not used in first implementation | Why we are avoiding it for now |
|---|---|
| multiple decoder-layer engines | Too much duplication of buffers, control, and HBM traffic for the likely benefit. |
| multiple shared GEMM engines | Would quickly turn into a bandwidth and integration problem, not just a compute upgrade. |
| inter-layer parallelism | TinyLlama layers are sequentially dependent; overlapping them physically would add complexity with little practical gain. |
| multi-head score parallelism | Score/softmax/KV traffic is heavy enough that this could become memory-bound before it becomes compute-bound. |
| parallel `gate` and `up` engines | Nice in theory, but costly for a first concrete FFN path. |
| runtime-programmable tile sizes | Flexibility here would make the first bitstream harder to verify and optimize. |
| broad NoC-style crossbar routing | Fixed-function routers are simpler, cheaper, and easier to reason about for this architecture. |

---

## Revisit Later

These are the places where more parallelism could be worth reconsidering after
we have real synthesis and runtime numbers:

| Candidate upgrade | Revisit trigger |
|---|---|
| `HEAD_GROUP_PAR > 1` | If attention becomes the dominant throughput bottleneck and HBM still has headroom. |
| partial duplication of the GEMM path | If the single shared GEMM engine is clearly the bottleneck and memory can sustain more issue rate. |
| more aggressive LM-head overlap | If LM-head latency dominates decode after the rest of the decoder is stable. |
| deeper or more specialized prefetching | If real HBM traces show the engine stalling on memory bubbles. |
| more broadcast-aware ROM / lookup structures | If synthesis reports show RoPE ROM fanout or muxing becoming expensive. |

---

## Bottom Line

The current design is not trying to use **all possible** FPGA parallelism.
It is trying to use the **highest-value** FPGA parallelism first:

- wide arithmetic
- banked memory
- overlapped movement and compute
- specialized streaming blocks

That is the right strategy for a concrete first implementation on U55C.
