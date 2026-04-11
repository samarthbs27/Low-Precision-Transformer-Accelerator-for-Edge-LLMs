# Dataflow & Control Block Diagram
## Samarth – DATAFLOW, TILING, AND CONTROL

> Visual diagram: open `block_diagram.drawio` with the Draw.io Integration VS Code extension.

> **Scope note (2026-04-10):** This document now covers two levels:
> - **Section 1 (GEMM Engine)** — the existing shared MAC array and tiling FSM validation core. This is still the implemented low-level compute foundation.
> - **Section 2 (Target TinyLlama System)** — the post-Phase-1 U55C architecture: prefill + decode, KV cache, one reused decoder-layer engine, and FPGA-side final token selection.

---

## Section 2 - Target TinyLlama U55C Architecture

### System Overview

```
Host CPU / XRT
    |
    | prompt token ids, generation parameters, launch
    v
FPGA Accelerator (U55C)
    |
    +-- On-chip Prefill/Decode Controller -------------------------------+
    |                                                                    |
    |   [Embedding Lookup]                                               |
    |      token ids -> x0                                               |
    |                                                                    |
    |   [One Reused TinyLlama Decoder-Layer Engine]                      |
    |      reused for layer_idx = 0..21                                  |
    |      same tensor sizes every layer                                 |
    |      different weights and KV-cache region per layer               |
    |                                                                    |
    |   [Final RMSNorm]                                                  |
    |   [LM Head Projection]                                             |
    |   [Greedy Argmax]                                                  |
    |                                                                    |
    +--------------------------------------------------------------------+
    |
    +-- HBM --------------------------------------------------------------+
        | layer weights
        | embedding / LM head weights
        | per-layer K cache
        | per-layer V cache
        +---------------------------------------------------------------+

Host CPU / XRT
    ^
    | generated token ids, status, optional debug dumps
    |
FPGA Accelerator
```

### Architecture Summary

- Target model: TinyLlama 1.1B (`hidden_size=2048`, `intermediate_size=5632`, `22` decoder layers, `32` Q heads, `4` KV heads, `head_dim=64`)
- Runtime modes:
  - `prefill`: process the full prompt sequence and populate the KV cache
  - `decode`: process one new token at a time using the existing KV cache
- Control ownership:
  - host: tokenization, launch, parameter upload, detokenization, display
  - FPGA: embedding lookup, layer execution, KV-cache management, final RMSNorm, LM head, greedy argmax, and EOS/max-token stop handling
- Memory ownership:
  - HBM: weights, embeddings, LM head, K cache, V cache, debug buffer
  - BRAM/URAM: on-chip activation tiles, weight tiles, partial sums, metadata
  - U55C HBM organization: `32` pseudo-channels across `16 GB` of HBM

### Reused Decoder-Layer Engine

All 22 TinyLlama decoder layers use the same block structure and the same tensor sizes.
The accelerator therefore instantiates one decoder-layer engine and reuses it for
`layer_idx = 0..21`.

What is identical across all layers:
- hidden size `2048`
- intermediate size `5632`
- `32` Q heads / `4` KV heads
- head dimension `64`
- same ordered block sequence

What changes per layer:
- the weights
- the two RMSNorm gamma vectors
- that layer's KV-cache region

### Decoder-Layer Engine Flow

```
layer_input
  -> RMSNorm 1
  -> Q / K / V projections
  -> RoPE on Q / K
  -> attention score computation
  -> causal mask + softmax
  -> weighted sum with V
  -> O projection
  -> residual add
  -> RMSNorm 2
  -> gate projection
  -> up projection
  -> SiLU + elementwise multiply
  -> down projection
  -> residual add
  -> layer_output
```

The shared GEMM engine is reused for:
- Q projection
- K projection
- V projection
- attention scores
- weighted sum with V
- O projection
- gate projection
- up projection
- down projection
- LM head projection

So there are two levels of reuse:
- one shared GEMM engine reused across all GEMM-heavy blocks inside a layer
- one decoder-layer engine reused across all 22 layers

### Prefill Versus Decode

#### Prefill

- input: full prompt token sequence
- prefill is sequence-tiled with a fixed tile length of `64` tokens
- embedding lookup produces the first layer input sequence
- every layer writes K and V rows for every processed position into that layer's KV cache
- softmax runs over the real prompt sequence length with a true causal mask
- after layer 21, final RMSNorm + LM head produce the next-token logits

#### Decode

- input: one new token id
- embedding lookup produces one new token embedding
- each layer reads its existing KV cache, appends one new K row and one new V row
- query length is `1`, key length is `cache_len + 1`
- after layer 21, final RMSNorm + LM head produce logits for the next token
- greedy argmax selects the emitted token id

### Mixed-Precision / Quantization Split

Quantized GEMM-heavy blocks:
- embedding output path into the decoder datapath
- Q / K / V / O projections
- attention score computation
- weighted sum with V
- gate / up / down projections
- LM head projection

Arithmetic for quantized GEMMs:
- inputs: `INT8`
- weights: `INT8`
- accumulation: `INT32`

Higher-precision blocks:
- RMSNorm
- RoPE
- softmax
- final RMSNorm

Residual accumulation remains `INT32` before requantization.
KV cache is stored in HBM as symmetric `INT8`, with scale metadata tracked separately for `K` and `V` per layer and per KV head.

### System Blocks And Responsibilities

| Block | Main job |
|---|---|
| Host CPU / XRT | tokenize, launch, upload prompt/params, read generated tokens |
| On-chip Prefill/Decode Controller | sequence prefill and decode, iterate `layer_idx`, manage stop conditions and cache pointers |
| Embedding Lookup | map token ids to the first activation vector/sequence |
| Reused Decoder-Layer Engine | execute one TinyLlama decoder layer for the current `layer_idx` |
| Shared GEMM Engine | run all GEMM-heavy projections and reductions in tiled form |
| Dedicated Blocks | RMSNorm, RoPE, softmax, SiLU, elementwise multiply, residual add |
| HBM Weight Store | hold layer weights, embeddings, LM head weights, and quantization metadata |
| HBM KV Cache | hold per-layer K and V cache state across prefill/decode in INT8 form |
| Final Norm + LM Head + Token Select | convert final hidden state to logits, then choose the next token with greedy argmax |

### Controller Pseudocode

```text
prefill(prompt_tokens):
    x = embedding_lookup(prompt_tokens)
    for layer_idx in 0..21:
        x = decoder_layer_engine_prefill(x, layer_idx, kv_cache[layer_idx])
    hidden = final_rmsnorm(x[-1])
    logits = lm_head(hidden)
    next_token = greedy_argmax(logits)

decode(next_input_token):
    x = embedding_lookup([next_input_token])
    for layer_idx in 0..21:
        x = decoder_layer_engine_decode(x, layer_idx, kv_cache[layer_idx])
    hidden = final_rmsnorm(x[-1])
    logits = lm_head(hidden)
    next_token = greedy_argmax(logits)
    return next_token
```

### Relationship To Section 1

Section 1 below remains useful and accurate as the currently implemented GEMM
validation core. It is not, by itself, the full TinyLlama inference system.
The full accelerator now adds:
- prefill/decode control
- full TinyLlama dimensions
- grouped-query attention
- HBM-resident weights
- HBM-resident KV cache
- FPGA-side final RMSNorm / LM head / token selection

---

## Section 1 — GEMM Engine (Shared MAC Array)

*(Fully implemented and verified. See control_fsm.sv, mac_array.sv, top.sv.)*

---

## FSM State Diagram

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> LOAD      : start=1\n(init k_idx=0, tile_idx=0\nclear_acc=1 for 1 cycle)
    LOAD --> COMPUTE   : BRAM read latency elapsed\n(1-2 cycles, mac_valid=LOW)
    COMPUTE --> COMPUTE : k_idx < K-1\n(k_idx++, mac_valid=HIGH)
    COMPUTE --> WRITE  : k_idx == K-1\n(mac_valid=LOW, wr_en=1)
    WRITE --> LOAD     : tile_idx < N/T-1\n(tile_idx++, clear_acc=1 for 1 cycle\nk_idx=0)
    WRITE --> DONE     : tile_idx == N/T-1
    DONE --> IDLE      : done acknowledged
```

> **Timing notes:**
> - `mac_valid` is HIGH **only** during COMPUTE. It is LOW during LOAD, WRITE, and DONE.
> - `clear_acc` is asserted for **exactly 1 cycle** at the start of IDLE→LOAD and WRITE→LOAD transitions, before COMPUTE begins.
> - LOAD state holds for 1–2 cycles to absorb BRAM read latency before `mac_valid` goes high.

---

## Block-by-Block Description

---

### Host CPU — *Om*
The outside world. Sits outside the FPGA boundary and communicates over PCIe.
In the earlier Phase 1 validation split, it also owned tokenization, embedding
lookup, the 22-layer reuse loop, final RMSNorm, LM head, and sampling. That is
legacy validation behavior, not the current post-Phase-1 target architecture.

| Direction | Signal | Width | Description |
|---|---|---|---|
| OUT → FSM | `start` | 1 | Pulses high to begin computation. FSM transitions IDLE → LOAD. |
| IN ← FSM | `done` | 1 | FSM asserts high when all tiles are complete. Host reads output buffer. |
| IN ← Output Buffer | `hidden state / results (PCIe)` | 64×32b | The completed decoder-layer hidden state y[0..N-1] transferred back to host, then either reused as the next layer input or post-processed after layer 22. |

---

### Input Vector BRAM — *Samarth (instantiation & control) · Satyarth (data)*
Stores the input vector **x** — 64 INT8 values loaded by the host before `start`.
Each cycle during COMPUTE, the FSM provides an address and the BRAM outputs one value that is **broadcast identically to all 8 MAC lanes**.

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← FSM | `k_idx` (rd_addr) | 7b | Read address. Selects which element x[k] to output this cycle. |
| OUT → MAC Array | `x[k]` | 8b INT8 | The current input element. Sent to all 8 lanes simultaneously (broadcast). |

---

### Weight Tile BRAM — *Samarth (instantiation & control) · Satyarth (data)*
Stores one **tile** of weight matrix W at a time — 8 rows × K columns = 512 INT8 values.
Each row maps to one MAC lane. After each tile finishes, the next 8 rows are loaded.

> **BRAM Banking (critical):** A standard single-port BRAM cannot supply 8 independent reads per cycle.
> The weight memory must be implemented as **8 separate BRAM banks — one per lane**.
> Each bank stores one row of the tile: lane `j` reads from its own bank at address `k_idx`.
> Option B (later): use a wide BRAM (64-bit) and pack all 8 weights into a single read.

**Within-tile address (current):**
```
rd_addr[j] = j * K + k_idx        (lane j reads its own row at column k)
```

**Full address if all weights are pre-loaded (future / reference):**
```
rd_addr[j] = (tile_idx * T * K) + (j * K) + k_idx
           = tile_idx * 512 + j * 64 + k_idx      (for T=8, K=64)
```
> The current design reloads the BRAM each tile, so only `j * K + k_idx` is needed in hardware.
> The full formula matters if weights are ever pre-loaded globally (e.g., Phase 2 double buffering).

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← FSM | `k_idx` (rd_addr per bank) | 7b | Column index into the current tile. Each bank uses the same `k_idx`; lane `j` reads from bank `j`. |
| OUT → MAC Array | `W[j,k] × 8 lanes` | 8×8b INT8 | One weight value per lane, read in parallel from 8 separate banks each cycle. |

---

### Control FSM — *Samarth*
The brain of the accelerator. Runs a two-level loop and generates every control signal that drives the other blocks. Contains two counters: `k_idx` (inner loop) and `tile_idx` (outer loop).

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← Host | `start` | 1 | Triggers computation. FSM leaves IDLE state. |
| OUT → Input BRAM | `k_idx` | 7b | Inner loop counter, 0→K-1. Acts as the BRAM read address for x. |
| OUT → Weight BRAM | `k_idx` | 7b | Column index into the current tile. Each of the 8 BRAM banks uses this same address; lane j reads from bank j at `k_idx`. |
| OUT → MAC Array | `mac_valid` | 1 | **HIGH only during COMPUTE state.** LOW during LOAD, WRITE, and DONE. MAC must not accumulate unless this is high. |
| OUT → Accumulators | `clear_acc` | 1 | Pulsed HIGH for **exactly 1 cycle** before COMPUTE begins (on IDLE→LOAD and WRITE→LOAD transitions). Resets all 8 INT32 regs to 0. |
| OUT → Output Buffer | `wr_en` | 1 | Asserted for 1 cycle in the WRITE state (immediately after k_idx == K-1). Triggers write of acc[0..7] into the output buffer at the correct slot. |
| OUT → Host | `done` | 1 | Asserted after the last tile's WRITE state completes. Signals host to read results. |

---

### MAC Array (8 Lanes) — *Rijul*
The actual compute engine. 8 lanes run fully in parallel each cycle.
Each lane performs: `acc[j] += W[j,k] × x[k]` (INT8 × INT8 → INT32).

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← Input BRAM | `x[k]` | 8b INT8 | Single value, broadcast to all 8 lanes. Every lane multiplies by the same x[k]. |
| IN ← Weight BRAM | `W[j,k] × 8` | 8×8b INT8 | One weight per lane. Lane j uses W[j,k], its own unique weight. |
| IN ← FSM | `mac_valid` | 1 | Gate signal. MAC only accumulates when this is high. |
| OUT → Accumulators | `partial sums` | 8×32b INT32 | Running multiply-accumulate result flowing into the accumulator registers each cycle. |

---

### Accumulator Registers — *Rijul*
8 INT32 registers, one per MAC lane. Hold the running dot-product sum across all K=64 steps.
They are **not** reset between k steps — only cleared at tile boundaries via `clear_acc`.

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← MAC Array | `partial sums` | 8×32b | Each cycle's MAC output added into the corresponding register. |
| IN ← FSM | `clear_acc` | 1 | Resets all 8 registers to 0. Fired once before each new tile begins. |
| OUT → Output Buffer | `tile results` | 8×32b INT32 | After the K loop finishes, the 8 final sums are written to the output buffer. |

---

### Output Buffer — *Samarth (write side) · Om (read/PCIe side)*
64 INT32 slots — one for every output element y[0..N-1].
After each tile, 8 values are written into the correct slot range (`tile_idx×8` to `tile_idx×8+7`).
After all 8 tiles, the buffer holds the complete result vector.

> **Write timing:** The write occurs in the dedicated **WRITE state**, exactly 1 cycle after `k_idx == K-1`.
> The FSM asserts `wr_en=1` and drives the write address `tile_idx * T` for 1 cycle, then transitions to LOAD (or DONE).

| Direction | Signal | Width | Description |
|---|---|---|---|
| IN ← Accumulators | `tile results` | 8×32b INT32 | The 8 final accumulator values captured at the end of the K loop. |
| IN ← FSM | `wr_en` | 1 | Write enable. Asserted for exactly 1 cycle in WRITE state. |
| IN ← FSM | `wr_addr` | 4b | Write address = `tile_idx`. Selects which group of 8 output slots to write into (`tile_idx * 8 .. tile_idx * 8 + 7`). |
| OUT → Host | `results (PCIe)` | 64×32b INT32 | Full output vector y transferred back to the host after `done` is asserted. |

---

## Tiling Strategy

```
Output dimension  N = 64   (rows of W1 or W2)
Input dimension   K = 64   (cols of W, length of x)
Tile size         T = 8    (MAC lanes = outputs computed in parallel)

Number of tiles = N / T = 64 / 8 = 8 tiles

For each tile (tile_idx = 0 .. 7):
    Load W rows [tile_idx*8 .. tile_idx*8 + 7] into Weight BRAM
    clear_acc = 1  (reset accumulators)
    For each k (k_idx = 0 .. 63):
        x[k]    = InputBRAM[ k_idx ]
        W[j,k]  = WeightBRAM[ j*K + k_idx ]   for j = 0..7
        mac_valid = 1
        All 8 lanes: acc[j] += W[j,k] * x[k]
    Write acc[0..7] → OutputBuffer[ tile_idx*8 .. tile_idx*8+7 ]

Total MAC operations = N × K = 64 × 64 = 4096
Per tile             = 8 lanes × 64 k-steps = 512 MACs in 64 cycles
Total cycles         ≈ K × (N/T) = 64 × 8 = 512 cycles  (excluding LOAD latency per tile)
```

---

## Memory Layout

```
Input Vector BRAM  (x):
  Depth = 64,  Width = 8 bits
  Address 0..63  →  x[0]..x[63]

Weight Tile BRAM  (8 banks, one per lane):
  Per bank: Depth = K = 64,  Width = 8 bits
  Bank j stores row j of the current tile: W[tile_idx*8 + j, 0..K-1]
  All banks share the same read address: k_idx
  Reloaded from host/DDR for each new tile

  Within-tile address:   rd_addr[j] = k_idx                (used in hardware)
  Full global address:   rd_addr[j] = tile_idx*T*K + j*K + k_idx  (reference only)

Output Buffer:
  Depth = 64,  Width = 32 bits
  Address = tile_idx*8 + lane  →  y[tile_idx*8 + lane]
```

---

## One Full Cycle of Operation

```
1.  Host loads x into Input Vector BRAM
    Host loads W[tile 0] into 8 Weight BRAM banks (one row per bank)
    Host asserts start=1

2.  FSM: IDLE → LOAD
    - k_idx = 0, tile_idx = 0
    - clear_acc = 1 for exactly 1 cycle  (zero all 8 accumulators)
    - mac_valid = LOW

3.  FSM: LOAD  (hold 1–2 cycles for BRAM read latency)
    - BRAM addresses presented, data not yet valid
    - mac_valid = LOW  (MAC must not accumulate during latency)

4.  FSM: LOAD → COMPUTE  (data now valid on BRAM outputs)

5.  For tile_idx = 0 to 7:

      For k_idx = 0 to 63:
        - Input BRAM[k_idx]     → x[k]        (broadcast to all 8 lanes)
        - Weight Bank j[k_idx]  → W[j,k]      (one per lane, 8 banks in parallel)
        - mac_valid = HIGH
        - All 8 lanes: acc[j] += W[j,k] * x[k]
        - k_idx++

      k_idx == K-1: FSM → WRITE state
        - mac_valid = LOW
        - wr_en = 1  (exactly 1 cycle)
        - wr_addr = tile_idx
        - acc[0..7] written to OutputBuffer[tile_idx*8 .. tile_idx*8+7]

      If tile_idx < 7:  FSM → LOAD (next tile)
        - Host reloads Weight BRAMs with next tile's rows
        - clear_acc = 1 for exactly 1 cycle
        - k_idx = 0, tile_idx++

6.  After tile_idx == 7 WRITE completes:
    - FSM → DONE
    - done = 1

7.  Host reads 64 × INT32 results from Output Buffer over PCIe
```
