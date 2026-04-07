# Theory & Computation — FPGA Transformer Accelerator

This document explains what the hardware is computing, the mathematics behind it, why INT8 is used, and how the design maps to TinyLlama.

---

## 1. What a Transformer Decoder Layer Does

A transformer decoder layer takes a sequence of token embeddings and applies two sub-blocks in order:

1. **Self-Attention** — lets each token attend to other tokens in the sequence, weighting their contributions by relevance
2. **Feed-Forward Network (FFN)** — processes each token independently through three linear layers with a gated activation between them (SwiGLU)

Each sub-block is wrapped with a **residual connection** and **RMSNorm**. TinyLlama uses **pre-norm** structure: normalization is applied *before* each sub-block, then the result is added to the residual stream.

```
x = x + attention(rms_norm(x))
x = x + mlp(rms_norm(x))
```

---

## 2. The Full Decoder Layer Computation

```
Input X  [seq_len × d_model]
  │
  ├─ attn_input = RMSNorm(X)                            ← pre-norm (dedicated HW)
  ├─ Q = attn_input × Wq                                ← Q projection (GEMM)
  ├─ K = attn_input × Wk                                ← K projection (GEMM)
  ├─ V = attn_input × Wv                                ← V projection (GEMM)
  ├─ Q, K = RoPE(Q, K)                                  ← rotary embeddings (dedicated HW)
  ├─ Scores = Q × Kᵀ / √d_k                             ← scaled dot-product (GEMM)
  ├─ Weights = softmax(Scores)                           ← row-wise softmax (dedicated HW)
  ├─ Attn = Weights × V                                  ← weighted sum (GEMM)
  ├─ AttnOut = Attn × W_O                                ← output projection (GEMM)
  ├─ X = X + AttnOut                                     ← residual add #1 (dedicated HW)
  │
  ├─ ffn_input = RMSNorm(X)                              ← pre-norm (dedicated HW)
  ├─ gate = ffn_input × W_gate                           ← gate projection (GEMM) ┐ parallel
  ├─ up   = ffn_input × W_up                             ← up projection   (GEMM) ┘
  ├─ hidden = SiLU(gate) × up                            ← SwiGLU activation (dedicated HW)
  ├─ FFNOut = hidden × W_down                            ← down projection (GEMM)
  └─ X = X + FFNOut                                      ← residual add #2 (dedicated HW)
       │
     Output  [seq_len × d_model]
```

Every GEMM step is handled by the same shared MAC array, reloaded with different weights. The dedicated hardware blocks handle all non-linear operations.

**TinyLlama uses SwiGLU, not a standard 2-layer FFN.** The key difference:
- Standard FFN: `W2 × activation(W1 × x)` — 2 GEMMs, sequential
- SwiGLU: `W_down × (SiLU(W_gate × x) × W_up × x)` — 3 GEMMs, gate and up run in parallel

---

## 3. The Core Operation — Matrix-Vector Multiply

All GEMM steps reduce to repeated matrix-vector multiplies. For one output vector:

```
y[i] = Σ_k  W[i,k] × x[k]     for i = 0..N-1
```

This is a dot product between row `i` of W and the input vector x. N dot products in total, each requiring K multiply-accumulate (MAC) operations.

**Total work per GEMM = N × K MACs.**

At TinyLlama scale (d_model=2048, d_ff=5632): Q/K/O projections are 2048×2048 = 4.2M MACs each; FFN projections are up to 2048×5632 = 11.5M MACs each.

---

## 4. Why This Is Expensive on a CPU

At inference time in a real model, N and K are in the thousands. A single forward pass through one TinyLlama decoder layer requires ~80M MACs across all 8 GEMMs.

CPUs spend most cycles on memory access, instruction decode, and branching. FPGAs can be wired specifically to keep MAC units fed every single clock cycle, with no overhead between multiplications. Weights stream directly from HBM into the compute array.

---

## 5. INT8 Quantization

Real models store weights in FP32 (4 bytes per weight). For inference, we reduce to INT8 (1 byte) — 4× less memory, 4× faster to move from storage.

### How it works

Map the float range `[min, max]` of a tensor to the integer range `[-128, 127]`:

```
x_quantized = round(x_float / scale)
scale = max(|x_float|) / 127
```

To recover approximate floats: `x_float ≈ x_quantized × scale`

### Why INT8 × INT8 → INT32

When multiplying two INT8 values and accumulating K of them:
- Max single product: `127 × 127 = 16,129`
- After K=2048 accumulations: `2048 × 16,129 = 33,032,192` — needs INT32

So: **multiply INT8 × INT8, accumulate in INT32**. This is exactly what `mac_unit.sv` does:

```systemverilog
acc_out = acc_in + $signed(a) * $signed(b);  // INT8×INT8, accumulates to INT32
```

### Accuracy

INT8 inference typically loses less than 1% accuracy on most models when weights and activations are carefully quantized. Validated in this project: same top-1 token prediction as FP32 reference (layer 0 MAE: 0.021).

---

## 6. Tiling — Handling N > MAC Lanes

The MAC array has T lanes and computes T dot products in parallel (one per output element). For N=2048 with T=512: 4 tiles. For test dims (N=64, T=8): 8 tiles.

### The tiling loop

```
for tile_idx = 0 to N/T-1:
    load W rows [tile_idx*T .. tile_idx*T+T-1] from HBM into on-chip buffer
    clear accumulators to 0
    for k_idx = 0 to K-1:
        x_k     = x[k_idx]                        ← broadcast to all T lanes
        w_jk[j] = W[tile_idx*T + j, k_idx]        ← T weights, one per lane
        acc[j] += w_jk[j] × x_k                   ← T MACs in parallel
    write acc[0..T-1] → y[tile_idx*T .. +T-1]
```

**Each k iteration is one clock cycle** — all T lanes compute simultaneously. This is **output-stationary** dataflow: accumulators stay fixed while weights and inputs stream through.

### Cycle count (N=2048, K=2048, T=512 — TinyLlama Q projection)

| Phase | Cycles per tile | Notes |
|-------|----------------|-------|
| LOAD | 2 | On-chip buffer fill latency |
| COMPUTE | 2048 | One MAC per lane per cycle |
| WRITE | 1 | Commit T accumulators |
| **Per tile** | **~2051** | |
| **4 tiles total** | **~8,204** | At 300 MHz → 27 μs |

---

## 7. Output-Stationary Dataflow

| Strategy | What stays fixed | What streams in |
|----------|-----------------|-----------------|
| **Output-stationary** (ours) | Accumulator for y[i] | Weights and inputs stream in |
| Weight-stationary | Weight tile in registers | Inputs and partial sums move |
| Input-stationary | Input vector in registers | Weights and partial sums move |

Output-stationary minimises accumulator register writes. The FSM reflects this directly: `clear_acc` once, accumulate for K cycles, write once.

---

## 8. The Non-Linear Operations

### RMSNorm

TinyLlama uses RMSNorm, not standard LayerNorm. The key difference: **no mean subtraction**.

```
rms = sqrt(mean(x²) + eps)
y[i] = (x[i] / rms) × weight[i]
```

Compared to LayerNorm: `y[i] = (x[i] - mean) / sqrt(variance + eps) × gamma + beta`

RMSNorm is simpler in hardware — one pass to compute mean-of-squares, one reciprocal sqrt, elementwise multiply by learned gain. No mean subtraction, no beta shift.

### RoPE (Rotary Positional Embedding)

Applied to Q and K after projection, before the attention score GEMM. Encodes position by rotating pairs of elements in the head dimension:

```
q_rot = (q × cos) + (rotate_half(q) × sin)
k_rot = (k × cos) + (rotate_half(k) × sin)
```

where `cos` and `sin` are precomputed tables indexed by position. In hardware: sin/cos tables stored in BRAM, applied as elementwise multiply/add on the Q and K datapaths.

### Grouped-Query Attention (GQA)

TinyLlama uses 32 Q heads but only 4 K/V heads. Each K/V head is shared by 8 Q heads.

```
num_attention_heads = 32
num_key_value_heads = 4
num_key_value_groups = 8   (= 32 / 4)
```

In hardware: K and V projections produce 4×64 = 256-element vectors instead of 2048. Each K/V head is reused across 8 Q heads. This reduces K/V projection GEMM cost by 8× and K/V memory by 8×.

### Softmax

Applied row-wise to the attention scores matrix. For each row:

```
softmax(z)[i] = exp(z[i] - max(z)) / Σ_j exp(z[j] - max(z))
```

Requires exponentiation (LUT-based), a running sum, and division. Implemented as a dedicated streaming unit (Rijul). For seq_len=1, softmax of a single score = 1.0 (passthrough in Phase 1).

### SwiGLU / SiLU Activation

TinyLlama's FFN uses SwiGLU, not ReLU or GELU. The activation function is SiLU:

```
SiLU(x) = x / (1 + exp(-x)) = x × sigmoid(x)
```

The full SwiGLU operation:

```
hidden = SiLU(gate_proj(x)) × up_proj(x)
output = down_proj(hidden)
```

Gate and up projections run in parallel (two simultaneous GEMMs or two sequential invocations of the shared MAC array). The SiLU unit applies the activation to the gate branch only, then elementwise-multiplies with the up branch.

In hardware: SiLU requires `exp(-x)` — approximated with a LUT or Vivado IP. Simpler than GELU but more complex than ReLU.

### Residual Adders

Two elementwise addition units — one before each RMSNorm. Each adds two vectors of the same length element-by-element. Simple combinational logic (Samarth).

---

## 9. How This Relates to TinyLlama

The hardware implements one full TinyLlama decoder layer, reused 22 times. The host drives the 22-iteration loop, loading the correct weight bank from HBM each iteration.

| Parameter | TinyLlama 1.1B | This project (Phase 1 test) |
|-----------|----------------|----------------------------|
| Hidden dimension (d_model) | 2048 | 64 |
| FFN intermediate (d_ff) | 5632 | 256 |
| Number of layers | 22 | 1 (reused 22×) |
| Attention heads (Q) | 32 | 1 |
| KV heads (GQA) | 4 | 1 |
| Normalization | RMSNorm | RMSNorm |
| Positional encoding | RoPE | RoPE |
| FFN activation | SiLU (SwiGLU) | SiLU (SwiGLU) |
| Weight precision | FP32 → INT8 | INT8 |
| MAC lanes (target) | ~512 | 8 (test) |

The hardware FSM and MAC array are fully parameterised (N, K, T). Scaling up requires larger on-chip buffers and more MAC lanes — the control logic, tiling, and dataflow are unchanged.

---

## 10. Resource Feasibility on U55C

### Weight storage

| Scope | INT8 weight size |
|---|---|
| One decoder layer | ~43.5 MB |
| All 22 layers | ~957 MB |
| Embeddings + LM head | ~131 MB |
| **Full model** | **~1.1 GB** |

### U55C on-chip memory

| Resource | Capacity |
|---|---|
| BRAM (2,016 × 36Kb) | ~9 MB |
| URAM (960 × 288Kb) | ~34.6 MB |
| **Total on-chip** | **~43.6 MB** |
| HBM | 8 GB |

**Conclusion:** One layer's weights (~43.5 MB) barely fits the total on-chip memory, leaving no room for activations or control. All 22 layers (957 MB) are impossible on-chip. The only viable approach: **weights stream from HBM; activations (tiny at ~2 KB per vector) stay on-chip.**

### MAC array sizing

| MAC lanes | Q projection cycles | Time at 300 MHz | Tokens/sec (estimated) |
|---|---|---|---|
| 8 (current) | 524,288 | 1.75 ms | <1 |
| 128 | 32,768 | 109 μs | ~30 |
| 512 | 8,192 | 27 μs | ~120 |
| 1024 | 4,096 | 14 μs | ~250 |

U55C has 9,024 DSPs. A 512-lane INT8 MAC array uses ~256 DSPs (3% of available). **Target: 512 lanes for TinyLlama-speed inference.**

---

## 11. The Full Picture

```
Host CPU (Om)
    │
    │  PCIe: sends input token embeddings (INT8)
    │  PCIe: drives 22-layer iteration loop (start/done per layer)
    │  PCIe: receives layer output after all 22 layers
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  FPGA (Alveo U55C)                                               │
│                                                                  │
│  ┌─────────┐   RMSNorm ──► Q, K, V projections                  │
│  │ Input X │   (dedicated)   (3 × MAC array, GQA: 32Q/4KV)     │
│  └─────────┘                      │                             │
│       │                      RoPE on Q, K                       │
│       │                   (dedicated HW, Rijul)                 │
│       │                          │                              │
│       │                   Attention Scores (MAC array)          │
│       │                   Softmax (dedicated HW, Rijul)         │
│       │                   Weighted Sum × V (MAC array)          │
│       │                   Output Projection × W_O (MAC array)   │
│       │                          │                              │
│       └──────────► Residual Add #1 ◄────────────────────────────┘
│                   (dedicated HW, Samarth)                        │
│                          │                                       │
│                     RMSNorm (dedicated HW, Rijul)               │
│                          │                                       │
│         gate_proj ──► SiLU ──┐                                  │
│         up_proj   ──────────►× ──► down_proj (MAC array)        │
│                          │         (SwiGLU MLP)                  │
│                          │                                       │
│       ┌──────────────────┘                                       │
│       └──────────► Residual Add #2 (dedicated HW, Samarth)      │
│                          │                                       │
│                     RMSNorm (dedicated HW, Rijul)               │
│                          │                                       │
│                     Layer Output  [→ host after 22 iterations]  │
│                                                                  │
│  Weights: streamed from HBM (8 GB) per GEMM step                │
│  Activations: on-chip BRAM/URAM (~2 KB per vector)              │
└──────────────────────────────────────────────────────────────────┘
```

**Satyarth** — quantizes TinyLlama weights to INT8 and verifies outputs match the FP32 reference.
**Rijul** — builds the MAC array (compute core) and all dedicated non-linear hardware blocks (RMSNorm, RoPE, softmax, SiLU, elementwise multiply).
**Samarth** — builds the FSM that sequences every operation, manages tiling and HBM weight streaming, and implements the residual adders.
**Om** — connects the FPGA to the host via PCIe/XRT, drives the 22-layer iteration loop, and handles data movement.
