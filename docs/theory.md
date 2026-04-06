# Theory & Computation — FPGA Transformer Accelerator

This document explains what the hardware is computing, the mathematics behind it, why INT8 is used, and how the design maps to a real transformer model.

---

## 1. What a Transformer Decoder Layer Does

A transformer decoder layer takes a sequence of token embeddings and applies two sub-blocks in order:

1. **Self-Attention** — lets each token attend to other tokens in the sequence, weighting their contributions by relevance
2. **Feed-Forward Network (FFN)** — processes each token independently through two linear layers with a non-linearity between them

Each sub-block is wrapped with a **residual connection** and **layer normalization**. This project implements the full decoder layer pipeline in hardware.

---

## 2. The Full Decoder Layer Computation

```
Input X  [seq_len × d_model]
  │
  ├─ Q = X × Wq,  K = X × Wk,  V = X × Wv      ← QKV projections (3 GEMMs)
  │
  ├─ Scores = Q × Kᵀ / √d_k                       ← scaled dot-product attention (GEMM)
  ├─ Weights = softmax(Scores)                     ← row-wise softmax (dedicated HW)
  ├─ Attn = Weights × V                            ← weighted sum (GEMM)
  ├─ AttnOut = Attn × W_O                          ← output projection (GEMM)
  │
  ├─ Residual Add #1: X + AttnOut                  ← elementwise add (dedicated HW)
  ├─ LayerNorm 1                                   ← mean/variance/scale (dedicated HW)
  │
  ├─ H = LayerNorm1_out × W1                       ← FFN Layer 1 (GEMM)
  ├─ H_act = Activation(H)                         ← ReLU / GELU (dedicated HW)
  ├─ FFNOut = H_act × W2                           ← FFN Layer 2 (GEMM)
  │
  ├─ Residual Add #2: LayerNorm1_out + FFNOut      ← elementwise add (dedicated HW)
  └─ LayerNorm 2                                   ← mean/variance/scale (dedicated HW)
       │
     Output  [seq_len × d_model]
```

Every GEMM step is handled by the same shared MAC array, reloaded with different weights. The dedicated hardware blocks (softmax, activation, layer norm, residual adders) are separate combinational/pipelined units.

---

## 3. The Core Operation — Matrix-Vector Multiply

All GEMM steps reduce to repeated matrix-vector multiplies. For one output vector:

```
y[i] = Σ_k  W[i,k] × x[k]     for i = 0..N-1
```

This is a dot product between row `i` of W and the input vector x. N dot products in total, each requiring K multiply-accumulate (MAC) operations.

**Total work per GEMM = N × K MACs.**

At test scale (N=64, K=64): 4,096 MACs. At a real model scale (e.g. d_model=512, d_k=64 for Pythia-70M): hundreds of thousands per step.

---

## 4. Why This Is Expensive on a CPU

At inference time in a real model, N and K are not 64 — they're in the hundreds to thousands. A single forward pass through a small model like Pythia-70M (~70M parameters) still requires billions of MACs across all layers and heads.

CPUs spend most cycles on memory access, instruction decode, and branching. FPGAs can be wired specifically to keep MAC units fed every single clock cycle, with no overhead between multiplications.

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
- After K=64 accumulations: `64 × 16,129 = 1,032,256` — needs INT32

So: **multiply INT8 × INT8, accumulate in INT32**. This is exactly what `mac_unit.sv` does:

```systemverilog
acc_out = acc_in + $signed(a) * $signed(b);  // INT8×INT8, accumulates to INT32
```

The INT32 accumulator holds the full-precision dot product. After all K steps, Satyarth's software re-scales to INT8 for the next layer.

### Accuracy

INT8 inference typically loses less than 1% accuracy on most models when weights and activations are carefully quantized.

---

## 6. Tiling — Handling N > 8 Lanes

The MAC array has 8 lanes and computes 8 dot products in parallel (one per output element). For N=64 we need 8 groups of 8 — **8 tiles**.

### The tiling loop

```
for tile_idx = 0 to N/T-1:
    load W rows [tile_idx*T .. tile_idx*T+T-1] into Weight BRAM
    clear accumulators to 0
    for k_idx = 0 to K-1:
        x_k     = x[k_idx]                        ← broadcast to all 8 lanes
        w_jk[j] = W[tile_idx*T + j, k_idx]        ← 8 weights, one per lane
        acc[j] += w_jk[j] × x_k                   ← 8 MACs in parallel
    write acc[0..T-1] → y[tile_idx*T .. +T-1]
```

**Each k iteration is one clock cycle** — all 8 lanes compute simultaneously. This is **output-stationary** dataflow: accumulators stay fixed while weights and inputs stream through.

### Cycle count (N=64, K=64, T=8)

| Phase | Cycles per tile | Notes |
|-------|----------------|-------|
| LOAD | 2 | BRAM read latency |
| COMPUTE | 64 | One MAC per lane per cycle |
| WRITE | 1 | Commit 8 accumulators |
| **Per tile** | **67** | |
| **8 tiles total** | **~538** | Matches testbench result |

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

The GEMM steps are handled by the MAC array. The non-linear operations require separate hardware:

### Softmax

Applied row-wise to the attention scores matrix. For each row:

```
softmax(z)[i] = exp(z[i]) / Σ_j exp(z[j])
```

Requires exponentiation (LUT-based), a running sum, and division. Cannot be done by a MAC array. Implemented as a dedicated streaming unit (Rijul).

### Layer Normalization

Applied after each residual add:

```
y[i] = (x[i] - μ) / √(σ² + ε) × γ + β
```

where μ and σ² are the mean and variance of the input vector, and γ, β are learned scale/shift parameters. Requires a two-pass reduction over the vector (first pass: compute mean; second pass: compute variance and normalize). Implemented as a dedicated pipelined unit (Rijul).

### Activation (ReLU / GELU)

Between FFN Layer 1 and FFN Layer 2:
- **ReLU:** `max(0, x)` — trivial, one comparison
- **GELU:** `x × Φ(x)` where Φ is the standard normal CDF — approximated with a LUT

Implemented as a small dedicated block (Rijul).

### Residual Adders

Two elementwise addition units — one before each layer norm. Each adds two vectors of the same length element-by-element. Simple combinational logic (Samarth).

---

## 9. How This Relates to a Real Model

The architecture is identical to a standard transformer decoder layer. The only difference from a real model is scale.

| Parameter | Real model (e.g. Pythia-70M) | This project |
|-----------|------------------------------|-------------|
| Hidden dimension (d_model) | 512 | 64 (test) |
| FFN hidden dimension | 2048 | 64 (test) |
| Number of layers | 6 | 1 layer |
| Attention heads | 8 | 1 head |
| Activation | GELU | ReLU / GELU |
| Weight precision | FP32 → INT8 | INT8 |

The hardware is not hardcoded to N=64, K=64. The FSM and MAC array are fully parameterised:

```systemverilog
parameter int N = 64,   // output dimension
parameter int K = 64,   // inner dimension
parameter int T = 8     // MAC lanes (tile size)
```

Scaling up requires larger BRAMs and more MAC lanes — the control logic, tiling, and dataflow are unchanged.

---

## 10. The Full Picture

```
Host CPU (Om)
    │
    │  PCIe: sends input token embeddings (INT8)
    │  PCIe: receives layer output (INT32, re-scaled by Satyarth's software)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  FPGA (Alveo U55C)                                               │
│                                                                  │
│  ┌─────────┐   Q, K, V projections ──► Attention Scores         │
│  │ Input X │   (3 × MAC array)          (MAC array)             │
│  └─────────┘                                │                   │
│       │                                  Softmax                │
│       │                              (dedicated HW, Rijul)      │
│       │                                     │                   │
│       │                              Weighted Sum × V           │
│       │                                 (MAC array)             │
│       │                                     │                   │
│       │                              Output Projection × W_O    │
│       │                                 (MAC array)             │
│       │                                     │                   │
│       └──────────────► Residual Add #1 ◄────┘                   │
│                        (dedicated HW, Samarth)                  │
│                                │                                │
│                           Layer Norm 1                          │
│                        (dedicated HW, Rijul)                    │
│                                │                                │
│                    FFN Layer 1 (MAC array)                      │
│                    Activation  (dedicated HW, Rijul)            │
│                    FFN Layer 2 (MAC array)                      │
│                                │                                │
│       ┌────────────────────────┘                                │
│       └──────────────► Residual Add #2                          │
│                        (dedicated HW, Samarth)                  │
│                                │                                │
│                           Layer Norm 2                          │
│                        (dedicated HW, Rijul)                    │
│                                │                                │
│                          Layer Output                           │
└──────────────────────────────────────────────────────────────────┘
```

**Satyarth** — quantizes weights to INT8 and verifies outputs match the FP32 reference.  
**Rijul** — builds the MAC array (compute core) and all dedicated non-linear hardware blocks.  
**Samarth** — builds the FSM that sequences every operation, manages tiling and BRAM addressing, and implements the residual adders.  
**Om** — connects the FPGA to the host via PCIe/XRT so data flows in and results flow out.
