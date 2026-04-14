# Model Folder

This folder contains the Python-side model reference code and helper scripts for
the FPGA accelerator project.

The two most important files here are:
- `tinyllama.py`: the floating-point golden reference for TinyLlama
- `tinyllama_gemm_int8.py`: the mixed-precision GEMM-only INT8 bridge
- `export_fpga_vectors.py`: the golden-trace exporter for RTL/HLS verification

Together, they describe both:
- what the TinyLlama decoder is supposed to do
- how we can start moving FPGA-friendly GEMM blocks toward INT8 without losing
  the floating-point reference

## Files In This Folder

- `tinyllama.py`: pure NumPy TinyLlama inference reference, written in a hardware-shaped style.
- `tinyllama_gemm_int8.py`: mixed-precision INT8 bridge with `analysis` and autoregressive `generate` modes.
- `export_fpga_vectors.py`: exports real TinyLlama golden traces under `sim/golden_traces/` for FPGA verification.
- `model.py`: smaller quantization and FFN experiments used for early datapath work.
- `gen_test_vectors.py`: helper script for producing simulation vectors.

## Purpose Of `tinyllama.py`

`tinyllama.py` is not just an inference script. It is the architectural software
spec for the model datapath we want to map onto the FPGA.

It is intentionally structured so that:
- inference uses pure NumPy only
- each major transformer block is its own function
- all computations are explicit matmuls, reductions, reshapes, and elementwise ops
- there is no PyTorch inside the forward path
- the script mirrors hardware dataflow instead of hiding work in framework kernels

This makes it suitable as:
- a functional golden reference
- a source of tensor shapes and weight formats
- a guide for block decomposition in RTL
- a tool for generating and checking expected behavior before quantization

## Purpose Of `tinyllama_gemm_int8.py`

`tinyllama_gemm_int8.py` is a separate quantization bridge for the same model.
It exists so we can study FPGA-friendly GEMM quantization without changing the
golden floating-point behavior in `tinyllama.py`.

What it does:
- keeps `RMSNorm`, `RoPE`, `softmax`, and residual adds in `float32`
- quantizes only the GEMM-heavy projections to `INT8 x INT8 -> INT32`
- dequantizes GEMM outputs back to `float32`
- compares the mixed-precision path against the `tinyllama.py` reference in `analysis` mode
- can run the mixed-precision path autoregressively in `generate` mode
- can optionally quantize the final LM head too
- can dump intermediate debug arrays for one chosen layer

What it quantizes:
- Q projection
- K projection
- V projection
- O projection
- gate projection
- up projection
- down projection
- optional LM head projection

What it does not quantize yet:
- RMSNorm
- RoPE
- causal masking
- softmax
- residual adds

This is the right intermediate step before a full FPGA implementation because it
lets us isolate how much error comes from quantized matrix multiplies alone.

## Purpose Of `export_fpga_vectors.py`

`export_fpga_vectors.py` is the canonical trace exporter for FPGA verification.

Its job is to:
- run the real TinyLlama Python reference path on deterministic token sequences
- collect model-derived tensors for selected hardware phases
- export canonical `.npz` cases plus a `manifest.json` under `sim/golden_traces/`
- export derived packed `.memh` fixtures for RTL testbenches
- provide arithmetic trace cases that RTL and HLS testbenches can consume later

The currently implemented export scopes are:
- Phase 3:
  - shared GEMM engine traces
  - requantization traces
- Phase 4:
  - prefill and decode RoPE traces
  - prefill and decode causal-mask traces
  - generated RoPE Q16.16 ROM memh files for the RTL rotary datapath
- Phase 5:
  - prefill and decode RMSNorm traces
  - prefill and decode softmax traces
  - prefill and decode SiLU traces
  - packed `.memh` fixtures for the Phase 5 RTL nonlinear-wrapper benches

It uses the fixed GEMM lane-packing contract from the hardware docs so the
exported traces match the production RTL interpretation of one `M_TILE x N_TILE`
output tile. For Phase 4, it also uses the frozen token-major head-slice
packing for RoPE and the fixed `8 x 64` score-chunk packing for the mask path.
For Phase 5, it also exports the wrapper-facing packed chunk layouts used by
the RMSNorm, softmax, and SiLU verification benches.

## TinyLlama Configuration Used By The Scripts

The TinyLlama reference and quantization scripts are built around the following
model metadata:

- model id: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- hidden size: `2048`
- intermediate size: `5632`
- number of decoder layers: `22`
- number of attention heads: `32`
- number of key/value heads: `4`
- number of key/value groups: `8`
- head dimension: `64`
- vocab size: `32000`
- max position embeddings: `2048`
- RMSNorm epsilon: `1e-5`
- RoPE theta: `10000.0`

When a local weights export is present, these values are loaded from archive
metadata by the `TinyLlamaWeights` class inside `tinyllama.py`.

## Important Architectural Differences From The Old Pythia Direction

TinyLlama is a LLaMA-family decoder, so the important model behaviors are:

- normalization is `RMSNorm`, not LayerNorm
- attention uses separate `Q`, `K`, `V`, and `O` projections, not fused QKV
- attention uses grouped-query attention
- the MLP uses `SwiGLU`, not GeLU
- residual structure is standard sequential pre-norm:
  `x = x + attention(rms_norm(x))`
  `x = x + mlp(rms_norm(x))`
- RoPE is applied across the full head dimension
- the reference currently recomputes the entire prefix each decode step
- the reference does not use a KV cache yet
- next-token generation is greedy `argmax`, not sampling

## High-Level Flow Of `tinyllama.py`

The script has three main responsibilities:

1. Save pretrained TinyLlama weights from HuggingFace into a local `.npz` archive.
2. Load those weights lazily for NumPy inference.
3. Run greedy autoregressive generation using a pure NumPy forward pass.

The top-level software path is:

```text
tokenizer.encode(prompt)
-> embedding lookup
-> 22 decoder layers
-> final RMSNorm
-> LM head projection
-> argmax on last-position logits
-> append token
-> repeat
```

## Block-By-Block Explanation

### 1. RMSNorm

Implemented by `rms_norm(x, w, eps)`.

Behavior:
- input shape: `(seq, hidden_size)`
- compute mean of squared values across the hidden dimension
- compute `1 / sqrt(mean_sq + eps)`
- scale the input by that inverse RMS
- multiply by learned gain vector `w`

Formula:

```text
variance = mean(x^2)
out = x * rsqrt(variance + eps) * weight
```

FPGA mapping:
- reduction across 2048 values
- reciprocal square root
- elementwise multiply by input
- elementwise multiply by learned weight vector

### 2. RoPE

Implemented by:
- `build_rope_cache(seq_len, head_dim, rope_theta)`
- `rotate_half(x)`
- `apply_rope(q, k, cos, sin)`

Behavior:
- builds cosine and sine tables for all positions
- applies rotary embedding to both `Q` and `K`
- uses the LLaMA rotate-half convention
- applies RoPE across the full head dimension of `64`

FPGA mapping:
- precompute or store sin/cos tables in BRAM
- apply RoPE as elementwise multiply/add on Q and K datapaths
- no positional embedding add at the token embedding stage

### 3. Attention Utilities

Implemented by:
- `causal_mask(seq_len)`
- `repeat_kv(x, n_rep)`
- `softmax(x)`
- `linear(x, w, b=None)`

Behavior:
- `causal_mask` creates a lower-triangular attention mask
- `repeat_kv` expands 4 K/V heads into 32 attention heads by repeating each K/V head 8 times
- `softmax` is plain stable softmax
- `linear` is a plain matrix multiply plus optional bias

Hardware note:
- the current script physically repeats K/V heads in software
- on FPGA this should be treated as stream fan-out, address remapping, or head reuse
- it should not be implemented as a wasteful full memory copy if avoidable

### 4. Self-Attention

Implemented by `self_attention(x, layer, cfg, cos, sin)`.

Input:
- `x` shape: `(seq, 2048)`

Operations:
- `Q = x @ Wq`
- `K = x @ Wk`
- `V = x @ Wv`
- reshape Q to `(32, seq, 64)`
- reshape K and V to `(4, seq, 64)`
- apply RoPE to Q and K
- repeat K and V heads from 4 to 32
- compute attention scores:
  `scores = Q @ K^T / sqrt(64)`
- apply causal mask
- softmax across the key dimension
- compute weighted value sum
- reshape back to `(seq, 2048)`
- apply output projection `Wo`

Representative actual weight shapes from the saved archive:
- `layer0_q_w`: `(2048, 2048)`
- `layer0_k_w`: `(2048, 256)`
- `layer0_v_w`: `(2048, 256)`
- `layer0_o_w`: `(2048, 2048)`

FPGA mapping:
- GEMM for Q
- GEMM for K
- GEMM for V
- RoPE unit for Q/K
- score GEMM or dot-product engine
- causal mask logic
- softmax block
- weighted-sum GEMM
- output GEMM

This is one of the most important sections for hardware planning because it shows
that TinyLlama uses grouped-query attention rather than full per-head K/V projections.

### 5. SwiGLU MLP

Implemented by:
- `silu(x)`
- `feed_forward(x, layer)`

Operations:
- `gate = x @ Wgate`
- `up = x @ Wup`
- `hidden = silu(gate) * up`
- `out = hidden @ Wdown`

Representative actual weight shapes:
- `layer0_gate_w`: `(2048, 5632)`
- `layer0_up_w`: `(2048, 5632)`
- `layer0_down_w`: `(5632, 2048)`

This is a `SwiGLU` MLP, not a standard 2-layer GeLU MLP.

FPGA mapping:
- one GEMM for gate projection
- one GEMM for up projection
- SiLU activation unit on the gate branch
- elementwise multiply between SiLU(gate) and up
- one GEMM for down projection

This is a good candidate for parallelization because `gate_proj` and `up_proj`
can be computed side by side if the hardware has enough resources.

### 6. Decoder Layer

Implemented by `decoder_layer(x, layer, cfg, cos, sin)`.

TinyLlama uses sequential pre-norm residual structure:

```text
attn_input = rms_norm(x)
x = x + attention(attn_input)
ffn_input = rms_norm(x)
x = x + mlp(ffn_input)
```

This matters for FPGA scheduling because:
- the MLP input depends on the attention-updated residual stream
- unlike Pythia parallel residual, attention and MLP are not independent branches here
- the residual stream must be updated between attention and MLP

### 7. Lazy Weight Archive

Implemented by the `TinyLlamaWeights` class.

What it does:
- opens the `.npz` archive
- reads model metadata into a `cfg` dictionary
- loads arrays only when requested
- optionally caches them in host RAM when `--cache-arrays` is enabled

Why it exists:
- TinyLlama is large enough that loading all arrays eagerly is wasteful for the Python reference
- it mirrors the eventual hardware idea of fetching weights from memory per block rather than materializing everything at once

Important note:
- `--cache-arrays` is a Python runtime convenience only
- it is not itself an FPGA feature
- the hardware analogue is weight residency in BRAM, URAM, HBM, or external DDR plus reuse across decode steps

### 8. HuggingFace Weight Export

Implemented by `save_weights_from_hf(...)`.

What it saves:
- model metadata
- token embedding matrix
- final RMSNorm weight
- LM head matrix
- per-layer attention weights
- per-layer MLP weights
- per-layer RMSNorm vectors

Why save as float16 by default:
- TinyLlama 1.1B is much larger than Pythia-70M
- on-disk float16 is much more manageable
- NumPy inference still casts arrays to float32 for stable compute

The script also supports `--save-dtype float32`, but the default is `float16`.

The saved key layout looks like:

```text
meta_model_id
meta_hidden_size
meta_intermediate_size
meta_num_hidden_layers
meta_num_attention_heads
meta_num_key_value_heads
meta_head_dim
meta_vocab_size
meta_max_position_embeddings
meta_rms_norm_eps
meta_rope_theta

embed_w
final_norm_w
lm_head_w

layer0_q_w
layer0_k_w
layer0_v_w
layer0_o_w
layer0_gate_w
layer0_up_w
layer0_down_w
layer0_input_norm_w
layer0_post_norm_w
...
layer21_*
```

Bias tensors are supported as optional arrays, but the current TinyLlama archive
does not include layer biases.

### 9. Full Forward Pass

Implemented by `forward(token_ids, weights, cos, sin)`.

Operations:
- embedding lookup from `embed_w`
- run all decoder layers in order
- apply final RMSNorm
- project to vocabulary with the LM head

Output:
- `logits` of shape `(seq, vocab_size)`

For token generation, only the last row matters.

### 10. Greedy Generation

Implemented by `generate(prompt, weights, model_id, max_new_tokens, local_files_only)`.

What it does:
- tokenize the prompt
- build the RoPE cache for all allowed positions
- repeatedly call `forward(...)`
- take `argmax(logits[-1])`
- append the next token
- stop on EOS or when `max_new_tokens` is reached

Important limitation:
- this recomputes the full prefix every step
- there is no KV cache in the current reference
- this is simpler for understanding and mapping, but much less efficient than a production decoder

## Command-Line Interface

Typical commands:

```powershell
python model/tinyllama.py --save-weights
python model/tinyllama.py --prompt "Once upon a time"
python model/tinyllama.py --prompt "Once upon a time" --local-files-only --cache-arrays
```

Important flags:

- `--save-weights`: download or load the HuggingFace model and save the `.npz` archive
- `--prompt`: prompt string for generation
- `--model-id`: HuggingFace model id, default is TinyLlama chat
- `--weights`: path to a local `.npz` archive produced by `--save-weights`
- `--max-tokens`: number of new tokens to generate
- `--save-dtype`: `float16` or `float32` for the saved archive
- `--local-files-only`: do not fetch from the network, use local HuggingFace cache only
- `--cache-arrays`: keep already-loaded weight arrays in host RAM after first access

## `tinyllama_gemm_int8.py` Interface

Typical commands:

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --local-files-only
python model/tinyllama_gemm_int8.py --mode analysis --token-ids "1,5,7,9" --layers 1 --local-files-only --cache-arrays
python model/tinyllama_gemm_int8.py --mode generate --prompt "Once upon a time" --layers 22 --max-tokens 20 --local-files-only
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --quantize-lm-head --local-files-only
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --dump-layer 0 --dump-dir sim/tinyllama_layer0 --local-files-only
```

Important flags:

- `--mode`: `analysis` for one-pass comparison, `generate` for autoregressive mixed-precision decoding
- `--prompt`: tokenize and analyze a text prompt
- `--token-ids`: bypass the tokenizer and provide comma-separated ids directly
- `--weights`: path to the TinyLlama `.npz` archive
- `--layers`: number of decoder layers to execute from the front of the model
- `--max-tokens`: number of new tokens to generate in `generate` mode
- `--dump-layer`: choose one executed layer for detailed debug capture
- `--dump-dir`: directory for saved debug arrays and summary text
- `--top-k`: number of highest-scoring next-token candidates to print for the float32 path and the INT8 path
- `--local-files-only`: load tokenizer only from local HuggingFace cache
- `--cache-arrays`: keep already-loaded arrays in host RAM during repeated runs
- `--quantize-lm-head`: also quantize the final vocab projection

In `analysis` mode it prints:
- layer-by-layer isolated and cumulative error
- final-logit error
- float32 next token vs INT8-path next token
- top-k token comparisons

In `generate` mode it prints:
- the input token ids
- one generated token per step from the mixed-precision path
- the final decoded continuation

Behavior difference from `tinyllama.py`:
- `tinyllama.py` always generates with the float32 reference path
- `tinyllama_gemm_int8.py` can either analyze drift or generate with the mixed-precision path
- `analysis` mode is still the right choice when you want layer error and FPGA validation data

## What Is Software-Only Versus FPGA-Relevant

Software-only conveniences:
- `argparse`
- HuggingFace tokenizer loading
- HuggingFace weight download and export
- `.npz` archive management
- `--cache-arrays` host RAM behavior

Directly FPGA-relevant logic:
- RMSNorm math
- RoPE generation and application
- Q/K/V/O projections
- grouped-query attention head handling
- causal masking
- softmax
- weighted sum
- SwiGLU MLP
- residual adds
- final RMSNorm
- LM head projection
- INT8 GEMM quantization strategy from `tinyllama_gemm_int8.py`
- scale handling for dequantizing `INT32` GEMM outputs back into the mixed-precision flow

## Current Hardware Mapping Implications

If we map this model to the FPGA, the reference suggests the following major units:

- embedding lookup unit
- RMSNorm block
- Q projection GEMM
- K projection GEMM
- V projection GEMM
- RoPE block
- attention score block
- causal mask logic
- softmax block
- weighted-sum block
- output projection GEMM
- gate projection GEMM
- up projection GEMM
- SiLU activation block
- elementwise multiply block
- down projection GEMM
- residual adders
- final RMSNorm block
- LM head projection

## FPGA Implementation For INT8 Quantization

This section describes how the current TinyLlama INT8 direction should map onto
the FPGA at a system level.

The key idea is:
- the host owns the autoregressive generation loop
- the FPGA owns the decoder computation for one decode step
- activations flow between layers on the FPGA
- a token is produced only after the full decoder stack and LM head complete

Important correction:
- a token is **not** emitted after every layer
- each layer outputs a hidden-state vector
- only after all decoder layers, final RMSNorm, and LM head projection do we get
  the next-token logits and choose a token

### Recommended INT8 / Mixed-Precision Split

For the first FPGA implementation, the cleanest match to `tinyllama_gemm_int8.py`
is mixed precision:

- GEMM-heavy projections:
  - `INT8 x INT8 -> INT32`
  - Q / K / V / O
  - gate / up / down
  - optional LM head later
- Keep higher precision at first for:
  - RMSNorm
  - RoPE
  - softmax
  - residual accumulation

This matches the current software bridge, where only GEMMs are quantized first.

### One Decode-Step Sequence

The following sequence describes one generated token.

```text
Host CPU
  |
  | 1. Maintain the current prompt/generated token list
  | 2. Send current decode input to FPGA
  v
FPGA Decoder Engine
  |
  | Embedding lookup
  | For layer 0 .. layer 21:
  |   RMSNorm
  |   Q/K/V GEMMs (INT8)
  |   RoPE on Q/K
  |   grouped-query attention
  |   softmax
  |   weighted sum
  |   O GEMM (INT8)
  |   residual add
  |   RMSNorm
  |   gate/up GEMMs (INT8)
  |   SiLU + multiply
  |   down GEMM (INT8)
  |   residual add
  |
  | Final RMSNorm
  | LM head projection
  | Argmax or top-k
  v
Host CPU
  |
  | 3. Receive next token (or logits)
  | 4. Append the token to the running sequence
  | 5. Check EOS / max_tokens
  | 6. Launch the next decode step
```

### Host / FPGA Responsibility Split

#### Host responsibilities

- keep the prompt token list and append newly generated tokens
- launch each decode step
- load or select the correct weight banks / buffers
- decide stop conditions like EOS or `max_tokens`
- optionally perform token decode for printing text

#### FPGA responsibilities

- run the TinyLlama decoder datapath for the current step
- move hidden-state vectors between layers
- execute the INT8 GEMMs and intermediate blocks
- produce final logits or the selected next token

### Layer-Level Dataflow Versus Token-Level Dataflow

Inside the FPGA:

```text
layer_input_vector
-> layer_0_output_vector
-> layer_1_output_vector
-> ...
-> layer_21_output_vector
-> final_norm
-> lm_head
-> logits
-> next token
```

Across autoregressive steps:

```text
prompt tokens
-> FPGA decode step
-> next token
-> host appends token
-> FPGA decode step
-> next token
-> host appends token
-> repeat
```

So the feedback loop happens at the **token level**, not between decoder layers.

### Phase 1 Recommendation

For the first working FPGA version:

- keep the host in charge of autoregressive generation
- keep the FPGA focused on one decode step at a time
- avoid KV cache initially
- match the arithmetic split used by `tinyllama_gemm_int8.py`
- verify one decoder layer first, then scale up to all 22 layers

This is the least risky path because it matches the current software structure
and avoids building a full on-chip token-generation controller too early.

### Later Optimization Path

Once correctness is established, the next upgrades are:

- move more sequencing from host to an on-chip controller FSM
- add KV cache support so old keys/values are reused instead of recomputed
- optimize grouped-query attention data movement
- decide whether the FPGA returns full logits, top-k, or just argmax
- decide whether the LM head stays on FPGA or is split with the host

## Important Practical Notes For FPGA Planning

- The Python reference currently computes in float32 even though the weight archive is stored as float16 by default.
- The FPGA implementation will almost certainly need mixed precision or quantization rather than full float32 everywhere.
- The current script is intentionally functional and readable, not cycle-optimized.
- The reference currently recomputes the full prefix on every decode step.
- A real accelerator will likely want a KV cache later to avoid redoing old attention work.
- The grouped-query attention structure is a key TinyLlama-specific optimization target.
- The LM head is very large: `2048 x 32000`, so it is a major memory and bandwidth consideration.
- The current INT8 bridge quantizes GEMMs only; it is not yet a full end-to-end INT8 TinyLlama decoder.
- The current INT8 bridge is the safest place to experiment with FPGA-style quantization before modifying the golden reference.

## Recommended Run Order

If you want to run the full workflow the same way we used the floating-point
script, use this sequence:

### 1. Save or refresh the TinyLlama weights

```powershell
python model/tinyllama.py --save-weights
```

### 2. Run the pure NumPy floating-point reference

```powershell
python model/tinyllama.py --prompt "Once upon a time"
```

### 3. Run the same model through the GEMM-only INT8 bridge

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --local-files-only
```

That gives you the first layer of mixed-precision analysis while keeping the rest
of the flow comparable to the floating-point reference.

This command evaluates the supplied prompt once, runs the selected decoder
layers, and compares the next-token prediction between the float32 reference
path and the mixed-precision INT8 path.

If you want to run the quantized bridge across the full TinyLlama decoder depth,
use all 22 layers:

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 22 --local-files-only
```

### 4. Run the bridge as an autoregressive mixed-precision generator

```powershell
python model/tinyllama_gemm_int8.py --mode generate --prompt "Once upon a time" --layers 22 --max-tokens 20 --local-files-only
```

This is the quantized equivalent of the generation loop in `tinyllama.py`, but
it uses the mixed-precision GEMM path instead of the float32 reference path.

### 5. Run the bridge with host-side caching for faster repeated tests

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --local-files-only --cache-arrays
```

### 6. Quantize the LM head too for a harsher end-to-end test

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --quantize-lm-head --local-files-only
```

### 7. Dump one layer for FPGA debugging

```powershell
python model/tinyllama_gemm_int8.py --mode analysis --prompt "Once upon a time" --layers 1 --dump-layer 0 --dump-dir sim/tinyllama_layer0 --local-files-only
```

This is the closest equivalent to “running all the steps” for the quantized path:
- save weights
- run the floating-point reference
- run the INT8 GEMM bridge in `analysis` mode
- run the INT8 GEMM bridge in `generate` mode
- optionally turn on LM-head quantization
- optionally dump one layer for RTL comparison

## Recommended Next Steps

For software:
- keep `tinyllama.py` as the golden floating-point reference
- add tensor dump modes for one layer at a time
- add mixed-precision or INT8 bridge experiments for TinyLlama specifically

For FPGA:
- start with one decoder layer
- keep sequence handling simple at first
- decide early whether softmax and RMSNorm stay higher precision
- decide how K/V head repetition is implemented without wasteful copying
- plan memory movement for the large LM head separately from the smaller layer weights

This README should be treated as the human-readable architecture summary of what
`tinyllama.py` is doing and why it is structured the way it is.
