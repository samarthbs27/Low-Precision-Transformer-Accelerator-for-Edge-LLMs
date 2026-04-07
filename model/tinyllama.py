"""
TinyLlama 1.1B - pure NumPy inference
=====================================

This file is a hardware-shaped NumPy reference for TinyLlama's LLaMA-family
decoder:

- pure NumPy at inference time
- one function per major RTL block
- explicit matmuls, reductions, and elementwise ops
- no PyTorch in the forward pass
- no KV cache in this reference path

Weight loading
--------------
Requires:  pip install torch transformers
Run once:  python model/tinyllama.py --save-weights
Then:      python model/tinyllama.py --prompt "Once upon a time"

Practical note
--------------
TinyLlama is large enough that the saved weight archive needs to stay practical,
this script saves weights as float16 by default, then casts arrays to float32 as
they are used during NumPy inference. That keeps the reference numerically stable
without forcing every tensor to live in host RAM at once.
"""

import argparse
import os
import sys

import numpy as np


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_WEIGHTS = "model/data/tinyllama_weights.npz"


# -----------------------------------------------------------------------------
# BLOCK 1 - RMSNorm
# [RTL] Dedicated normalization block. Computes mean(square(x)) across D_MODEL,
#        then multiplies by rsqrt(mean_sq + eps), then applies the learned gain.
#        TinyLlama uses RMSNorm, not LayerNorm, so there is no mean subtraction.
# -----------------------------------------------------------------------------
def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x_fp = np.asarray(x, dtype=np.float32)
    w_fp = np.asarray(w, dtype=np.float32)
    variance = np.mean(x_fp * x_fp, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps)
    return x_fp * inv_rms * w_fp


# -----------------------------------------------------------------------------
# BLOCK 2 - RoPE
# [RTL] Dedicated rotary unit on Q and K datapaths. The sine/cosine table can
#        live in BRAM and be indexed by position. LLaMA applies RoPE across the
#        full head dimension.
# -----------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, rope_theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return cos, sin


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seq = q.shape[1]
    cos_view = cos[:seq][np.newaxis, :, :]
    sin_view = sin[:seq][np.newaxis, :, :]
    q_rot = (q * cos_view) + (rotate_half(q) * sin_view)
    k_rot = (k * cos_view) + (rotate_half(k) * sin_view)
    return q_rot, k_rot


# -----------------------------------------------------------------------------
# BLOCK 3 - Attention utilities
# [RTL] Causal mask is just the comparison query_pos >= key_pos. TinyLlama also
#        uses grouped-query attention, so K/V heads must be repeated to match the
#        number of Q heads. In hardware this can be handled as an address remap or
#        stream fan-out instead of a true data copy.
# -----------------------------------------------------------------------------
def causal_mask(seq_len: int) -> np.ndarray:
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32)
    if b is not None:
        out = out + np.asarray(b, dtype=np.float32)
    return out


# -----------------------------------------------------------------------------
# BLOCK 4 - Multi-head self-attention with grouped-query attention
# [RTL] Four projection GEMMs (Q, K, V, O), RoPE on Q/K, causal mask, softmax,
#        weighted sum, and a final output projection. This is the same overall
#        hardware shape as any decoder-only attention block, but TinyLlama uses:
#          - separate Q/K/V projection matrices
#          - RMSNorm before the block
#          - grouped-query attention: num_key_value_heads < num_attention_heads
# -----------------------------------------------------------------------------
def self_attention(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    seq = x.shape[0]

    q = linear(x, layer["q_w"], layer["q_b"])
    k = linear(x, layer["k_w"], layer["k_b"])
    v = linear(x, layer["v_w"], layer["v_b"])

    q = q.reshape(seq, cfg["num_attention_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    k = k.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    v = v.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)

    q, k = apply_rope(q, k, cos, sin)

    k = repeat_kv(k, cfg["num_key_value_groups"])
    v = repeat_kv(v, cfg["num_key_value_groups"])

    scores = (q @ k.transpose(0, 2, 1)) * cfg["attn_scale"]
    scores[:, ~causal_mask(seq)] = -1e9

    weights = softmax(scores, axis=-1)
    attended = weights @ v
    attended = attended.transpose(1, 0, 2).reshape(seq, cfg["hidden_size"])

    return linear(attended, layer["o_w"], layer["o_b"])


# -----------------------------------------------------------------------------
# BLOCK 5 - SwiGLU MLP
# [RTL] Two parallel GEMMs (gate_proj and up_proj), SiLU on the gate branch,
#        elementwise multiply, then one down projection GEMM.
# -----------------------------------------------------------------------------
def silu(x: np.ndarray) -> np.ndarray:
    x_fp = np.asarray(x, dtype=np.float32)
    return x_fp / (1.0 + np.exp(-np.clip(x_fp, -60.0, 60.0)))


def feed_forward(x: np.ndarray, layer: dict) -> np.ndarray:
    gate = linear(x, layer["gate_w"], layer["gate_b"])
    up = linear(x, layer["up_w"], layer["up_b"])
    hidden = silu(gate) * up
    return linear(hidden, layer["down_w"], layer["down_b"])


# -----------------------------------------------------------------------------
# BLOCK 6 - Decoder layer
# [RTL] TinyLlama uses standard sequential pre-norm residuals:
#          x = x + attention(rms_norm(x))
#          x = x + mlp(rms_norm(x))
# -----------------------------------------------------------------------------
def decoder_layer(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    attn_input = rms_norm(x, layer["input_norm_w"], cfg["rms_norm_eps"])
    x = x + self_attention(attn_input, layer, cfg, cos, sin)

    ffn_input = rms_norm(x, layer["post_norm_w"], cfg["rms_norm_eps"])
    x = x + feed_forward(ffn_input, layer)
    return x


# -----------------------------------------------------------------------------
# BLOCK 7 - Lazy weight archive
# [RTL] This mirrors how the FPGA will eventually read weights from external
#        memory a block at a time instead of materializing everything in one
#        giant host structure.
# -----------------------------------------------------------------------------
class TinyLlamaWeights:
    def __init__(self, path: str, compute_dtype=np.float32, cache_arrays: bool = False):
        self.path = path
        self.data = np.load(path)
        self.compute_dtype = compute_dtype
        self.cache_arrays = cache_arrays
        self._cache = {}
        self.cfg = {
            "model_id": str(self.data["meta_model_id"].item()),
            "hidden_size": int(self.data["meta_hidden_size"]),
            "intermediate_size": int(self.data["meta_intermediate_size"]),
            "num_hidden_layers": int(self.data["meta_num_hidden_layers"]),
            "num_attention_heads": int(self.data["meta_num_attention_heads"]),
            "num_key_value_heads": int(self.data["meta_num_key_value_heads"]),
            "head_dim": int(self.data["meta_head_dim"]),
            "vocab_size": int(self.data["meta_vocab_size"]),
            "max_position_embeddings": int(self.data["meta_max_position_embeddings"]),
            "rms_norm_eps": float(self.data["meta_rms_norm_eps"]),
            "rope_theta": float(self.data["meta_rope_theta"]),
        }
        self.cfg["num_key_value_groups"] = (
            self.cfg["num_attention_heads"] // self.cfg["num_key_value_heads"]
        )
        self.cfg["attn_scale"] = 1.0 / np.sqrt(self.cfg["head_dim"])

    def _array(self, key: str) -> np.ndarray:
        if self.cache_arrays and key in self._cache:
            return self._cache[key]

        arr = self.data[key]
        if arr.dtype != self.compute_dtype:
            arr = arr.astype(self.compute_dtype, copy=False)

        if self.cache_arrays:
            self._cache[key] = arr
        return arr

    def _optional_array(self, key: str) -> np.ndarray | None:
        if key not in self.data.files:
            return None
        return self._array(key)

    def embed(self, token_ids: list[int]) -> np.ndarray:
        return self._array("embed_w")[token_ids]

    def final_norm_w(self) -> np.ndarray:
        return self._array("final_norm_w")

    def lm_head_w(self) -> np.ndarray:
        return self._array("lm_head_w")

    def layer(self, idx: int) -> dict:
        return {
            "q_w": self._array(f"layer{idx}_q_w"),
            "k_w": self._array(f"layer{idx}_k_w"),
            "v_w": self._array(f"layer{idx}_v_w"),
            "o_w": self._array(f"layer{idx}_o_w"),
            "q_b": self._optional_array(f"layer{idx}_q_b"),
            "k_b": self._optional_array(f"layer{idx}_k_b"),
            "v_b": self._optional_array(f"layer{idx}_v_b"),
            "o_b": self._optional_array(f"layer{idx}_o_b"),
            "gate_w": self._array(f"layer{idx}_gate_w"),
            "up_w": self._array(f"layer{idx}_up_w"),
            "down_w": self._array(f"layer{idx}_down_w"),
            "gate_b": self._optional_array(f"layer{idx}_gate_b"),
            "up_b": self._optional_array(f"layer{idx}_up_b"),
            "down_b": self._optional_array(f"layer{idx}_down_b"),
            "input_norm_w": self._array(f"layer{idx}_input_norm_w"),
            "post_norm_w": self._array(f"layer{idx}_post_norm_w"),
        }


def load_weights(path: str = DEFAULT_WEIGHTS, compute_dtype=np.float32, cache_arrays: bool = False) -> TinyLlamaWeights:
    return TinyLlamaWeights(path, compute_dtype=compute_dtype, cache_arrays=cache_arrays)


# -----------------------------------------------------------------------------
# BLOCK 8 - Weight export from HuggingFace
# [RTL] These arrays are the host-side representation of what later gets loaded
#        into BRAM/HBM. TinyLlama is large, so float16 is the practical default
#        storage format for the weight archive.
# -----------------------------------------------------------------------------
def config_rope_theta(config) -> float:
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])

    return 10000.0


def save_weights_from_hf(
    save_path: str = DEFAULT_WEIGHTS,
    model_id: str = DEFAULT_MODEL_ID,
    save_dtype: str = "float16",
    local_files_only: bool = False,
) -> None:
    from transformers import AutoModelForCausalLM
    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if save_dtype not in dtype_map:
        raise ValueError(f"Unsupported save dtype: {save_dtype}")

    print(f"Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()

    cfg = model.config
    sd = model.state_dict()
    arrays = {}

    torch_dtype = dtype_map[save_dtype]
    np_dtype = np.float16 if save_dtype == "float16" else np.float32

    def to_numpy(tensor):
        return tensor.detach().cpu().to(dtype=torch_dtype).numpy().astype(np_dtype, copy=False)

    arrays["meta_model_id"] = np.array(model_id)
    arrays["meta_hidden_size"] = np.array(cfg.hidden_size, dtype=np.int32)
    arrays["meta_intermediate_size"] = np.array(cfg.intermediate_size, dtype=np.int32)
    arrays["meta_num_hidden_layers"] = np.array(cfg.num_hidden_layers, dtype=np.int32)
    arrays["meta_num_attention_heads"] = np.array(cfg.num_attention_heads, dtype=np.int32)
    arrays["meta_num_key_value_heads"] = np.array(cfg.num_key_value_heads, dtype=np.int32)
    arrays["meta_head_dim"] = np.array(
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        dtype=np.int32,
    )
    arrays["meta_vocab_size"] = np.array(cfg.vocab_size, dtype=np.int32)
    arrays["meta_max_position_embeddings"] = np.array(cfg.max_position_embeddings, dtype=np.int32)
    arrays["meta_rms_norm_eps"] = np.array(cfg.rms_norm_eps, dtype=np.float32)
    arrays["meta_rope_theta"] = np.array(config_rope_theta(cfg), dtype=np.float32)

    arrays["embed_w"] = to_numpy(sd["model.embed_tokens.weight"])
    arrays["final_norm_w"] = to_numpy(sd["model.norm.weight"])
    arrays["lm_head_w"] = to_numpy(sd["lm_head.weight"]).T

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"

        arrays[f"layer{i}_q_w"] = to_numpy(sd[f"{prefix}.self_attn.q_proj.weight"]).T
        arrays[f"layer{i}_k_w"] = to_numpy(sd[f"{prefix}.self_attn.k_proj.weight"]).T
        arrays[f"layer{i}_v_w"] = to_numpy(sd[f"{prefix}.self_attn.v_proj.weight"]).T
        arrays[f"layer{i}_o_w"] = to_numpy(sd[f"{prefix}.self_attn.o_proj.weight"]).T

        q_bias = f"{prefix}.self_attn.q_proj.bias"
        k_bias = f"{prefix}.self_attn.k_proj.bias"
        v_bias = f"{prefix}.self_attn.v_proj.bias"
        o_bias = f"{prefix}.self_attn.o_proj.bias"
        if q_bias in sd:
            arrays[f"layer{i}_q_b"] = to_numpy(sd[q_bias])
        if k_bias in sd:
            arrays[f"layer{i}_k_b"] = to_numpy(sd[k_bias])
        if v_bias in sd:
            arrays[f"layer{i}_v_b"] = to_numpy(sd[v_bias])
        if o_bias in sd:
            arrays[f"layer{i}_o_b"] = to_numpy(sd[o_bias])

        arrays[f"layer{i}_gate_w"] = to_numpy(sd[f"{prefix}.mlp.gate_proj.weight"]).T
        arrays[f"layer{i}_up_w"] = to_numpy(sd[f"{prefix}.mlp.up_proj.weight"]).T
        arrays[f"layer{i}_down_w"] = to_numpy(sd[f"{prefix}.mlp.down_proj.weight"]).T

        gate_bias = f"{prefix}.mlp.gate_proj.bias"
        up_bias = f"{prefix}.mlp.up_proj.bias"
        down_bias = f"{prefix}.mlp.down_proj.bias"
        if gate_bias in sd:
            arrays[f"layer{i}_gate_b"] = to_numpy(sd[gate_bias])
        if up_bias in sd:
            arrays[f"layer{i}_up_b"] = to_numpy(sd[up_bias])
        if down_bias in sd:
            arrays[f"layer{i}_down_b"] = to_numpy(sd[down_bias])

        arrays[f"layer{i}_input_norm_w"] = to_numpy(sd[f"{prefix}.input_layernorm.weight"])
        arrays[f"layer{i}_post_norm_w"] = to_numpy(sd[f"{prefix}.post_attention_layernorm.weight"])

    np.savez(save_path, **arrays)
    print(f"Saved {len(arrays)} arrays to {save_path}")
    print(f"Archive size: {os.path.getsize(save_path) / 1e9:.2f} GB")
    print(f"Stored dtype: {save_dtype}")


# -----------------------------------------------------------------------------
# BLOCK 9 - Full forward pass
# [RTL] Host loop: embedding lookup -> N decoder layers -> final RMSNorm -> LM head.
#        This reference recomputes the whole prefix each decode step rather than
#        using a KV cache so the dataflow stays simple and explicit.
# -----------------------------------------------------------------------------
def forward(token_ids: list[int], weights: TinyLlamaWeights, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    cfg = weights.cfg
    seq = len(token_ids)
    if seq > cfg["max_position_embeddings"]:
        raise ValueError(
            f"Sequence length {seq} exceeds max_position_embeddings={cfg['max_position_embeddings']}"
        )

    x = weights.embed(token_ids)
    for layer_idx in range(cfg["num_hidden_layers"]):
        x = decoder_layer(x, weights.layer(layer_idx), cfg, cos, sin)

    x = rms_norm(x, weights.final_norm_w(), cfg["rms_norm_eps"])
    logits = x @ weights.lm_head_w()
    return logits


# -----------------------------------------------------------------------------
# BLOCK 10 - Greedy generation
# [RTL] Outer control loop for next-token generation. The FPGA only needs the
#        final row of logits for each decode step.
# -----------------------------------------------------------------------------
def generate(
    prompt: str,
    weights: TinyLlamaWeights,
    model_id: str,
    max_new_tokens: int = 20,
    local_files_only: bool = False,
) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    token_ids = tokenizer.encode(prompt)

    cfg = weights.cfg
    cos, sin = build_rope_cache(
        cfg["max_position_embeddings"],
        cfg["head_dim"],
        cfg["rope_theta"],
    )

    print(f"\nPrompt tokens: {token_ids}")
    print(f"Decoded:       {[tokenizer.decode([t]) for t in token_ids]}\n")

    for step in range(max_new_tokens):
        logits = forward(token_ids, weights, cos, sin)
        next_token = int(np.argmax(logits[-1]))
        token_ids.append(next_token)

        piece = tokenizer.decode([next_token])
        print(f"  step {step + 1:2d}: token {next_token:6d} -> {piece!r}")

        if next_token == tokenizer.eos_token_id:
            print("  [EOS]")
            break

    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cache-arrays",
        action="store_true",
        help="Keep loaded arrays in host RAM after first access. Faster, but uses more memory.",
    )
    args = parser.parse_args()

    if args.save_weights:
        save_weights_from_hf(
            save_path=args.weights,
            model_id=args.model_id,
            save_dtype=args.save_dtype,
            local_files_only=args.local_files_only,
        )
        return

    if not os.path.exists(args.weights):
        print(
            f"Weights not found at {args.weights}. Run first: "
            f"python model/tinyllama.py --save-weights"
        )
        sys.exit(1)

    weights = load_weights(args.weights, compute_dtype=np.float32, cache_arrays=args.cache_arrays)
    if weights.cfg["model_id"] != args.model_id:
        print(
            f"Warning: weight archive was saved from {weights.cfg['model_id']}, "
            f"but tokenizer/model-id is {args.model_id}."
        )

    result = generate(
        prompt=args.prompt,
        weights=weights,
        model_id=args.model_id,
        max_new_tokens=args.max_tokens,
        local_files_only=args.local_files_only,
    )
    print(f"\nFull output: {result}")


if __name__ == "__main__":
    main()
