"""
TinyLlama GEMM-only INT8 bridge
===============================

This script is a mixed-precision INT8 bridge for TinyLlama.

It supports two modes:
- analysis: one-pass float32 vs INT8-path comparison
- generate: autoregressive mixed-precision token generation

It keeps the numerically sensitive blocks in float32:
- RMSNorm
- RoPE
- softmax
- residual adds

It quantizes only the GEMM-heavy linear projections to INT8 with INT32
accumulation and float32 dequantization:
- Q / K / V / O projections
- gate / up / down projections
- optional LM head projection

The goal is to measure how close an FPGA-friendly INT8 GEMM path stays to the
golden floating-point reference in model/tinyllama.py.
"""

import argparse
import os
from typing import Any

import numpy as np

import tinyllama as ref


def quantize_int8(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = np.asarray(tensor, dtype=np.float32)
    max_abs = float(np.max(np.abs(tensor))) if tensor.size else 0.0
    scale = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q = np.round(tensor / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_int32(accum: np.ndarray, input_scale: float, weight_scale: float) -> np.ndarray:
    return accum.astype(np.float32) * (input_scale * weight_scale)


def diff_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    abs_diff = np.abs(diff)
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(abs_diff.max()),
    }


def quantized_linear(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_fp = np.asarray(x, dtype=np.float32)
    w_fp = np.asarray(w, dtype=np.float32)

    x_q, x_scale = quantize_int8(x_fp)
    w_q, w_scale = quantize_int8(w_fp)
    accum = x_q.astype(np.int32) @ w_q.astype(np.int32)
    out = dequantize_int32(accum, x_scale, w_scale)
    if b is not None:
        out = out + np.asarray(b, dtype=np.float32)

    debug: dict[str, Any] = {
        "input_scale": np.array([x_scale], dtype=np.float32),
        "weight_scale": np.array([w_scale], dtype=np.float32),
    }
    if capture_arrays:
        debug.update({
            "input_fp": x_fp,
            "input_q": x_q,
            "weight_fp": w_fp,
            "weight_q": w_q,
            "accum_int32": accum,
            "output_fp": out.astype(np.float32),
        })
    return out.astype(np.float32), debug


def reference_self_attention(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    seq = x.shape[0]

    q = ref.linear(x, layer["q_w"], layer["q_b"])
    k = ref.linear(x, layer["k_w"], layer["k_b"])
    v = ref.linear(x, layer["v_w"], layer["v_b"])

    q_heads = q.reshape(seq, cfg["num_attention_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    k_heads = k.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    v_heads = v.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)

    q_heads, k_heads = ref.apply_rope(q_heads, k_heads, cos, sin)
    k_rep = ref.repeat_kv(k_heads, cfg["num_key_value_groups"])
    v_rep = ref.repeat_kv(v_heads, cfg["num_key_value_groups"])

    scores = (q_heads @ k_rep.transpose(0, 2, 1)) * cfg["attn_scale"]
    scores[:, ~ref.causal_mask(seq)] = -1e9
    weights = ref.softmax(scores, axis=-1)

    attended = weights @ v_rep
    attended = attended.transpose(1, 0, 2).reshape(seq, cfg["hidden_size"])
    out = ref.linear(attended, layer["o_w"], layer["o_b"])

    debug: dict[str, Any] = {
        "q_proj_fp": q,
        "k_proj_fp": k,
        "v_proj_fp": v,
    }
    if capture_arrays:
        debug.update({
            "scores_fp": scores,
            "weights_fp": weights,
            "attended_fp": attended,
            "out_fp": out,
        })
    return out.astype(np.float32), debug


def mixed_precision_self_attention(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    seq = x.shape[0]

    q, q_debug = quantized_linear(x, layer["q_w"], layer["q_b"], capture_arrays=capture_arrays)
    k, k_debug = quantized_linear(x, layer["k_w"], layer["k_b"], capture_arrays=capture_arrays)
    v, v_debug = quantized_linear(x, layer["v_w"], layer["v_b"], capture_arrays=capture_arrays)

    q_heads = q.reshape(seq, cfg["num_attention_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    k_heads = k.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)
    v_heads = v.reshape(seq, cfg["num_key_value_heads"], cfg["head_dim"]).transpose(1, 0, 2)

    q_heads, k_heads = ref.apply_rope(q_heads, k_heads, cos, sin)
    k_rep = ref.repeat_kv(k_heads, cfg["num_key_value_groups"])
    v_rep = ref.repeat_kv(v_heads, cfg["num_key_value_groups"])

    scores = (q_heads @ k_rep.transpose(0, 2, 1)) * cfg["attn_scale"]
    scores[:, ~ref.causal_mask(seq)] = -1e9
    weights = ref.softmax(scores, axis=-1)

    attended = weights @ v_rep
    attended = attended.transpose(1, 0, 2).reshape(seq, cfg["hidden_size"])
    out, o_debug = quantized_linear(attended, layer["o_w"], layer["o_b"], capture_arrays=capture_arrays)

    debug: dict[str, Any] = {
        "q_proj": q_debug,
        "k_proj": k_debug,
        "v_proj": v_debug,
        "o_proj": o_debug,
        "q_proj_fp": q,
        "k_proj_fp": k,
        "v_proj_fp": v,
    }
    if capture_arrays:
        debug.update({
            "scores_fp": scores,
            "weights_fp": weights,
            "attended_fp": attended,
            "out_fp": out,
        })
    return out.astype(np.float32), debug


def reference_feed_forward(
    x: np.ndarray,
    layer: dict,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    gate = ref.linear(x, layer["gate_w"], layer["gate_b"])
    up = ref.linear(x, layer["up_w"], layer["up_b"])
    hidden = ref.silu(gate) * up
    out = ref.linear(hidden, layer["down_w"], layer["down_b"])

    debug: dict[str, Any] = {
        "gate_proj_fp": gate,
        "up_proj_fp": up,
        "hidden_fp": hidden,
    }
    if capture_arrays:
        debug["out_fp"] = out
    return out.astype(np.float32), debug


def mixed_precision_feed_forward(
    x: np.ndarray,
    layer: dict,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    gate, gate_debug = quantized_linear(x, layer["gate_w"], layer["gate_b"], capture_arrays=capture_arrays)
    up, up_debug = quantized_linear(x, layer["up_w"], layer["up_b"], capture_arrays=capture_arrays)
    hidden = ref.silu(gate) * up
    out, down_debug = quantized_linear(hidden, layer["down_w"], layer["down_b"], capture_arrays=capture_arrays)

    debug: dict[str, Any] = {
        "gate_proj": gate_debug,
        "up_proj": up_debug,
        "down_proj": down_debug,
        "gate_proj_fp": gate,
        "up_proj_fp": up,
        "hidden_fp": hidden,
    }
    if capture_arrays:
        debug["out_fp"] = out
    return out.astype(np.float32), debug


def reference_decoder_layer(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    attn_input = ref.rms_norm(x, layer["input_norm_w"], cfg["rms_norm_eps"])
    attn_out, attn_debug = reference_self_attention(attn_input, layer, cfg, cos, sin, capture_arrays=capture_arrays)
    x_after_attn = x + attn_out

    ffn_input = ref.rms_norm(x_after_attn, layer["post_norm_w"], cfg["rms_norm_eps"])
    ffn_out, ffn_debug = reference_feed_forward(ffn_input, layer, capture_arrays=capture_arrays)
    out = x_after_attn + ffn_out

    debug: dict[str, Any] = {
        "attn_input_fp": attn_input,
        "x_after_attn_fp": x_after_attn,
        "ffn_input_fp": ffn_input,
        "attn": attn_debug,
        "ffn": ffn_debug,
    }
    if capture_arrays:
        debug["layer_out_fp"] = out
    return out.astype(np.float32), debug


def mixed_precision_decoder_layer(
    x: np.ndarray,
    layer: dict,
    cfg: dict,
    cos: np.ndarray,
    sin: np.ndarray,
    capture_arrays: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    attn_input = ref.rms_norm(x, layer["input_norm_w"], cfg["rms_norm_eps"])
    attn_out, attn_debug = mixed_precision_self_attention(attn_input, layer, cfg, cos, sin, capture_arrays=capture_arrays)
    x_after_attn = x + attn_out

    ffn_input = ref.rms_norm(x_after_attn, layer["post_norm_w"], cfg["rms_norm_eps"])
    ffn_out, ffn_debug = mixed_precision_feed_forward(ffn_input, layer, capture_arrays=capture_arrays)
    out = x_after_attn + ffn_out

    debug: dict[str, Any] = {
        "attn_input_fp": attn_input,
        "x_after_attn_fp": x_after_attn,
        "ffn_input_fp": ffn_input,
        "attn": attn_debug,
        "ffn": ffn_debug,
    }
    if capture_arrays:
        debug["layer_out_fp"] = out
    return out.astype(np.float32), debug


def decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return ""
    return tokenizer.decode([token_id])


def safe_repr(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def top_tokens(logits: np.ndarray, tokenizer, k: int) -> list[tuple[int, float, str]]:
    indices = np.argsort(logits)[-k:][::-1]
    rows = []
    for idx in indices:
        rows.append((int(idx), float(logits[idx]), decode_token(tokenizer, int(idx))))
    return rows


def flatten_named_arrays(prefix: str, obj: Any) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}_{key}" if prefix else key
            flat.update(flatten_named_arrays(child_prefix, value))
    elif isinstance(obj, np.ndarray):
        flat[prefix] = obj
    return flat


def dump_debug(
    dump_dir: str,
    token_ids: list[int],
    reference_debug: dict[str, Any],
    mixed_debug: dict[str, Any],
    summary_lines: list[str],
) -> None:
    os.makedirs(dump_dir, exist_ok=True)
    arrays = {"token_ids": np.asarray(token_ids, dtype=np.int32)}
    arrays.update(flatten_named_arrays("reference", reference_debug))
    arrays.update(flatten_named_arrays("mixed", mixed_debug))
    np.savez(os.path.join(dump_dir, "debug_arrays.npz"), **arrays)

    with open(os.path.join(dump_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def maybe_load_tokenizer(model_id: str, local_files_only: bool):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as exc:
        print(f"Warning: could not load tokenizer for {model_id}: {exc}")
        return None


def parse_token_ids(token_ids_arg: str | None) -> list[int] | None:
    if token_ids_arg is None:
        return None
    values = []
    for piece in token_ids_arg.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return values


def projection_stats_line(name: str, ref_array: np.ndarray, mix_array: np.ndarray) -> str:
    stats = diff_stats(ref_array, mix_array)
    return f"{name}: mae={stats['mae']:.6f}, max_abs={stats['max_abs']:.6f}"


def validate_layer_count(cfg: dict, layer_count: int) -> None:
    if layer_count < 1 or layer_count > cfg["num_hidden_layers"]:
        raise ValueError(f"--layers must be between 1 and {cfg['num_hidden_layers']}")


def resolve_token_ids(
    prompt: str | None,
    token_ids_arg: str | None,
    tokenizer,
) -> list[int]:
    token_ids = parse_token_ids(token_ids_arg)
    if token_ids is None:
        if prompt is None:
            raise ValueError("Provide either --prompt or --token-ids")
        if tokenizer is None:
            raise ValueError("Tokenizer unavailable, so --token-ids is required")
        token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Input token sequence is empty")
    return token_ids


def load_runtime(
    weights_path: str,
    layer_count: int,
    local_files_only: bool,
    cache_arrays: bool,
    prompt: str | None,
    token_ids_arg: str | None,
) -> tuple[ref.TinyLlamaWeights, dict, Any, list[int], np.ndarray, np.ndarray]:
    weights = ref.load_weights(weights_path, compute_dtype=np.float32, cache_arrays=cache_arrays)
    cfg = weights.cfg
    validate_layer_count(cfg, layer_count)

    tokenizer = maybe_load_tokenizer(cfg["model_id"], local_files_only=local_files_only)
    token_ids = resolve_token_ids(prompt, token_ids_arg, tokenizer)

    cos, sin = ref.build_rope_cache(
        cfg["max_position_embeddings"],
        cfg["head_dim"],
        cfg["rope_theta"],
    )
    return weights, cfg, tokenizer, token_ids, cos, sin


def print_token_context(token_ids: list[int], tokenizer) -> None:
    print(f"Token ids: {token_ids}")
    if tokenizer is not None:
        print(f"Decoded tokens: {[safe_repr(tokenizer.decode([tok])) for tok in token_ids]}")
    print("")


def forward_reference(
    token_ids: list[int],
    weights: ref.TinyLlamaWeights,
    cos: np.ndarray,
    sin: np.ndarray,
    layer_count: int,
) -> np.ndarray:
    cfg = weights.cfg
    seq = len(token_ids)
    if seq > cfg["max_position_embeddings"]:
        raise ValueError(
            f"Sequence length {seq} exceeds max_position_embeddings={cfg['max_position_embeddings']}"
        )

    x = weights.embed(token_ids).astype(np.float32)
    for layer_idx in range(layer_count):
        x, _ = reference_decoder_layer(
            x,
            weights.layer(layer_idx),
            cfg,
            cos,
            sin,
            capture_arrays=False,
        )

    x = ref.rms_norm(x, weights.final_norm_w(), cfg["rms_norm_eps"])
    return (x @ weights.lm_head_w()).astype(np.float32)


def forward_mixed(
    token_ids: list[int],
    weights: ref.TinyLlamaWeights,
    cos: np.ndarray,
    sin: np.ndarray,
    layer_count: int,
    quantize_lm_head: bool,
) -> np.ndarray:
    cfg = weights.cfg
    seq = len(token_ids)
    if seq > cfg["max_position_embeddings"]:
        raise ValueError(
            f"Sequence length {seq} exceeds max_position_embeddings={cfg['max_position_embeddings']}"
        )

    x = weights.embed(token_ids).astype(np.float32)
    for layer_idx in range(layer_count):
        x, _ = mixed_precision_decoder_layer(
            x,
            weights.layer(layer_idx),
            cfg,
            cos,
            sin,
            capture_arrays=False,
        )

    x = ref.rms_norm(x, weights.final_norm_w(), cfg["rms_norm_eps"])
    if quantize_lm_head:
        logits, _ = quantized_linear(x, weights.lm_head_w(), None, capture_arrays=False)
        return logits.astype(np.float32)
    return (x @ weights.lm_head_w()).astype(np.float32)


def run_analysis(
    prompt: str | None,
    token_ids_arg: str | None,
    weights_path: str,
    layer_count: int,
    dump_layer: int | None,
    dump_dir: str | None,
    local_files_only: bool,
    top_k: int,
    quantize_lm_head: bool,
    cache_arrays: bool,
) -> None:
    weights, cfg, tokenizer, token_ids, cos, sin = load_runtime(
        weights_path=weights_path,
        layer_count=layer_count,
        local_files_only=local_files_only,
        cache_arrays=cache_arrays,
        prompt=prompt,
        token_ids_arg=token_ids_arg,
    )
    if dump_layer is not None and (dump_layer < 0 or dump_layer >= layer_count):
        raise ValueError("--dump-layer must be within the executed layer range")
    print_token_context(token_ids, tokenizer)

    x_ref = weights.embed(token_ids).astype(np.float32)
    x_mix = x_ref.copy()

    summary_lines = [
        f"Token ids: {token_ids}",
        f"Layers analyzed: {layer_count}",
        "Layer stats:",
    ]

    dump_reference: dict[str, Any] | None = None
    dump_mixed: dict[str, Any] | None = None

    for layer_idx in range(layer_count):
        layer = weights.layer(layer_idx)
        capture = dump_layer == layer_idx

        ref_out, ref_debug = reference_decoder_layer(
            x_ref,
            layer,
            cfg,
            cos,
            sin,
            capture_arrays=capture,
        )
        iso_mix_out, iso_mix_debug = mixed_precision_decoder_layer(
            x_ref,
            layer,
            cfg,
            cos,
            sin,
            capture_arrays=capture,
        )
        cum_mix_out, _ = mixed_precision_decoder_layer(
            x_mix,
            layer,
            cfg,
            cos,
            sin,
            capture_arrays=False,
        )

        isolated = diff_stats(ref_out, iso_mix_out)
        cumulative = diff_stats(ref_out, cum_mix_out)

        line = (
            f"Layer {layer_idx}: "
            f"isolated MAE={isolated['mae']:.6f}, isolated max={isolated['max_abs']:.6f}, "
            f"cumulative MAE={cumulative['mae']:.6f}, cumulative max={cumulative['max_abs']:.6f}"
        )
        print(line)
        summary_lines.append(line)

        if capture:
            dump_reference = ref_debug
            dump_mixed = iso_mix_debug

            breakdown_lines = [
                projection_stats_line("  attn q_proj", ref_debug["attn"]["q_proj_fp"], iso_mix_debug["attn"]["q_proj_fp"]),
                projection_stats_line("  attn k_proj", ref_debug["attn"]["k_proj_fp"], iso_mix_debug["attn"]["k_proj_fp"]),
                projection_stats_line("  attn v_proj", ref_debug["attn"]["v_proj_fp"], iso_mix_debug["attn"]["v_proj_fp"]),
                projection_stats_line("  ffn gate_proj", ref_debug["ffn"]["gate_proj_fp"], iso_mix_debug["ffn"]["gate_proj_fp"]),
                projection_stats_line("  ffn up_proj", ref_debug["ffn"]["up_proj_fp"], iso_mix_debug["ffn"]["up_proj_fp"]),
            ]

            if "out_fp" in ref_debug["attn"] and "out_fp" in iso_mix_debug["attn"]:
                breakdown_lines.append(
                    projection_stats_line("  attn o_proj", ref_debug["attn"]["out_fp"], iso_mix_debug["attn"]["out_fp"])
                )
            if "out_fp" in ref_debug["ffn"] and "out_fp" in iso_mix_debug["ffn"]:
                breakdown_lines.append(
                    projection_stats_line("  ffn down_proj", ref_debug["ffn"]["out_fp"], iso_mix_debug["ffn"]["out_fp"])
                )

            for breakdown in breakdown_lines:
                print(breakdown)
                summary_lines.append(breakdown)

        x_ref = ref_out
        x_mix = cum_mix_out

    x_ref = ref.rms_norm(x_ref, weights.final_norm_w(), cfg["rms_norm_eps"])
    x_mix = ref.rms_norm(x_mix, weights.final_norm_w(), cfg["rms_norm_eps"])

    ref_logits = x_ref[-1:] @ weights.lm_head_w()
    lm_debug = {}
    if quantize_lm_head:
        mix_logits, lm_debug = quantized_linear(
            x_mix[-1:],
            weights.lm_head_w(),
            None,
            capture_arrays=dump_dir is not None,
        )
    else:
        mix_logits = x_mix[-1:] @ weights.lm_head_w()

    logit_stats = diff_stats(ref_logits, mix_logits)
    ref_token = int(np.argmax(ref_logits[0]))
    mix_token = int(np.argmax(mix_logits[0]))

    print("")
    print(
        f"Final logits: MAE={logit_stats['mae']:.6f}, "
        f"RMSE={logit_stats['rmse']:.6f}, max_abs={logit_stats['max_abs']:.6f}"
    )
    print(f"Float32 next token: {ref_token} -> '{safe_repr(decode_token(tokenizer, ref_token))}'")
    print(f"INT8 path next token: {mix_token} -> '{safe_repr(decode_token(tokenizer, mix_token))}'")

    print("")
    print(f"Top {top_k} float32 tokens:")
    for idx, value, decoded in top_tokens(ref_logits[0], tokenizer, top_k):
        print(f"  {idx:6d}  {value:12.6f}  '{safe_repr(decoded)}'")

    print("")
    print(f"Top {top_k} INT8-path tokens:")
    for idx, value, decoded in top_tokens(mix_logits[0], tokenizer, top_k):
        print(f"  {idx:6d}  {value:12.6f}  '{safe_repr(decoded)}'")

    summary_lines.extend([
        "",
        f"Final logits: mae={logit_stats['mae']:.6f}, rmse={logit_stats['rmse']:.6f}, max_abs={logit_stats['max_abs']:.6f}",
        f"Float32 next token: {ref_token} -> '{safe_repr(decode_token(tokenizer, ref_token))}'",
        f"INT8 path next token: {mix_token} -> '{safe_repr(decode_token(tokenizer, mix_token))}'",
    ])

    if dump_dir is not None:
        arrays_ref = dump_reference or {}
        arrays_mix = dump_mixed or {}
        if quantize_lm_head and lm_debug:
            arrays_mix = dict(arrays_mix)
            arrays_mix["lm_head"] = lm_debug
        dump_debug(dump_dir, token_ids, arrays_ref, arrays_mix, summary_lines)
        print("")
        print(f"Saved debug dump to {dump_dir}")


def run_generate(
    prompt: str | None,
    token_ids_arg: str | None,
    weights_path: str,
    layer_count: int,
    max_tokens: int,
    local_files_only: bool,
    quantize_lm_head: bool,
    cache_arrays: bool,
) -> None:
    if max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    weights, cfg, tokenizer, token_ids, cos, sin = load_runtime(
        weights_path=weights_path,
        layer_count=layer_count,
        local_files_only=local_files_only,
        cache_arrays=cache_arrays,
        prompt=prompt,
        token_ids_arg=token_ids_arg,
    )

    print_token_context(token_ids, tokenizer)
    if layer_count != cfg["num_hidden_layers"]:
        print(
            f"Generating with {layer_count} / {cfg['num_hidden_layers']} decoder layers. "
            "Use the full layer count for normal TinyLlama behavior."
        )
        print("")

    generated_ids = list(token_ids)
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    for step in range(max_tokens):
        logits = forward_mixed(
            generated_ids,
            weights,
            cos,
            sin,
            layer_count,
            quantize_lm_head=quantize_lm_head,
        )
        next_token = int(np.argmax(logits[-1]))
        generated_ids.append(next_token)

        piece = decode_token(tokenizer, next_token)
        print(f"  step {step + 1:2d}: token {next_token:6d} -> '{safe_repr(piece)}'")

        if eos_token_id is not None and next_token == eos_token_id:
            print("  [EOS]")
            break

    print("")
    if tokenizer is not None:
        print(f"Full output: {safe_repr(tokenizer.decode(generated_ids))}")
    else:
        print(f"Output token ids: {generated_ids}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["analysis", "generate"],
        default="analysis",
        help="analysis = one-pass float32 vs INT8 comparison, generate = autoregressive INT8-path generation",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--token-ids", type=str, default=None)
    parser.add_argument("--weights", type=str, default=ref.DEFAULT_WEIGHTS)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--dump-layer", type=int, default=None)
    parser.add_argument("--dump-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-arrays", action="store_true")
    parser.add_argument(
        "--quantize-lm-head",
        action="store_true",
        help="Also quantize the final vocab projection. Off by default to isolate decoder GEMMs first.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found at {args.weights}. Run model/tinyllama.py --save-weights first."
        )

    if args.mode == "generate":
        if args.dump_layer is not None or args.dump_dir is not None:
            raise ValueError("--dump-layer and --dump-dir are only supported in --mode analysis")

        run_generate(
            prompt=args.prompt,
            token_ids_arg=args.token_ids,
            weights_path=args.weights,
            layer_count=args.layers,
            max_tokens=args.max_tokens,
            local_files_only=args.local_files_only,
            quantize_lm_head=args.quantize_lm_head,
            cache_arrays=args.cache_arrays,
        )
        return

    run_analysis(
        prompt=args.prompt,
        token_ids_arg=args.token_ids,
        weights_path=args.weights,
        layer_count=args.layers,
        dump_layer=args.dump_layer,
        dump_dir=args.dump_dir,
        local_files_only=args.local_files_only,
        top_k=args.top_k,
        quantize_lm_head=args.quantize_lm_head,
        cache_arrays=args.cache_arrays,
    )


if __name__ == "__main__":
    main()
