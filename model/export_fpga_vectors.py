"""
TinyLlama FPGA golden-trace exporter
===================================

This script exports deterministic, real-model trace cases for FPGA verification.

The first implementation focuses on Phase 3 arithmetic blocks:
- shared GEMM engine
- requantize unit

Trace outputs are written under:
  sim/golden_traces/

The canonical format and policy are documented in:
  docs/golden_trace_plan.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import tinyllama as ref
import tinyllama_gemm_int8 as bridge


PHASE3 = "phase3"

M_TILE = 16
N_TILE = 32
GEMM_LANES = 512
SCALE_VECTOR_ELEMS = 16
Q16_16_FRAC_BITS = 16
K_TILE = 64

DEFAULT_PREFILL_TOKEN_IDS = ",".join(str(i) for i in range(1, 17))
DEFAULT_DECODE_TOKEN_IDS = "1"


PHASE3_OPS = (
    ("q_proj", "attn", "q_proj"),
    ("k_proj", "attn", "k_proj"),
    ("v_proj", "attn", "v_proj"),
    ("o_proj", "attn", "o_proj"),
    ("gate_proj", "ffn", "gate_proj"),
    ("up_proj", "ffn", "up_proj"),
    ("down_proj", "ffn", "down_proj"),
)


def parse_token_ids(token_ids_arg: str) -> list[int]:
    token_ids: list[int] = []
    for piece in token_ids_arg.split(","):
        piece = piece.strip()
        if piece:
            token_ids.append(int(piece))
    if not token_ids:
        raise ValueError("Token-id list is empty")
    return token_ids


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def q16_16_from_float(value: float) -> np.uint32:
    scaled = int(round(max(0.0, value) * float(1 << Q16_16_FRAC_BITS)))
    scaled = max(0, min(scaled, 0xFFFFFFFF))
    return np.uint32(scaled)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_hex_memh(
    path: Path,
    values: np.ndarray,
    *,
    bits: int,
    signed: bool,
) -> None:
    ensure_dir(path.parent)
    flat = np.asarray(values).reshape(-1)
    mask = (1 << bits) - 1
    width = ceil_div(bits, 4)
    with path.open("w", encoding="utf-8") as handle:
        for raw_value in flat:
            value = int(raw_value)
            if signed and value < 0:
                value = (value + (1 << bits)) & mask
            else:
                value &= mask
            handle.write(f"{value:0{width}x}\n")


def write_packed_lane_memh(
    path: Path,
    values: np.ndarray,
    *,
    bits_per_lane: int,
    signed: bool,
) -> None:
    ensure_dir(path.parent)
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    lane_mask = (1 << bits_per_lane) - 1
    lane_width = ceil_div(bits_per_lane, 4)
    with path.open("w", encoding="utf-8") as handle:
        for row in arr:
            words: list[str] = []
            for raw_value in row[::-1]:
                value = int(raw_value)
                if signed and value < 0:
                    value = (value + (1 << bits_per_lane)) & lane_mask
                else:
                    value &= lane_mask
                words.append(f"{value:0{lane_width}x}")
            handle.write("".join(words))
            handle.write("\n")


def collect_layer_input(
    weights: ref.TinyLlamaWeights,
    token_ids: list[int],
    target_layer: int,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    x = np.asarray(weights.embed(token_ids), dtype=np.float32)
    for layer_idx in range(target_layer):
        x = ref.decoder_layer(x, weights.layer(layer_idx), weights.cfg, cos, sin)
    return x.astype(np.float32)


def pack_row_major_lane_steps(
    act_tile: np.ndarray,
    wt_tile: np.ndarray,
    acc_tile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    m_count, k_total = act_tile.shape
    k_total_w, n_count = wt_tile.shape

    if k_total != k_total_w:
        raise ValueError(
            f"GEMM tile shape mismatch: act K={k_total}, wt K={k_total_w}"
        )
    if m_count > M_TILE or n_count > N_TILE:
        raise ValueError(
            f"Tile exceeds fixed packing geometry: m_count={m_count}, n_count={n_count}"
        )

    active_lane_count = m_count * n_count
    act_steps = np.zeros((k_total, GEMM_LANES), dtype=np.int8)
    wt_steps = np.zeros((k_total, GEMM_LANES), dtype=np.int8)
    acc_lane = np.zeros((GEMM_LANES,), dtype=np.int32)

    for m_local in range(m_count):
        for n_local in range(n_count):
            lane = (m_local * N_TILE) + n_local
            act_steps[:, lane] = act_tile[m_local, :]
            wt_steps[:, lane] = wt_tile[:, n_local]
            acc_lane[lane] = np.int32(acc_tile[m_local, n_local])

    return act_steps, wt_steps, acc_lane, active_lane_count


def build_gemm_case(
    *,
    runtime_mode: str,
    layer_id: int,
    op_name: str,
    debug_dict: dict[str, Any],
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    input_q = np.asarray(debug_dict["input_q"], dtype=np.int8)
    weight_q = np.asarray(debug_dict["weight_q"], dtype=np.int8)
    accum = np.asarray(debug_dict["accum_int32"], dtype=np.int32)

    seq_count, k_total = input_q.shape
    _, output_dim = weight_q.shape
    m_count = min(M_TILE, seq_count)
    n_count = min(N_TILE, output_dim)

    act_tile = input_q[:m_count, :]
    wt_tile = weight_q[:, :n_count]
    acc_tile = accum[:m_count, :n_count]
    act_steps, wt_steps, acc_lane, active_lane_count = pack_row_major_lane_steps(
        act_tile=act_tile,
        wt_tile=wt_tile,
        acc_tile=acc_tile,
    )

    case_id = f"{PHASE3}_{runtime_mode}_layer{layer_id}_{op_name}_gemm_m0_n0"
    case_path = output_root / PHASE3 / f"{case_id}.npz"
    ensure_dir(case_path.parent)

    np.savez(
        case_path,
        act=act_tile,
        wt=wt_tile,
        acc_expected=acc_tile,
        act_steps=act_steps,
        wt_steps=wt_steps,
        acc_expected_lane=acc_lane,
        active_lane_count=np.asarray([active_lane_count], dtype=np.int32),
    )

    rtl_root = output_root / PHASE3 / "rtl"
    fixture_base = rtl_root / case_id
    write_hex_memh(
        fixture_base.with_suffix(".meta.memh"),
        np.asarray([k_total, active_lane_count, m_count, n_count], dtype=np.uint32),
        bits=32,
        signed=False,
    )
    write_hex_memh(
        fixture_base.with_suffix(".act_steps.memh"),
        act_steps,
        bits=8,
        signed=True,
    )
    write_hex_memh(
        fixture_base.with_suffix(".wt_steps.memh"),
        wt_steps,
        bits=8,
        signed=True,
    )
    write_hex_memh(
        fixture_base.with_suffix(".acc_expected_lane.memh"),
        acc_lane,
        bits=32,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".act_steps_packed.memh"),
        act_steps,
        bits_per_lane=8,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".wt_steps_packed.memh"),
        wt_steps,
        bits_per_lane=8,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".acc_expected_packed.memh"),
        acc_lane,
        bits_per_lane=32,
        signed=True,
    )

    manifest_entries.append(
        {
            "case_id": case_id,
            "phase": PHASE3,
            "block": "shared_gemm_engine",
            "path": str(case_path.as_posix()),
            "runtime_mode": runtime_mode,
            "layer_id": layer_id,
            "token_start": 0,
            "token_count": int(seq_count),
            "m_tile_idx": 0,
            "n_tile_idx": 0,
            "k_tile_idx": 0,
            "dtype_summary": {
                "act": "int8",
                "wt": "int8",
                "acc_expected": "int32",
                "act_steps": "int8",
                "wt_steps": "int8",
                "acc_expected_lane": "int32",
            },
            "shape_summary": {
                "act": list(act_tile.shape),
                "wt": list(wt_tile.shape),
                "acc_expected": list(acc_tile.shape),
                "act_steps": list(act_steps.shape),
                "wt_steps": list(wt_steps.shape),
                "acc_expected_lane": list(acc_lane.shape),
            },
            "lane_packing": "row_major_mxn",
            "active_lane_count": active_lane_count,
            "k_total": int(k_total),
            "m_count": int(m_count),
            "n_count": int(n_count),
            "rtl_fixture_base": str(fixture_base.as_posix()),
        }
    )


def build_gemm_smoke_case(
    *,
    runtime_mode: str,
    layer_id: int,
    op_name: str,
    debug_dict: dict[str, Any],
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    input_q = np.asarray(debug_dict["input_q"], dtype=np.int8)
    weight_q = np.asarray(debug_dict["weight_q"], dtype=np.int8)

    seq_count, k_total = input_q.shape
    _, output_dim = weight_q.shape
    m_count = min(M_TILE, seq_count)
    n_count = min(N_TILE, output_dim)
    k_slice = min(K_TILE, k_total)

    act_tile = input_q[:m_count, :k_slice]
    wt_tile = weight_q[:k_slice, :n_count]
    acc_tile = act_tile.astype(np.int32) @ wt_tile.astype(np.int32)
    act_steps, wt_steps, acc_lane, active_lane_count = pack_row_major_lane_steps(
        act_tile=act_tile,
        wt_tile=wt_tile,
        acc_tile=acc_tile,
    )

    case_id = f"{PHASE3}_{runtime_mode}_layer{layer_id}_{op_name}_gemm_smoke_m0_n0_k0"
    case_path = output_root / PHASE3 / f"{case_id}.npz"
    ensure_dir(case_path.parent)

    np.savez(
        case_path,
        act=act_tile,
        wt=wt_tile,
        acc_expected=acc_tile,
        act_steps=act_steps,
        wt_steps=wt_steps,
        acc_expected_lane=acc_lane,
        active_lane_count=np.asarray([active_lane_count], dtype=np.int32),
    )

    rtl_root = output_root / PHASE3 / "rtl"
    fixture_base = rtl_root / case_id
    write_hex_memh(
        fixture_base.with_suffix(".meta.memh"),
        np.asarray([k_slice, active_lane_count, m_count, n_count], dtype=np.uint32),
        bits=32,
        signed=False,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".act_steps_packed.memh"),
        act_steps,
        bits_per_lane=8,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".wt_steps_packed.memh"),
        wt_steps,
        bits_per_lane=8,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".acc_expected_packed.memh"),
        acc_lane,
        bits_per_lane=32,
        signed=True,
    )

    manifest_entries.append(
        {
            "case_id": case_id,
            "phase": PHASE3,
            "block": "shared_gemm_engine",
            "path": str(case_path.as_posix()),
            "runtime_mode": runtime_mode,
            "layer_id": layer_id,
            "token_start": 0,
            "token_count": int(seq_count),
            "m_tile_idx": 0,
            "n_tile_idx": 0,
            "k_tile_idx": 0,
            "dtype_summary": {
                "act": "int8",
                "wt": "int8",
                "acc_expected": "int32",
                "act_steps": "int8",
                "wt_steps": "int8",
                "acc_expected_lane": "int32",
            },
            "shape_summary": {
                "act": list(act_tile.shape),
                "wt": list(wt_tile.shape),
                "acc_expected": list(acc_tile.shape),
                "act_steps": list(act_steps.shape),
                "wt_steps": list(wt_steps.shape),
                "acc_expected_lane": list(acc_lane.shape),
            },
            "lane_packing": "row_major_mxn",
            "active_lane_count": active_lane_count,
            "k_total": int(k_slice),
            "full_k_total": int(k_total),
            "m_count": int(m_count),
            "n_count": int(n_count),
            "rtl_fixture_base": str(fixture_base.as_posix()),
            "purpose": "rtl_smoke",
        }
    )


def build_requant_case(
    *,
    runtime_mode: str,
    layer_id: int,
    op_name: str,
    debug_dict: dict[str, Any],
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    accum = np.asarray(debug_dict["accum_int32"], dtype=np.int32)
    output_fp = np.asarray(debug_dict["output_fp"], dtype=np.float32)
    input_scale = float(np.asarray(debug_dict["input_scale"], dtype=np.float32)[0])
    weight_scale = float(np.asarray(debug_dict["weight_scale"], dtype=np.float32)[0])

    out_q_full, out_scale = bridge.quantize_int8(output_fp)
    multiplier_fp = (input_scale * weight_scale) / float(out_scale)
    multiplier_q16 = q16_16_from_float(multiplier_fp)
    scale_vec = np.full((SCALE_VECTOR_ELEMS,), multiplier_q16, dtype=np.uint32)

    seq_count, output_dim = accum.shape
    m_count = min(M_TILE, seq_count)
    n_count = min(N_TILE, output_dim)
    acc_tile = accum[:m_count, :n_count]
    out_tile = np.asarray(out_q_full[:m_count, :n_count], dtype=np.int8)

    acc_lane = np.zeros((GEMM_LANES,), dtype=np.int32)
    out_lane = np.zeros((GEMM_LANES,), dtype=np.int8)
    active_lane_count = m_count * n_count

    for m_local in range(m_count):
        for n_local in range(n_count):
            lane = (m_local * N_TILE) + n_local
            acc_lane[lane] = np.int32(acc_tile[m_local, n_local])
            out_lane[lane] = np.int8(out_tile[m_local, n_local])

    case_id = f"{PHASE3}_{runtime_mode}_layer{layer_id}_{op_name}_requant_m0_n0"
    case_path = output_root / PHASE3 / f"{case_id}.npz"
    ensure_dir(case_path.parent)

    np.savez(
        case_path,
        acc=acc_lane,
        scale=scale_vec,
        out_expected=out_lane,
        elem_count=np.asarray([active_lane_count], dtype=np.int32),
        output_scale=np.asarray([out_scale], dtype=np.float32),
        requant_multiplier_fp=np.asarray([multiplier_fp], dtype=np.float32),
    )

    rtl_root = output_root / PHASE3 / "rtl"
    fixture_base = rtl_root / case_id
    write_hex_memh(
        fixture_base.with_suffix(".meta.memh"),
        np.asarray([active_lane_count], dtype=np.uint32),
        bits=32,
        signed=False,
    )
    write_hex_memh(
        fixture_base.with_suffix(".acc.memh"),
        acc_lane,
        bits=32,
        signed=True,
    )
    write_hex_memh(
        fixture_base.with_suffix(".scale.memh"),
        scale_vec,
        bits=32,
        signed=False,
    )
    write_hex_memh(
        fixture_base.with_suffix(".out_expected.memh"),
        out_lane,
        bits=8,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".acc_packed.memh"),
        acc_lane,
        bits_per_lane=32,
        signed=True,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".scale_packed.memh"),
        scale_vec,
        bits_per_lane=32,
        signed=False,
    )
    write_packed_lane_memh(
        fixture_base.with_suffix(".out_expected_packed.memh"),
        out_lane,
        bits_per_lane=8,
        signed=True,
    )

    manifest_entries.append(
        {
            "case_id": case_id,
            "phase": PHASE3,
            "block": "requantize_unit",
            "path": str(case_path.as_posix()),
            "runtime_mode": runtime_mode,
            "layer_id": layer_id,
            "token_start": 0,
            "token_count": int(seq_count),
            "m_tile_idx": 0,
            "n_tile_idx": 0,
            "k_tile_idx": 0,
            "dtype_summary": {
                "acc": "int32",
                "scale": "uint32_q16_16",
                "out_expected": "int8",
            },
            "shape_summary": {
                "acc": list(acc_lane.shape),
                "scale": list(scale_vec.shape),
                "out_expected": list(out_lane.shape),
            },
            "lane_packing": "row_major_mxn",
            "active_lane_count": active_lane_count,
            "output_scale": float(out_scale),
            "requant_multiplier_fp": multiplier_fp,
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "rtl_fixture_base": str(fixture_base.as_posix()),
        }
    )


def export_phase3_for_tokens(
    *,
    runtime_mode: str,
    token_ids: list[int],
    target_layer: int,
    weights: ref.TinyLlamaWeights,
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    cos, sin = ref.build_rope_cache(
        seq_len=len(token_ids),
        head_dim=weights.cfg["head_dim"],
        rope_theta=weights.cfg["rope_theta"],
    )
    layer_input = collect_layer_input(
        weights=weights,
        token_ids=token_ids,
        target_layer=target_layer,
        cos=cos,
        sin=sin,
    )
    _, debug = bridge.mixed_precision_decoder_layer(
        layer_input,
        weights.layer(target_layer),
        weights.cfg,
        cos,
        sin,
        capture_arrays=True,
    )

    for op_name, section_key, debug_key in PHASE3_OPS:
        debug_dict = debug[section_key][debug_key]
        build_gemm_case(
            runtime_mode=runtime_mode,
            layer_id=target_layer,
            op_name=op_name,
            debug_dict=debug_dict,
            output_root=output_root,
            manifest_entries=manifest_entries,
        )
        if runtime_mode == "prefill" and op_name == "q_proj":
            build_gemm_smoke_case(
                runtime_mode=runtime_mode,
                layer_id=target_layer,
                op_name=op_name,
                debug_dict=debug_dict,
                output_root=output_root,
                manifest_entries=manifest_entries,
            )
        build_requant_case(
            runtime_mode=runtime_mode,
            layer_id=target_layer,
            op_name=op_name,
            debug_dict=debug_dict,
            output_root=output_root,
            manifest_entries=manifest_entries,
        )


def export_phase3_cases(
    *,
    weights_path: str,
    layer: int,
    prefill_token_ids: list[int],
    decode_token_ids: list[int],
    output_dir: str,
) -> Path:
    output_root = Path(output_dir)
    ensure_dir(output_root / PHASE3)

    weights = ref.load_weights(weights_path, compute_dtype=np.float32, cache_arrays=False)
    if layer < 0 or layer >= weights.cfg["num_hidden_layers"]:
        raise ValueError(
            f"--layer must be in [0, {weights.cfg['num_hidden_layers'] - 1}]"
        )

    manifest_entries: list[dict[str, Any]] = []

    export_phase3_for_tokens(
        runtime_mode="prefill",
        token_ids=prefill_token_ids,
        target_layer=layer,
        weights=weights,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )
    export_phase3_for_tokens(
        runtime_mode="decode",
        token_ids=decode_token_ids,
        target_layer=layer,
        weights=weights,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )

    manifest = {
        "trace_format_version": 1,
        "phase": PHASE3,
        "model_id": weights.cfg["model_id"],
        "weights_path": weights_path,
        "layer": layer,
        "prefill_token_ids": prefill_token_ids,
        "decode_token_ids": decode_token_ids,
        "lane_packing": "row_major_mxn",
        "tile_constants": {
            "M_TILE": M_TILE,
            "N_TILE": N_TILE,
            "GEMM_LANES": GEMM_LANES,
            "SCALE_VECTOR_ELEMS": SCALE_VECTOR_ELEMS,
        },
        "cases": manifest_entries,
    }

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TinyLlama golden traces for FPGA verification."
    )
    parser.add_argument(
        "--phase",
        choices=[PHASE3],
        default=PHASE3,
        help="Trace-export phase. Only phase3 is implemented in the first pass.",
    )
    parser.add_argument(
        "--weights",
        default=ref.DEFAULT_WEIGHTS,
        help="Path to the TinyLlama weights archive.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index for exported Phase 3 cases.",
    )
    parser.add_argument(
        "--prefill-token-ids",
        default=DEFAULT_PREFILL_TOKEN_IDS,
        help="Comma-separated token ids for the prefill case.",
    )
    parser.add_argument(
        "--decode-token-ids",
        default=DEFAULT_DECODE_TOKEN_IDS,
        help="Comma-separated token ids for the decode case.",
    )
    parser.add_argument(
        "--output-dir",
        default="sim/golden_traces",
        help="Export root directory.",
    )
    args = parser.parse_args()

    manifest_path = export_phase3_cases(
        weights_path=args.weights,
        layer=args.layer,
        prefill_token_ids=parse_token_ids(args.prefill_token_ids),
        decode_token_ids=parse_token_ids(args.decode_token_ids),
        output_dir=args.output_dir,
    )
    print(f"Wrote Phase 3 golden traces to {manifest_path.parent}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
