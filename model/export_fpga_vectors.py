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
PHASE4 = "phase4"

M_TILE = 16
N_TILE = 32
GEMM_LANES = 512
SCALE_VECTOR_ELEMS = 16
Q16_16_FRAC_BITS = 16
K_TILE = 64
HEAD_DIM = 64
ROPE_CHUNK_TOKENS = 8
ROPE_HALF_DIM = HEAD_DIM // 2
SCORE_K_TILE = 64
SCORE_ROWS_PER_CHUNK = 8
MASK_NEG_INF = np.int32(-1000000000)

DEFAULT_PREFILL_TOKEN_IDS = ",".join(str(i) for i in range(1, 17))
DEFAULT_DECODE_TOKEN_IDS = "1"
DEFAULT_PHASE4_DECODE_CONTEXT_TOKEN_IDS = DEFAULT_PREFILL_TOKEN_IDS

ROPE_COS_ROM_PATH = Path("rtl/compute/rope_cos_rom.memh")
ROPE_SIN_ROM_PATH = Path("rtl/compute/rope_sin_rom.memh")


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


def q16_16_signed_from_float(value: float) -> np.int32:
    scaled = int(round(float(value) * float(1 << Q16_16_FRAC_BITS)))
    scaled = max(-(1 << 31), min(scaled, (1 << 31) - 1))
    return np.int32(scaled)


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


def build_rope_rom_tables(
    *,
    rope_theta: float,
    source_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cos_fp, sin_fp = ref.build_rope_cache(
        seq_len=2048,
        head_dim=HEAD_DIM,
        rope_theta=rope_theta,
    )
    cos_q16 = np.vectorize(q16_16_signed_from_float, otypes=[np.int32])(cos_fp)
    sin_q16 = np.vectorize(q16_16_signed_from_float, otypes=[np.int32])(sin_fp)

    write_hex_memh(
        source_root / ROPE_COS_ROM_PATH,
        cos_q16[:, :ROPE_HALF_DIM].reshape(-1),
        bits=32,
        signed=True,
    )
    write_hex_memh(
        source_root / ROPE_SIN_ROM_PATH,
        sin_q16[:, :ROPE_HALF_DIM].reshape(-1),
        bits=32,
        signed=True,
    )
    return cos_q16, sin_q16


def round_q16_16_sum_to_int8(sum_term: int) -> np.int8:
    abs_sum = -sum_term if sum_term < 0 else sum_term
    quotient_mag = abs_sum >> Q16_16_FRAC_BITS
    remainder_bits = abs_sum & ((1 << Q16_16_FRAC_BITS) - 1)
    rounded_mag = quotient_mag

    if remainder_bits > (1 << (Q16_16_FRAC_BITS - 1)):
        rounded_mag = quotient_mag + 1
    elif remainder_bits == (1 << (Q16_16_FRAC_BITS - 1)) and (quotient_mag & 1):
        rounded_mag = quotient_mag + 1

    rounded_signed = -rounded_mag if sum_term < 0 else rounded_mag
    rounded_signed = max(-127, min(127, rounded_signed))
    return np.int8(rounded_signed)


def apply_rope_int8_slice(
    tile_q: np.ndarray,
    tile_k: np.ndarray,
    cos_q16: np.ndarray,
    sin_q16: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    token_count = tile_q.shape[0]
    q_rot = np.zeros_like(tile_q, dtype=np.int8)
    k_rot = np.zeros_like(tile_k, dtype=np.int8)

    for token_idx in range(token_count):
        for dim_idx in range(HEAD_DIM):
            if dim_idx < ROPE_HALF_DIM:
                pair_dim = dim_idx + ROPE_HALF_DIM
                lower_half = True
            else:
                pair_dim = dim_idx - ROPE_HALF_DIM
                lower_half = False

            cos_val = int(cos_q16[token_idx, dim_idx])
            sin_val = int(sin_q16[token_idx, dim_idx])

            q_curr = int(tile_q[token_idx, dim_idx])
            q_pair = int(tile_q[token_idx, pair_dim])
            q_sum = (q_curr * cos_val) - (q_pair * sin_val) if lower_half else (q_curr * cos_val) + (q_pair * sin_val)
            q_rot[token_idx, dim_idx] = round_q16_16_sum_to_int8(q_sum)

            k_curr = int(tile_k[token_idx, dim_idx])
            k_pair = int(tile_k[token_idx, pair_dim])
            k_sum = (k_curr * cos_val) - (k_pair * sin_val) if lower_half else (k_curr * cos_val) + (k_pair * sin_val)
            k_rot[token_idx, dim_idx] = round_q16_16_sum_to_int8(k_sum)

    return q_rot, k_rot


def pack_token_major_head_tile(tile: np.ndarray) -> np.ndarray:
    packed = np.zeros((GEMM_LANES,), dtype=np.int8)
    token_count = min(tile.shape[0], ROPE_CHUNK_TOKENS)
    for token_local in range(token_count):
        lane_base = token_local * HEAD_DIM
        packed[lane_base:lane_base + HEAD_DIM] = np.asarray(tile[token_local, :HEAD_DIM], dtype=np.int8)
    return packed


def pack_score_chunk(score_tile: np.ndarray) -> np.ndarray:
    packed = np.zeros((GEMM_LANES,), dtype=np.int32)
    row_count = min(score_tile.shape[0], SCORE_ROWS_PER_CHUNK)
    col_count = min(score_tile.shape[1], SCORE_K_TILE)
    for row_local in range(row_count):
        lane_base = row_local * SCORE_K_TILE
        packed[lane_base:lane_base + col_count] = np.asarray(score_tile[row_local, :col_count], dtype=np.int32)
    return packed


def build_rope_case(
    *,
    runtime_mode: str,
    layer_id: int,
    q_head_id: int,
    kv_head_id: int,
    token_base: int,
    q_in_tile: np.ndarray,
    k_in_tile: np.ndarray,
    q_out_tile: np.ndarray,
    k_out_tile: np.ndarray,
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    token_count = q_in_tile.shape[0]
    q_in_lane = pack_token_major_head_tile(q_in_tile)
    k_in_lane = pack_token_major_head_tile(k_in_tile)
    q_out_lane = pack_token_major_head_tile(q_out_tile)
    k_out_lane = pack_token_major_head_tile(k_out_tile)

    case_id = f"{PHASE4}_{runtime_mode}_layer{layer_id}_rope_q{q_head_id}_kv{kv_head_id}_t{token_base}"
    case_path = output_root / PHASE4 / f"{case_id}.npz"
    ensure_dir(case_path.parent)

    np.savez(
        case_path,
        q_in=q_in_tile,
        k_in=k_in_tile,
        q_out_expected=q_out_tile,
        k_out_expected=k_out_tile,
        token_base=np.asarray([token_base], dtype=np.int32),
        token_count=np.asarray([token_count], dtype=np.int32),
        q_head_id=np.asarray([q_head_id], dtype=np.int32),
        kv_head_id=np.asarray([kv_head_id], dtype=np.int32),
    )

    rtl_root = output_root / PHASE4 / "rtl"
    fixture_base = rtl_root / case_id
    write_hex_memh(
        fixture_base.with_suffix(".meta.memh"),
        np.asarray([token_base, token_count, q_head_id, kv_head_id], dtype=np.uint32),
        bits=32,
        signed=False,
    )
    write_packed_lane_memh(fixture_base.with_suffix(".q_in_packed.memh"), q_in_lane, bits_per_lane=8, signed=True)
    write_packed_lane_memh(fixture_base.with_suffix(".k_in_packed.memh"), k_in_lane, bits_per_lane=8, signed=True)
    write_packed_lane_memh(fixture_base.with_suffix(".q_out_expected_packed.memh"), q_out_lane, bits_per_lane=8, signed=True)
    write_packed_lane_memh(fixture_base.with_suffix(".k_out_expected_packed.memh"), k_out_lane, bits_per_lane=8, signed=True)

    manifest_entries.append(
        {
            "case_id": case_id,
            "phase": PHASE4,
            "block": "rope_unit",
            "path": str(case_path.as_posix()),
            "runtime_mode": runtime_mode,
            "layer_id": layer_id,
            "token_base": token_base,
            "token_count": token_count,
            "q_head_id": q_head_id,
            "kv_head_id": kv_head_id,
            "dtype_summary": {
                "q_in": "int8",
                "k_in": "int8",
                "q_out_expected": "int8",
                "k_out_expected": "int8",
            },
            "shape_summary": {
                "q_in": list(q_in_tile.shape),
                "k_in": list(k_in_tile.shape),
                "q_out_expected": list(q_out_tile.shape),
                "k_out_expected": list(k_out_tile.shape),
            },
            "lane_packing": "token_major_head64",
            "rtl_fixture_base": str(fixture_base.as_posix()),
        }
    )


def build_causal_mask_case(
    *,
    runtime_mode: str,
    layer_id: int,
    q_head_id: int,
    kv_head_id: int,
    query_pos_base: int,
    key_pos_base: int,
    query_row_count: int,
    key_col_count: int,
    score_in_tile: np.ndarray,
    score_out_tile: np.ndarray,
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    score_in_lane = pack_score_chunk(score_in_tile)
    score_out_lane = pack_score_chunk(score_out_tile)

    case_id = (
        f"{PHASE4}_{runtime_mode}_layer{layer_id}_causal_mask_q{q_head_id}_"
        f"kv{kv_head_id}_qb{query_pos_base}_kb{key_pos_base}"
    )
    case_path = output_root / PHASE4 / f"{case_id}.npz"
    ensure_dir(case_path.parent)

    np.savez(
        case_path,
        score_in=score_in_tile,
        score_out_expected=score_out_tile,
        query_pos_base=np.asarray([query_pos_base], dtype=np.int32),
        key_pos_base=np.asarray([key_pos_base], dtype=np.int32),
        query_row_count=np.asarray([query_row_count], dtype=np.int32),
        key_col_count=np.asarray([key_col_count], dtype=np.int32),
        q_head_id=np.asarray([q_head_id], dtype=np.int32),
        kv_head_id=np.asarray([kv_head_id], dtype=np.int32),
    )

    rtl_root = output_root / PHASE4 / "rtl"
    fixture_base = rtl_root / case_id
    write_hex_memh(
        fixture_base.with_suffix(".meta.memh"),
        np.asarray(
            [query_pos_base, key_pos_base, query_row_count, key_col_count, q_head_id, kv_head_id],
            dtype=np.uint32,
        ),
        bits=32,
        signed=False,
    )
    write_packed_lane_memh(fixture_base.with_suffix(".score_in_packed.memh"), score_in_lane, bits_per_lane=32, signed=True)
    write_packed_lane_memh(fixture_base.with_suffix(".score_out_expected_packed.memh"), score_out_lane, bits_per_lane=32, signed=True)

    manifest_entries.append(
        {
            "case_id": case_id,
            "phase": PHASE4,
            "block": "causal_mask_unit",
            "path": str(case_path.as_posix()),
            "runtime_mode": runtime_mode,
            "layer_id": layer_id,
            "query_pos_base": query_pos_base,
            "key_pos_base": key_pos_base,
            "query_row_count": query_row_count,
            "key_col_count": key_col_count,
            "q_head_id": q_head_id,
            "kv_head_id": kv_head_id,
            "dtype_summary": {
                "score_in": "int32",
                "score_out_expected": "int32",
            },
            "shape_summary": {
                "score_in": list(score_in_tile.shape),
                "score_out_expected": list(score_out_tile.shape),
            },
            "lane_packing": "score_chunk_row_major",
            "mask_neg_inf": int(MASK_NEG_INF),
            "rtl_fixture_base": str(fixture_base.as_posix()),
        }
    )


def collect_phase4_attention_tensors(
    *,
    token_ids: list[int],
    target_layer: int,
    weights: ref.TinyLlamaWeights,
    cos_rom_q16: np.ndarray,
    sin_rom_q16: np.ndarray,
) -> dict[str, np.ndarray]:
    cos_local, sin_local = ref.build_rope_cache(
        seq_len=len(token_ids),
        head_dim=weights.cfg["head_dim"],
        rope_theta=weights.cfg["rope_theta"],
    )
    layer_input = collect_layer_input(
        weights=weights,
        token_ids=token_ids,
        target_layer=target_layer,
        cos=cos_local,
        sin=sin_local,
    )
    _, debug = bridge.mixed_precision_decoder_layer(
        layer_input,
        weights.layer(target_layer),
        weights.cfg,
        cos_local,
        sin_local,
        capture_arrays=True,
    )

    q_q, _ = bridge.quantize_int8(np.asarray(debug["attn"]["q_proj_fp"], dtype=np.float32))
    k_q, _ = bridge.quantize_int8(np.asarray(debug["attn"]["k_proj_fp"], dtype=np.float32))

    seq_count = len(token_ids)
    q_heads = q_q.reshape(seq_count, weights.cfg["num_attention_heads"], HEAD_DIM).transpose(1, 0, 2)
    k_heads = k_q.reshape(seq_count, weights.cfg["num_key_value_heads"], HEAD_DIM).transpose(1, 0, 2)

    q_rope = np.zeros_like(q_heads, dtype=np.int8)
    k_rope = np.zeros_like(k_heads, dtype=np.int8)

    for q_head_id in range(weights.cfg["num_attention_heads"]):
        kv_head_id = q_head_id // weights.cfg["num_key_value_groups"]
        q_rot_tile, _ = apply_rope_int8_slice(
            q_heads[q_head_id],
            k_heads[kv_head_id],
            cos_rom_q16[:seq_count, :HEAD_DIM],
            sin_rom_q16[:seq_count, :HEAD_DIM],
        )
        q_rope[q_head_id] = q_rot_tile

    for kv_head_id in range(weights.cfg["num_key_value_heads"]):
        _, k_rot_tile = apply_rope_int8_slice(
            q_heads[kv_head_id * weights.cfg["num_key_value_groups"]],
            k_heads[kv_head_id],
            cos_rom_q16[:seq_count, :HEAD_DIM],
            sin_rom_q16[:seq_count, :HEAD_DIM],
        )
        k_rope[kv_head_id] = k_rot_tile

    return {
        "q_heads_q": q_heads,
        "k_heads_q": k_heads,
        "q_heads_rope_q": q_rope,
        "k_heads_rope_q": k_rope,
    }


def mask_score_tile(
    *,
    runtime_mode: str,
    score_in_tile: np.ndarray,
    query_pos_base: int,
    key_pos_base: int,
    query_row_count: int,
    key_col_count: int,
) -> np.ndarray:
    score_out_tile = np.zeros((SCORE_ROWS_PER_CHUNK, SCORE_K_TILE), dtype=np.int32)

    for row_local in range(SCORE_ROWS_PER_CHUNK):
        if row_local >= query_row_count:
            continue

        query_pos = query_pos_base + row_local
        for col_local in range(SCORE_K_TILE):
            key_pos = key_pos_base + col_local
            allow = False
            if runtime_mode in ("prefill", "decode"):
                allow = (col_local < key_col_count) and (key_pos <= query_pos)

            if allow:
                score_out_tile[row_local, col_local] = np.int32(score_in_tile[row_local, col_local])
            else:
                score_out_tile[row_local, col_local] = MASK_NEG_INF

    return score_out_tile


def export_phase4_for_tokens(
    *,
    runtime_mode: str,
    token_ids: list[int],
    target_layer: int,
    weights: ref.TinyLlamaWeights,
    cos_rom_q16: np.ndarray,
    sin_rom_q16: np.ndarray,
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    tensors = collect_phase4_attention_tensors(
        token_ids=token_ids,
        target_layer=target_layer,
        weights=weights,
        cos_rom_q16=cos_rom_q16,
        sin_rom_q16=sin_rom_q16,
    )

    if runtime_mode == "prefill":
        token_base = 0
        token_count = min(ROPE_CHUNK_TOKENS, len(token_ids))
        q_head_id = 0
        kv_head_id = 0
        q_in_tile = np.asarray(tensors["q_heads_q"][q_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        k_in_tile = np.asarray(tensors["k_heads_q"][kv_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        q_out_tile = np.asarray(tensors["q_heads_rope_q"][q_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        k_out_tile = np.asarray(tensors["k_heads_rope_q"][kv_head_id, token_base:token_base + token_count, :], dtype=np.int8)

        query_pos_base = max(0, len(token_ids) - SCORE_ROWS_PER_CHUNK)
        key_pos_base = 0
        query_row_count = min(SCORE_ROWS_PER_CHUNK, len(token_ids))
        key_col_count = min(SCORE_K_TILE, len(token_ids))
    else:
        token_base = max(0, len(token_ids) - 1)
        token_count = 1
        q_head_id = 0
        kv_head_id = 0
        q_in_tile = np.asarray(tensors["q_heads_q"][q_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        k_in_tile = np.asarray(tensors["k_heads_q"][kv_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        q_out_tile = np.asarray(tensors["q_heads_rope_q"][q_head_id, token_base:token_base + token_count, :], dtype=np.int8)
        k_out_tile = np.asarray(tensors["k_heads_rope_q"][kv_head_id, token_base:token_base + token_count, :], dtype=np.int8)

        query_pos_base = token_base
        key_pos_base = 0
        query_row_count = 1
        key_col_count = min(SCORE_K_TILE, len(token_ids))

    build_rope_case(
        runtime_mode=runtime_mode,
        layer_id=target_layer,
        q_head_id=q_head_id,
        kv_head_id=kv_head_id,
        token_base=token_base,
        q_in_tile=q_in_tile,
        k_in_tile=k_in_tile,
        q_out_tile=q_out_tile,
        k_out_tile=k_out_tile,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )

    q_rot_full = np.asarray(tensors["q_heads_rope_q"][q_head_id], dtype=np.int8)
    k_rot_full = np.asarray(tensors["k_heads_rope_q"][kv_head_id], dtype=np.int8)
    score_active = q_rot_full[query_pos_base:query_pos_base + query_row_count, :].astype(np.int32) @ k_rot_full[key_pos_base:key_pos_base + key_col_count, :].T.astype(np.int32)
    score_in_tile = np.zeros((SCORE_ROWS_PER_CHUNK, SCORE_K_TILE), dtype=np.int32)
    score_in_tile[:query_row_count, :key_col_count] = score_active
    score_out_tile = mask_score_tile(
        runtime_mode=runtime_mode,
        score_in_tile=score_in_tile,
        query_pos_base=query_pos_base,
        key_pos_base=key_pos_base,
        query_row_count=query_row_count,
        key_col_count=key_col_count,
    )

    build_causal_mask_case(
        runtime_mode=runtime_mode,
        layer_id=target_layer,
        q_head_id=q_head_id,
        kv_head_id=kv_head_id,
        query_pos_base=query_pos_base,
        key_pos_base=key_pos_base,
        query_row_count=query_row_count,
        key_col_count=key_col_count,
        score_in_tile=score_in_tile,
        score_out_tile=score_out_tile,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )


def export_phase4_cases(
    *,
    weights_path: str,
    layer: int,
    prefill_token_ids: list[int],
    decode_context_token_ids: list[int],
    output_dir: str,
    source_root: Path,
) -> Path:
    output_root = Path(output_dir)
    ensure_dir(output_root / PHASE4)

    weights = ref.load_weights(weights_path, compute_dtype=np.float32, cache_arrays=False)
    if layer < 0 or layer >= weights.cfg["num_hidden_layers"]:
        raise ValueError(
            f"--layer must be in [0, {weights.cfg['num_hidden_layers'] - 1}]"
        )

    cos_rom_q16, sin_rom_q16 = build_rope_rom_tables(
        rope_theta=weights.cfg["rope_theta"],
        source_root=source_root,
    )

    manifest_entries: list[dict[str, Any]] = []

    export_phase4_for_tokens(
        runtime_mode="prefill",
        token_ids=prefill_token_ids,
        target_layer=layer,
        weights=weights,
        cos_rom_q16=cos_rom_q16,
        sin_rom_q16=sin_rom_q16,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )
    export_phase4_for_tokens(
        runtime_mode="decode",
        token_ids=decode_context_token_ids,
        target_layer=layer,
        weights=weights,
        cos_rom_q16=cos_rom_q16,
        sin_rom_q16=sin_rom_q16,
        output_root=output_root,
        manifest_entries=manifest_entries,
    )

    manifest = {
        "trace_format_version": 1,
        "phase": PHASE4,
        "model_id": weights.cfg["model_id"],
        "weights_path": weights_path,
        "layer": layer,
        "prefill_token_ids": prefill_token_ids,
        "decode_context_token_ids": decode_context_token_ids,
        "rope_rom_cos_path": str((source_root / ROPE_COS_ROM_PATH).as_posix()),
        "rope_rom_sin_path": str((source_root / ROPE_SIN_ROM_PATH).as_posix()),
        "tile_constants": {
            "ROPE_CHUNK_TOKENS": ROPE_CHUNK_TOKENS,
            "HEAD_DIM": HEAD_DIM,
            "SCORE_ROWS_PER_CHUNK": SCORE_ROWS_PER_CHUNK,
            "SCORE_K_TILE": SCORE_K_TILE,
            "MASK_NEG_INF": int(MASK_NEG_INF),
        },
        "cases": manifest_entries,
    }

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


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
        choices=[PHASE3, PHASE4],
        default=PHASE3,
        help="Trace-export phase.",
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

    prefill_token_ids = parse_token_ids(args.prefill_token_ids)
    decode_token_ids_arg = args.decode_token_ids
    if (args.phase == PHASE4) and (decode_token_ids_arg == DEFAULT_DECODE_TOKEN_IDS):
        decode_token_ids_arg = DEFAULT_PHASE4_DECODE_CONTEXT_TOKEN_IDS
    decode_token_ids = parse_token_ids(decode_token_ids_arg)

    if args.phase == PHASE3:
        manifest_path = export_phase3_cases(
            weights_path=args.weights,
            layer=args.layer,
            prefill_token_ids=prefill_token_ids,
            decode_token_ids=decode_token_ids,
            output_dir=args.output_dir,
        )
    else:
        manifest_path = export_phase4_cases(
            weights_path=args.weights,
            layer=args.layer,
            prefill_token_ids=prefill_token_ids,
            decode_context_token_ids=decode_token_ids,
            output_dir=args.output_dir,
            source_root=Path(__file__).resolve().parent.parent,
        )

    print(f"Wrote {args.phase} golden traces to {manifest_path.parent}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
