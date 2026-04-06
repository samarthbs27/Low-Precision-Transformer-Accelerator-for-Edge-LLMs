import os
import numpy as np


def relu(x):
    return np.maximum(0, x)


def quantize_int8(tensor):
    max_abs = np.max(np.abs(tensor))
    scale = max_abs / 127.0 if max_abs != 0 else 1.0
    q = np.round(tensor / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize(int32_val, s1, s2):
    return int32_val.astype(np.float32) * (s1 * s2)


def save_txt(filename, arr):
    np.savetxt(filename, arr.reshape(-1), fmt="%d")


def run_ffn(K=64, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(0)

    # ---------------- FLOAT MODEL ----------------
    x = np.random.randn(K).astype(np.float32)
    W1 = np.random.randn(K, K).astype(np.float32)
    W2 = np.random.randn(K, K).astype(np.float32)

    y1 = W1 @ x
    y2 = relu(y1)
    y3 = W2 @ y2

    # ---------------- QUANTIZATION ----------------
    x_q, sx = quantize_int8(x)
    W1_q, sW1 = quantize_int8(W1)
    W2_q, sW2 = quantize_int8(W2)

    # ---------------- LAYER 1 ----------------
    y1_int = W1_q.astype(np.int32) @ x_q.astype(np.int32)
    y1_deq = dequantize(y1_int, sW1, sx)
    y2_deq = relu(y1_deq)

    # ---------------- RE-QUANTIZE ----------------
    y2_q, sy2 = quantize_int8(y2_deq)

    # ---------------- LAYER 2 ----------------
    y3_int = W2_q.astype(np.int32) @ y2_q.astype(np.int32)
    y3_deq = dequantize(y3_int, sW2, sy2)

    # ---------------- ERROR ----------------
    err = np.mean(np.abs(y3 - y3_deq))
    print(f"K={K} | Mean Error: {err:.6f}")

    # ---------------- SAVE ----------------
    save_txt(f"{out_dir}/x_q.txt", x_q)
    save_txt(f"{out_dir}/W1_q.txt", W1_q)
    save_txt(f"{out_dir}/W2_q.txt", W2_q)
    save_txt(f"{out_dir}/y_expected.txt", y3_int)

    # Debug files
    save_txt(f"{out_dir}/y1_int.txt", y1_int)
    save_txt(f"{out_dir}/y2_q.txt", y2_q)

    with open(f"{out_dir}/scales.txt", "w") as f:
        f.write(f"scale_x {sx}\n")
        f.write(f"scale_W1 {sW1}\n")
        f.write(f"scale_W2 {sW2}\n")
        f.write(f"scale_y2 {sy2}\n")


if __name__ == "__main__":
    run_ffn(16, "test_k16")
    run_ffn(64, "test_k64")