import numpy as np
import os

def quantize(x):
    scale = np.max(np.abs(x)) / 127 if np.max(np.abs(x)) != 0 else 1
    q = np.round(x / scale)
    return np.clip(q, -127, 127).astype(np.int8), scale

def save_vector(filename, vec):
    np.savetxt(filename, vec, fmt="%d")

def save_matrix_flat(filename, mat):
    np.savetxt(filename, mat.reshape(-1), fmt="%d")

def generate(K=64, N=64):
    os.makedirs("sim", exist_ok=True)
    np.random.seed(1)

    x = np.random.randn(K).astype(np.float32)
    W = np.random.randn(N, K).astype(np.float32)

    x_q, sx = quantize(x)
    W_q, sW = quantize(W)

    # INT32 accumulation
    y_int = W_q.astype(np.int32) @ x_q.astype(np.int32)

    save_vector("sim/x.txt", x_q)
    save_matrix_flat("sim/w.txt", W_q)
    save_vector("sim/expected.txt", y_int)

    print("Generated sim test vectors.")

if __name__ == "__main__":
    generate()