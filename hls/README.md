# HLS

This folder holds the HLS side of the TinyLlama accelerator.

The current production HLS path includes:

- shared fixed-point/common helpers under `hls/common/`
- a fixed-point RMSNorm kernel under `hls/rmsnorm/`
- a fixed-point softmax kernel under `hls/softmax/`
- a fixed-point SiLU kernel under `hls/silu/`

## Files

| File | What it is | Smoke test |
|------|------------|------------|
| `common/fixed_types.hpp` | Shared fixed-point aliases, scalar types, vector helpers, and TinyLlama architectural constants for HLS code. It also provides host-side fallback types when Xilinx HLS headers are not installed. | Include it in a tiny C++ translation unit and run a syntax-only `g++` compile. |
| `common/stream_utils.hpp` | Shared helper functions for clamp, rounding, INT8 requantization, probability-byte quantization, vector copy, and simple stream wrappers. | Include it together with `fixed_types.hpp` in a tiny C++ translation unit and run a syntax-only `g++` compile. |
| `common/nonlinear_math.hpp` | Shared fixed-point helper math used by the nonlinear kernels, including square-root, exponent, sigmoid, and vector fill utilities. | Covered by the kernel smoke tests below. |
| `common/test_memh.hpp` | Host-side `.memh` parsing helpers used by the C++ testbenches to consume exported golden-trace fixtures. | Covered by the kernel smoke tests below. |
| `rmsnorm/rmsnorm_core_hls.hpp` | Declares the fixed-point RMSNorm HLS core stream contract. | Run the `tb_rmsnorm` command below. |
| `rmsnorm/rmsnorm_core_hls.cpp` | Fixed-point RMSNorm implementation used by the RTL wrapper and host-side testbench. | Run the `tb_rmsnorm` command below. |
| `rmsnorm/tb_rmsnorm.cpp` | Host-side C++ smoke test for the RMSNorm HLS core. | Run the `tb_rmsnorm` command below. |
| `softmax/softmax_core_hls.hpp` | Declares the fixed-point softmax HLS core stream contract. | Run the `tb_softmax` command below. |
| `softmax/softmax_core_hls.cpp` | Fixed-point softmax implementation used by the RTL wrapper and host-side testbench. | Run the `tb_softmax` command below. |
| `softmax/tb_softmax.cpp` | Host-side C++ smoke test for the softmax HLS core. | Run the `tb_softmax` command below. |
| `silu/silu_core_hls.hpp` | Declares the fixed-point SiLU HLS core stream contract. | Run the `tb_silu` command below. |
| `silu/silu_core_hls.cpp` | Fixed-point SiLU implementation used by the RTL wrapper and host-side testbench. | Run the `tb_silu` command below. |
| `silu/tb_silu.cpp` | Host-side C++ smoke test for the SiLU HLS core. | Run the `tb_silu` command below. |

## Header Smoke Test

Run this from the project root:

```powershell
@'
#include "hls/common/fixed_types.hpp"
#include "hls/common/stream_utils.hpp"
int main(){return 0;}
'@ | Set-Content -NoNewline sim/hls_smoke.cpp
g++ -std=c++17 -fsyntax-only -I. sim/hls_smoke.cpp
Remove-Item sim/hls_smoke.cpp
```

If the compile succeeds with no errors, the current HLS common layer is in good shape for host-side syntax checking.

## Kernel Smoke Tests

All commands below run from the project root and place outputs under `sim/`.

### `tb_rmsnorm`

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
g++ -std=c++17 -I. hls/rmsnorm/tb_rmsnorm.cpp hls/rmsnorm/rmsnorm_core_hls.cpp -o sim/tb_rmsnorm.exe
sim/tb_rmsnorm.exe
```

Expected pass string:

```text
PASS: tb_rmsnorm
```

### `tb_softmax`

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
g++ -std=c++17 -I. hls/softmax/tb_softmax.cpp hls/softmax/softmax_core_hls.cpp -o sim/tb_softmax.exe
sim/tb_softmax.exe
```

Expected pass string:

```text
PASS: tb_softmax
```

### `tb_silu`

```powershell
python model/export_fpga_vectors.py --phase phase5 --layer 0 --output-dir sim/golden_traces
g++ -std=c++17 -I. hls/silu/tb_silu.cpp hls/silu/silu_core_hls.cpp -o sim/tb_silu.exe
sim/tb_silu.exe
```

Expected pass string:

```text
PASS: tb_silu
```

Any local smoke-test scratch files, logs, or generated outputs for HLS bring-up should also live under `sim/`.
