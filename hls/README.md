# HLS

This folder holds the HLS side of the TinyLlama accelerator. At the moment, only the shared common headers exist; the HLS kernels themselves will be added later under folders such as `hls/rmsnorm/`, `hls/softmax/`, and `hls/silu/`.

## Files

| File | What it is | Smoke test |
|------|------------|------------|
| `common/fixed_types.hpp` | Shared fixed-point aliases, scalar types, vector helpers, and TinyLlama architectural constants for HLS code. It also provides host-side fallback types when Xilinx HLS headers are not installed. | Include it in a tiny C++ translation unit and run a syntax-only `g++` compile. |
| `common/stream_utils.hpp` | Shared helper functions for clamp, rounding, INT8 requantization, probability-byte quantization, vector copy, and simple stream wrappers. | Include it together with `fixed_types.hpp` in a tiny C++ translation unit and run a syntax-only `g++` compile. |

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

Any local smoke-test scratch files, logs, or generated outputs for HLS bring-up should also live under `sim/`.
