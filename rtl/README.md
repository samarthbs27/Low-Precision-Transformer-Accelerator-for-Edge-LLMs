# RTL — Low-Precision Transformer Accelerator

SystemVerilog implementation of a tiled INT8 matrix-vector engine: `y = W * x`.  
Target: Xilinx Alveo U55C. Simulator: Icarus Verilog (`iverilog -g2012`).

---

## Files

| File | Owner | Description |
|------|-------|-------------|
| `control_fsm.sv` | Samarth | Tiling and dataflow control FSM |
| `top.sv` | Samarth | Top-level integration: FSM + MAC array + BRAM stubs |
| `mac_array.sv` | Rijul | 8-lane parallel MAC array with accumulators |
| `mac_unit.sv` | Rijul | Single combinational INT8×INT8→INT32 MAC unit |
| `tb_control_fsm.sv` | Samarth | FSM-only testbench (7 checks, no external files needed) |
| `tb_top.sv` | Samarth | Integration testbench (4 checks, no external files needed) |
| `tb_mac_array.sv` | Rijul | MAC array testbench (requires test vectors from Satyarth) |

---

## Module Overview

### `mac_unit.sv`
Purely combinational. Computes one INT8×INT8 multiply-accumulate:
```
acc_out = acc_in + (a * b)
```
Inputs `a`, `b` are signed 8-bit. Output `acc_out` is signed 32-bit.

### `mac_array.sv`
Instantiates 8 `mac_unit`s in parallel (one per lane). Each lane holds a 32-bit accumulator. Control signals:
- `mac_valid` HIGH → all 8 lanes accumulate this cycle
- `clear_acc` HIGH → all accumulators reset to zero (one-cycle pulse, before each tile)
- `wr_en` HIGH → write all 8 accumulators to the output buffer (one-cycle pulse, after each tile)

### `control_fsm.sv`
Drives the MAC array through a tiled matrix-vector multiply. FSM states:

```
IDLE → LOAD → COMPUTE → WRITE → LOAD → ... → DONE → IDLE
```

- **IDLE**: waits for `start` pulse
- **LOAD**: presents BRAM read addresses, fires `clear_acc`, waits `LOAD_LAT` cycles
- **COMPUTE**: asserts `mac_valid` every cycle, increments `k_idx` (0 → K−1)
- **WRITE**: asserts `wr_en` for one cycle to commit tile results
- **DONE**: asserts `done` for one cycle, returns to IDLE

Parameters: `N` (output dim), `K` (input dim), `T` (tile size = lanes), `LOAD_LAT` (BRAM latency).  
Defaults: N=64, K=64, T=8, LOAD_LAT=2 → 8 tiles × 64 cycles = 512 MAC cycles per run.

### `top.sv`
Wires `control_fsm` → `mac_array` with behavioral register arrays as BRAM stubs.  
BRAM reads are **combinational** — the FSM's registered address outputs provide the required one-cycle pipeline alignment.  
Exposes write ports (`x_wr_en/addr/data`, `w_wr_en/row/col/data`) for the testbench to preload memories.

---

## Running the Testbenches

All commands run from the **project root** (one level above `rtl/`).  
Replace `<build-output>.vvp` below with any local simulator output filename you want to use.

### 1. FSM testbench — `tb_control_fsm.sv`

Tests the control FSM in isolation. No external files needed.

```bash
iverilog -g2012 -o <build-output>.vvp rtl/control_fsm.sv rtl/tb_control_fsm.sv
vvp <build-output>.vvp
```

Checks:
1. `mac_valid` LOW in IDLE before start
2. `mac_valid` LOW during WRITE
3. `clear_acc` pulses exactly 8 times (once per tile)
4. `wr_en` pulses exactly 8 times (once per tile)
5. `wr_addr` increments 0 → 7 correctly
6. `done` asserts after all tiles complete
7. `done` de-asserts next cycle (FSM returns to IDLE)

Expected output:
```
PASS check 1: mac_valid LOW in IDLE
PASS check 5: tile 0 wr_addr=0
...
PASS check 3: clear_acc pulsed 8 times
PASS check 4: wr_en pulsed 8 times
PASS check 1b: mac_valid asserted 512 cycles
PASS check 6: done asserted after 538 cycles
PASS check 7: done de-asserted, FSM back to IDLE
```

---

### 2. Integration testbench — `tb_top.sv`

Tests `top.sv` (FSM + MAC array end-to-end). No external files needed.

```bash
iverilog -g2012 -o <build-output>.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/control_fsm.sv rtl/top.sv rtl/tb_top.sv
vvp <build-output>.vvp
```

Checks:
1. **All-ones**: `x=[1…1]`, `W=all 1s` → `y[i]=64` for all i
2. **Identity**: `W=I`, `x[k]=k` → `y[i]=i`
3. `done` de-asserts after one cycle
4. Second `start` pulse produces identical results (no stale state)

Expected output:
```
PASS CHECK_1: all 64 outputs == 64
PASS CHECK_2: identity test — all y[i] == x[i]
PASS CHECK_3: done de-asserted correctly
PASS CHECK_4: second run matches
ALL CHECKS PASSED — 0 errors
```

---

### 3. MAC array testbench — `tb_mac_array.sv` *(requires Satyarth's test vectors)*

```bash
iverilog -g2012 -o <build-output>.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/tb_mac_array.sv
vvp <build-output>.vvp
```

Requires `sim/x.txt`, `sim/w.txt`, `sim/expected.txt`. These are already generated in `sim/`.  
To regenerate: run `py -3.13 model/gen_test_vectors.py` from the project root.

---

## Known Icarus Warnings (benign)

These appear on every compile and can be ignored:

```
warning: Static variable initialization requires explicit lifetime in this context.
sorry: Case unique/unique0 qualities are ignored.
```

The `unique case` qualifier is a synthesis hint — not supported by Icarus but has no effect on simulation correctness.
