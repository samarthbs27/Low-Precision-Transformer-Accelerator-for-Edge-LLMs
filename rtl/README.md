# RTL

This folder contains two kinds of SystemVerilog code:

- legacy validation RTL at the root of `rtl/`
- new production TinyLlama RTL under subfolders such as `rtl/common/`, `rtl/control/`, and later `rtl/memory/`, `rtl/compute/`, `rtl/nonlinear/`, and `rtl/top/`

The new production path should use the subfolder-based code. The flat root-level files are still useful as references and validation scaffolding, but they are not the final TinyLlama runtime.

## Production Foundation

These are the first production RTL files that now exist.

| File | What it is | Smoke test |
|------|------------|------------|
| `common/tinyllama_pkg.sv` | Shared architectural constants, widths, tiling parameters, and enums for the TinyLlama accelerator. | Syntax-check it together with the dependent common files using the package compile command below. |
| `common/tinyllama_bus_pkg.sv` | Shared packed bus, descriptor, and sideband tag types used across the production RTL. | Syntax-check it together with the dependent common files using the package compile command below. |
| `common/stream_fifo.sv` | Parameterized ready/valid FIFO primitive used at elastic datapath boundaries. | Run `rtl/tb/tb_stream_fifo.sv`. |
| `common/skid_buffer.sv` | Two-entry skid buffer built on top of `stream_fifo.sv` for short backpressure absorption. | Covered by the common syntax check; no standalone testbench yet. |
| `common/descriptor_fifo.sv` | Descriptor-oriented FIFO wrapper used for command and DMA-style queues. | Run `rtl/tb/tb_descriptor_fifo.sv`. |

### Common Package Smoke Test

Run this from the project root:

```powershell
iverilog -g2012 -t null `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/common/stream_fifo.sv `
  rtl/common/skid_buffer.sv `
  rtl/common/descriptor_fifo.sv
```

This is the quickest syntax/integration check for the current production common layer.

When a command in this folder produces simulator output, waveform dumps, or logs, put them under `sim/`.

## Legacy Validation Files

These root-level files are older validation infrastructure. They are still useful, but they do not define the final TinyLlama production architecture.

| File | What it is | Smoke test |
|------|------------|------------|
| `control_fsm.sv` | Legacy tiled matrix-vector control FSM. | Run `tb_control_fsm.sv`. |
| `top.sv` | Legacy top-level wrapper connecting the FSM to the MAC array and BRAM stubs. | Run `tb_top.sv`. |
| `mac_array.sv` | Legacy 8-lane INT8 MAC array with accumulators. | Run `tb_mac_array.sv` or `tb_top.sv`. |
| `mac_unit.sv` | Legacy single-lane combinational INT8 multiply-accumulate unit. | Covered by `tb_mac_array.sv` and `tb_top.sv`. |
| `tb_control_fsm.sv` | Legacy FSM-only testbench. | See command below. |
| `tb_top.sv` | Legacy end-to-end top-level testbench. | See command below. |
| `tb_mac_array.sv` | Legacy MAC-array testbench that uses generated vectors. | See command below. |

### Legacy Smoke Tests

All commands below run from the project root.

FSM-only smoke test:

```powershell
iverilog -g2012 -o sim/tb_control_fsm.vvp rtl/control_fsm.sv rtl/tb_control_fsm.sv
vvp sim/tb_control_fsm.vvp
```

Legacy top-level smoke test:

```powershell
iverilog -g2012 -o sim/tb_top.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/control_fsm.sv rtl/top.sv rtl/tb_top.sv
vvp sim/tb_top.vvp
```

Legacy MAC-array smoke test:

```powershell
iverilog -g2012 -o sim/tb_mac_array.vvp rtl/mac_unit.sv rtl/mac_array.sv rtl/tb_mac_array.sv
vvp sim/tb_mac_array.vvp
```

This test expects local vector files under `sim/`.

## Where To Look Next

- For production testbenches, see [rtl/tb/README.md](/c:/Users/samar/OneDrive/Desktop/Reconfigureable%20Computing/Project/rtl/tb/README.md).
- For HLS common utilities and header-only smoke checks, see [hls/README.md](/c:/Users/samar/OneDrive/Desktop/Reconfigureable%20Computing/Project/hls/README.md).
