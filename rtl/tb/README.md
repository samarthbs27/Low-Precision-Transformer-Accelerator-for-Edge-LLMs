# RTL Testbenches

This folder contains production TinyLlama RTL smoke tests for the new `rtl/common/` building blocks.

## Files

| File | What it tests | Smoke test |
|------|----------------|------------|
| `tb_stream_fifo.sv` | Directed self-check of `rtl/common/stream_fifo.sv`, including push/pop ordering, occupancy tracking, simultaneous push/pop, full-condition backpressure, and blocked-push behavior. | Run the `tb_stream_fifo` command below. |
| `tb_descriptor_fifo.sv` | Directed self-check of `rtl/common/descriptor_fifo.sv`, including push, pop, and occupancy behavior. | Run the `tb_descriptor_fifo` command below. |
| `tb_axi_lite_ctrl_slave.sv` | AXI-Lite plus register-file smoke test covering register writes, register reads, sticky status, clear-on-START behavior, launch mode, and abort behavior. | Run the `tb_axi_lite_ctrl_slave` command below. |
| `tb_host_cmd_status_mgr.sv` | Fake-HBM smoke test for one-beat command fetch from PC30, one-beat terminal status writeback to PC30, and relaunch/error-only command-status behavior. | Run the `tb_host_cmd_status_mgr` command below. |
| `tb_prefill_decode_controller.sv` | Control-path smoke test covering prefill launch, 22-layer iteration, LM-head/token handoff, EOS stop, MAX_TOKENS stop, host-abort stop, and zero-token prefill rejection. | Run the `tb_prefill_decode_controller` command below. |

## Smoke Tests

Run all commands from the project root.

Place all generated simulator outputs and logs under `sim/`.

### `tb_stream_fifo.sv`

```powershell
iverilog -g2012 -o sim/tb_stream_fifo.vvp `
  rtl/common/stream_fifo.sv `
  rtl/tb/tb_stream_fifo.sv
vvp sim/tb_stream_fifo.vvp
```

Expected pass string:

```text
PASS: tb_stream_fifo
```

### `tb_descriptor_fifo.sv`

```powershell
iverilog -g2012 -o sim/tb_descriptor_fifo.vvp `
  rtl/common/stream_fifo.sv `
  rtl/common/descriptor_fifo.sv `
  rtl/tb/tb_descriptor_fifo.sv
vvp sim/tb_descriptor_fifo.vvp
```

Expected pass string:

```text
PASS: tb_descriptor_fifo
```

### `tb_axi_lite_ctrl_slave.sv`

```powershell
iverilog -g2012 -o sim/tb_axi_lite_ctrl_slave.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/control/axi_lite_ctrl_slave.sv `
  rtl/control/kernel_reg_file.sv `
  rtl/tb/tb_axi_lite_ctrl_slave.sv
vvp sim/tb_axi_lite_ctrl_slave.vvp
```

Expected pass string:

```text
PASS: tb_axi_lite_ctrl_slave
```

### `tb_host_cmd_status_mgr.sv`

```powershell
iverilog -g2012 -o sim/tb_host_cmd_status_mgr.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/common/tinyllama_bus_pkg.sv `
  rtl/control/host_cmd_status_mgr.sv `
  rtl/tb/tb_host_cmd_status_mgr.sv
vvp sim/tb_host_cmd_status_mgr.vvp
```

Expected pass string:

```text
PASS: tb_host_cmd_status_mgr
```

### `tb_prefill_decode_controller.sv`

```powershell
iverilog -g2012 -o sim/tb_prefill_decode_controller.vvp `
  rtl/common/tinyllama_pkg.sv `
  rtl/control/prefill_decode_controller.sv `
  rtl/control/layer_controller.sv `
  rtl/control/stop_condition_unit.sv `
  rtl/tb/tb_prefill_decode_controller.sv
vvp sim/tb_prefill_decode_controller.vvp
```

Expected pass string:

```text
PASS: tb_prefill_decode_controller
```
