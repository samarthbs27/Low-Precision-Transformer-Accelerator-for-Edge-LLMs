# RTL Testbenches

This folder contains production TinyLlama RTL smoke tests for the new `rtl/common/` building blocks.

## Files

| File | What it tests | Smoke test |
|------|----------------|------------|
| `tb_stream_fifo.sv` | Directed self-check of `rtl/common/stream_fifo.sv`, including push/pop ordering, occupancy tracking, simultaneous push/pop, full-condition backpressure, and blocked-push behavior. | Run the `tb_stream_fifo` command below. |
| `tb_descriptor_fifo.sv` | Directed self-check of `rtl/common/descriptor_fifo.sv`, including push, pop, and occupancy behavior. | Run the `tb_descriptor_fifo` command below. |

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
