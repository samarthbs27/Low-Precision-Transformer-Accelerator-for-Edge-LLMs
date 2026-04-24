# U55C Bring-Up And First Vivado Synthesis Checklist

This guide is the handoff document for the next stage of the project:

- the repo is through the first concrete Phase 9 runtime-acceptance and shell-wrapper step
- the current verified top-level seam is the normalized shell DMA boundary
- the raw U55C-facing `m_axi_pc00..31` wrapper is still future work

That means the next vendor-tool step is **not** a full board run yet. The next
step is:

1. move the real FPGA flow to Linux
2. verify XRT + U55C platform installation
3. rerun the current Phase 9 Linux-side smoke checks
4. run the **first Vivado synthesis sanity pass** on the current runtime core
5. use that synthesis result to guide the raw U55C wrapper work

If you only have Windows today, keep using it for repo work, Python export, and
Icarus smoke tests. Do the real Alveo bring-up on Linux.

---

## Current Repo State

What is already concrete:

- [tinyllama_u55c_kernel_top.sv](../rtl/top/tinyllama_u55c_kernel_top.sv)
- [tinyllama_u55c_shell_wrapper.sv](../rtl/top/tinyllama_u55c_shell_wrapper.sv)
- Phase 8 runtime smoke:
  - [tb_prefill_decode_smoke.sv](../rtl/tb/tb_prefill_decode_smoke.sv)
  - [tb_kernel_top_smoke.sv](../rtl/tb/tb_kernel_top_smoke.sv)
- Phase 9 acceptance/shell-seam smoke:
  - [tb_kernel_top_acceptance.sv](../rtl/tb/tb_kernel_top_acceptance.sv)
  - [tb_shell_wrapper_smoke.sv](../rtl/tb/tb_shell_wrapper_smoke.sv)
- real-model runtime fixtures in `sim/golden_traces/phase8/rtl/` and
  `sim/golden_traces/phase9/rtl/`

What is **not** done yet:

- the raw U55C-facing `m_axi_pc00..31` wrapper
- the first Vitis RTL-kernel packaging step
- the first `.xo` / `.xclbin` build
- the first on-card smoke run

So the right next move is an **early Vivado synthesis pass**, not a full Vitis
hardware build.

---

## Why This Is Vivado First, Not Vitis First

Use **both** tools, but in this order:

- **Vivado now**:
  - elaborate and synthesize the current runtime core + shell wrapper
  - catch RTL/tool issues early
  - get the first resource/toplevel sanity signal
- **Vitis later**:
  - package the design as an RTL kernel
  - link against the U55C platform
  - generate the `.xclbin`

This matches AMD's RTL-kernel flow: Vivado is used to develop, verify, and
package the RTL kernel, and Vitis links the kernel with the platform into the
final device binary.

---

## Step-By-Step From Now

### Step 1: Use A Linux Machine For The Real FPGA Flow

Use native Linux for:

- XRT
- U55C platform installation
- Vitis platform visibility
- real board/runtime bring-up

Windows is still fine for:

- RTL edits
- Python trace export
- local Icarus simulation

### Step 2: Verify The Tool And Platform Install

This guide assumes Vivado and Vitis are already installed. You still need to
check the Alveo-side pieces:

- XRT
- U55C development platform
- U55C deployment platform
- ideally the U55C Vivado board file / XDC if you plan to move into raw
  platform-facing wrapper work soon

Run:

```bash
vivado -version
vitis -version
v++ --version
xbutil --version
```

If your XRT release uses the renamed utility, this also works:

```bash
xrt-smi --version
```

### Step 3: Source The Tool Environments

Do this in every new shell before using the tools:

```bash
source /path/to/Vivado/<version>/settings64.sh
source /path/to/Vitis/<version>/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

Replace the paths with your actual install paths.

### Step 4: Verify The Card And Platform Are Visible

First verify the card:

```bash
xbutil examine
```

If your environment uses the new XRT naming:

```bash
xrt-smi examine
```

Then verify the installed platform:

```bash
platforminfo -p <u55c_platform_name>
```

or:

```bash
platforminfo /full/path/to/<platform>.xpfm
```

You want to confirm:

- the U55C card is visible
- the U55C platform is installed
- the platform has the memory/clock metadata Vitis expects

### Step 5: Re-Run The Current Phase 9 Sanity Checks On Linux

From the repo root:

```bash
python model/export_fpga_vectors.py --phase phase9 --output-dir sim/golden_traces
```

Then rerun the current Phase 9 benches documented in
[rtl/tb/README.md](../rtl/tb/README.md):

- `tb_kernel_top_smoke`
- `tb_kernel_top_acceptance`
- `tb_shell_wrapper_smoke`

These are the current gates that should still pass before starting vendor-tool
work.

If you want the explicit commands, they are:

```bash
iverilog -g2012 -o sim/tb_kernel_top_smoke.vvp \
  rtl/common/tinyllama_pkg.sv \
  rtl/common/tinyllama_bus_pkg.sv \
  rtl/common/stream_fifo.sv \
  rtl/common/skid_buffer.sv \
  rtl/common/descriptor_fifo.sv \
  rtl/control/axi_lite_ctrl_slave.sv \
  rtl/control/kernel_reg_file.sv \
  rtl/control/host_cmd_status_mgr.sv \
  rtl/control/prefill_decode_controller.sv \
  rtl/control/layer_controller.sv \
  rtl/control/stop_condition_unit.sv \
  rtl/memory/hbm_port_router.sv \
  rtl/memory/embedding_lmhead_dma_reader.sv \
  rtl/memory/prompt_token_reader.sv \
  rtl/memory/generated_token_writer.sv \
  rtl/compute/embedding_lookup.sv \
  rtl/compute/embedding_quantizer.sv \
  rtl/compute/residual_add.sv \
  rtl/compute/requantize_unit.sv \
  rtl/compute/elementwise_mul.sv \
  rtl/compute/mac_lane.sv \
  rtl/compute/accumulator_bank.sv \
  rtl/compute/shared_gemm_engine.sv \
  rtl/compute/lm_head_controller.sv \
  rtl/compute/argmax_reduction.sv \
  rtl/nonlinear/rmsnorm_core_hls_ip.sv \
  rtl/nonlinear/rmsnorm_wrapper.sv \
  rtl/nonlinear/silu_core_hls_ip.sv \
  rtl/nonlinear/silu_wrapper.sv \
  rtl/top/runtime_embedding_frontend.sv \
  rtl/top/runtime_decoder_datapath.sv \
  rtl/top/runtime_final_rmsnorm_tail.sv \
  rtl/top/runtime_lm_head_tail.sv \
  rtl/top/tinyllama_u55c_kernel_top.sv \
  rtl/tb/tb_kernel_top_smoke.sv
vvp sim/tb_kernel_top_smoke.vvp
```

```bash
iverilog -g2012 -o sim/tb_kernel_top_acceptance.vvp \
  rtl/common/tinyllama_pkg.sv \
  rtl/common/tinyllama_bus_pkg.sv \
  rtl/common/stream_fifo.sv \
  rtl/common/skid_buffer.sv \
  rtl/common/descriptor_fifo.sv \
  rtl/control/axi_lite_ctrl_slave.sv \
  rtl/control/kernel_reg_file.sv \
  rtl/control/host_cmd_status_mgr.sv \
  rtl/control/prefill_decode_controller.sv \
  rtl/control/layer_controller.sv \
  rtl/control/stop_condition_unit.sv \
  rtl/memory/hbm_port_router.sv \
  rtl/memory/embedding_lmhead_dma_reader.sv \
  rtl/memory/prompt_token_reader.sv \
  rtl/memory/generated_token_writer.sv \
  rtl/compute/embedding_lookup.sv \
  rtl/compute/embedding_quantizer.sv \
  rtl/compute/residual_add.sv \
  rtl/compute/requantize_unit.sv \
  rtl/compute/elementwise_mul.sv \
  rtl/compute/mac_lane.sv \
  rtl/compute/accumulator_bank.sv \
  rtl/compute/shared_gemm_engine.sv \
  rtl/compute/lm_head_controller.sv \
  rtl/compute/argmax_reduction.sv \
  rtl/nonlinear/rmsnorm_core_hls_ip.sv \
  rtl/nonlinear/rmsnorm_wrapper.sv \
  rtl/nonlinear/silu_core_hls_ip.sv \
  rtl/nonlinear/silu_wrapper.sv \
  rtl/top/runtime_embedding_frontend.sv \
  rtl/top/runtime_decoder_datapath.sv \
  rtl/top/runtime_final_rmsnorm_tail.sv \
  rtl/top/runtime_lm_head_tail.sv \
  rtl/top/tinyllama_u55c_kernel_top.sv \
  rtl/tb/tb_kernel_top_acceptance.sv
vvp sim/tb_kernel_top_acceptance.vvp
```

```bash
iverilog -g2012 -o sim/tb_shell_wrapper_smoke.vvp \
  rtl/common/tinyllama_pkg.sv \
  rtl/common/tinyllama_bus_pkg.sv \
  rtl/common/stream_fifo.sv \
  rtl/common/skid_buffer.sv \
  rtl/common/descriptor_fifo.sv \
  rtl/control/axi_lite_ctrl_slave.sv \
  rtl/control/kernel_reg_file.sv \
  rtl/control/host_cmd_status_mgr.sv \
  rtl/control/prefill_decode_controller.sv \
  rtl/control/layer_controller.sv \
  rtl/control/stop_condition_unit.sv \
  rtl/memory/hbm_port_router.sv \
  rtl/memory/embedding_lmhead_dma_reader.sv \
  rtl/memory/prompt_token_reader.sv \
  rtl/memory/generated_token_writer.sv \
  rtl/compute/embedding_lookup.sv \
  rtl/compute/embedding_quantizer.sv \
  rtl/compute/residual_add.sv \
  rtl/compute/requantize_unit.sv \
  rtl/compute/elementwise_mul.sv \
  rtl/compute/mac_lane.sv \
  rtl/compute/accumulator_bank.sv \
  rtl/compute/shared_gemm_engine.sv \
  rtl/compute/lm_head_controller.sv \
  rtl/compute/argmax_reduction.sv \
  rtl/nonlinear/rmsnorm_core_hls_ip.sv \
  rtl/nonlinear/rmsnorm_wrapper.sv \
  rtl/nonlinear/silu_core_hls_ip.sv \
  rtl/nonlinear/silu_wrapper.sv \
  rtl/top/runtime_embedding_frontend.sv \
  rtl/top/runtime_decoder_datapath.sv \
  rtl/top/runtime_final_rmsnorm_tail.sv \
  rtl/top/runtime_lm_head_tail.sv \
  rtl/top/tinyllama_u55c_kernel_top.sv \
  rtl/top/tinyllama_u55c_shell_wrapper.sv \
  rtl/tb/tb_shell_wrapper_smoke.sv
vvp sim/tb_shell_wrapper_smoke.vvp
```

Expected pass strings:

- `PASS: tb_kernel_top_smoke`
- `PASS: tb_kernel_top_acceptance`
- `PASS: tb_shell_wrapper_smoke`

---

## Step 6: Very Concrete Vivado Synthesis Checklist

This is the **first vendor-tool sanity pass** for the current runtime core.

### Goal

Prove that the current top-level RTL:

- elaborates in Vivado
- synthesizes without fatal tool issues
- produces a usable hierarchy and utilization report

### Non-Goals

This step is **not**:

- the final U55C platform wrapper
- the final timing-closure run
- the `.xo` packaging step
- the `.xclbin` build
- the first board run

### Recommended Top Module

Set the synthesis top to:

- `tinyllama_u55c_shell_wrapper`

Reason:

- it is the current outer verified RTL seam
- it includes the runtime core
- it includes the new Phase 9 shell-side skid buffering

### Recommended Source Set

Add exactly these design files for the first pass:

```text
rtl/common/tinyllama_pkg.sv
rtl/common/tinyllama_bus_pkg.sv
rtl/common/stream_fifo.sv
rtl/common/skid_buffer.sv
rtl/common/descriptor_fifo.sv
rtl/control/axi_lite_ctrl_slave.sv
rtl/control/kernel_reg_file.sv
rtl/control/host_cmd_status_mgr.sv
rtl/control/prefill_decode_controller.sv
rtl/control/layer_controller.sv
rtl/control/stop_condition_unit.sv
rtl/memory/hbm_port_router.sv
rtl/memory/embedding_lmhead_dma_reader.sv
rtl/memory/prompt_token_reader.sv
rtl/memory/generated_token_writer.sv
rtl/compute/embedding_lookup.sv
rtl/compute/embedding_quantizer.sv
rtl/top/tinyllama_u55c_kernel_top.sv
rtl/top/runtime_embedding_frontend.sv
rtl/top/tinyllama_u55c_shell_wrapper.sv
```

Do **not** add:

- testbenches
- legacy root-level demo RTL such as `rtl/top.sv` or `rtl/control_fsm.sv`
- the future raw U55C wrapper that does not exist yet

### GUI Checklist

1. Launch Vivado.
2. Create a new project.
3. Project name: `tinyllama_phase9_shell_synth`.
4. Project type: `RTL Project`.
5. Check `Do not specify sources at this time`.
6. On the board/device page:
   - if the AMD U55C board file is installed, select the U55C board
   - if not, stop and install the board file before treating this as a real
     U55C-targeted sanity run
7. Finish project creation.
8. In `Sources`, choose `Add Sources`.
9. Add the exact source set listed above as `Design Sources`.
10. Verify compile order:
    - `tinyllama_pkg.sv` must compile before modules that import it
    - `tinyllama_bus_pkg.sv` must compile before modules that use bus typedefs
11. Set the top module to `tinyllama_u55c_shell_wrapper`.
12. Run `Open Elaborated Design`.
13. If elaboration fails, stop and fix the first fatal error before going
    further.
14. If elaboration succeeds, run `Synthesis`.
15. When synthesis completes, save:
    - synthesis log
    - messages report
    - hierarchy screenshot or hierarchy text export
    - utilization report

### What To Check In The Result

Treat Step 6 as a pass only if all of these are true:

- Vivado elaborates the design successfully
- Vivado synthesizes the design successfully
- top module is `tinyllama_u55c_shell_wrapper`
- both `tinyllama_u55c_shell_wrapper` and `tinyllama_u55c_kernel_top` appear in
  the synthesized hierarchy
- `runtime_embedding_frontend` appears beneath `tinyllama_u55c_kernel_top`
- there are no fatal errors about:
  - unsupported SystemVerilog packages
  - packed-struct port handling
  - multiple drivers
  - inferred latches we did not intend
  - missing modules

Warnings to pay attention to:

- package/import ordering problems
- signals optimized away unexpectedly at the top seam
- width truncation/sign-extension warnings
- unsupported cast/packed-array constructs
- tool complaints around the DMA descriptor struct boundary

### What To Record For The Team

Write down:

- Vivado version
- target board/device used
- whether elaboration passed
- whether synthesis passed
- total LUT/FF/BRAM/URAM/DSP counts from the utilization report
- the first 10-20 nontrivial warnings, if any

Remember what this pass proves:

- the current runtime core plus shell wrapper is synthesizable
- the first real-inference closure slice, `runtime_embedding_frontend`, is in
  the synth target

It still does **not** prove the final full TinyLlama accelerator fits or meets
timing, because the integrated decoder/LM/argmax top-level datapath is not
fully wired yet.

### Optional Batch Flow

If someone prefers Tcl instead of the GUI, the first-pass flow can be done with
the same source set and top module. Keep it synthesis-only. Do not jump into
kernel packaging yet.

---

## Step 7: How To Interpret Step 6

If Step 6 passes:

- good, the current runtime core is ready for the next repo task
- the next repo task is the raw U55C-facing wrapper around the verified shell
  seam

If Step 6 fails:

- do not try to "push through" into Vitis
- fix the first fatal Vivado issue in the RTL first
- rerun Step 6 until elaboration and synthesis are clean

The point of this step is to catch problems while the design boundary is still
small and understandable.

---

## Step 8: What Not To Do Yet

Do **not** do these yet:

- do not try to build the final `.xo`
- do not try to link the final `.xclbin`
- do not try to run on the U55C board
- do not build the host/runtime handoff around the final Alveo shell yet

Reason:

- the raw `m_axi_pc00..31` wrapper is still missing
- the current verified seam is still the normalized shell DMA boundary

---

## Step 9: Repo Work That Comes Immediately After Step 6

Once the first Vivado synthesis pass is clean, the next code milestone is:

- build the raw U55C-facing wrapper outside
  [tinyllama_u55c_shell_wrapper.sv](../rtl/top/tinyllama_u55c_shell_wrapper.sv)
- map the verified normalized shell DMA boundary onto the real platform-facing
  memory interfaces
- rerun focused smoke checks around that wrapper
- only then move into full Vitis kernel packaging and linking

---

## Step 10: What The Later Vitis Flow Will Look Like

This is **later**, not now:

1. package the RTL as an RTL kernel
2. generate the `.xo`
3. link the `.xo` against the U55C platform with Vitis
4. let Vitis call Vivado for implementation
5. generate the `.xclbin`
6. load it with XRT and run the first on-card smoke

That is the right final flow, but this guide intentionally stops one stage
earlier because the raw U55C wrapper is not in the repo yet.

---

## Official References

- AMD U55C downloads and platform packages:
  https://www.amd.com/en/support/downloads/alveo-downloads.html/accelerators/alveo/u55c.html
- AMD UG1393, RTL-designer Vitis flow:
  https://docs.amd.com/r/2024.1-English/ug1393-vitis-application-acceleration/Vitis-Development-Flow-for-RTL-Designers
- AMD UG1393, `v++` / platform / kernel flow:
  https://docs.amd.com/r/2024.1-English/ug1393-vitis-application-acceleration/v-General-Options
- AMD `platforminfo` examples:
  https://docs.amd.com/r/2024.1-English/Vitis-Tutorials-Vitis-Platform-Creation/Test-1-Read-Platform-Info
- XRT `xbutil` / `xrt-smi` reference:
  https://xilinx.github.io/XRT/2023.1/html/xbutil.html
  https://xilinx.github.io/XRT/master/html/xrt-smi.html
