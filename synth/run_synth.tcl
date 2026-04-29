# Vivado batch-mode synthesis script for tinyllama_u55c_kernel_top.
# Target: Alveo U55C (xcu55c-fsvh2892-2L-e), 300 MHz, out-of-context.
# HLS IP sim models are excluded; synthesis stubs are used instead.
# Run via:  .\synth\run_synth.ps1

set PART      xcu55c-fsvh2892-2L-e
set TOP       tinyllama_u55c_kernel_top
set PROJ_ROOT [file normalize [file dirname [info script]]/..]

# cd to project root so read_verilog/read_xdc use relative paths.
# Relative paths avoid Vivado's word-splitting on "Reconfigureable Computing".
# NOTE: this also anchors .Xil/ to $PROJ_ROOT (inside OneDrive), which causes
# [Designutils 20-411] "realtime dir could not be deleted" at synthesis end.
# The catch block below handles that by collecting cell counts and attempting
# write_checkpoint even when synth_design returns a TCL error.
cd $PROJ_ROOT

# OUT_DIR: normalize to absolute before any subsequent cd changes.
# OUT_DIR_TMP: writable temp dir outside OneDrive — used for all Vivado file
# writes (write_checkpoint, report_utilization, report_timing_summary).
# OneDrive can mark synth/out read-only; %TEMP% is always writable.
# Outputs are copied from OUT_DIR_TMP → OUT_DIR at the end of the script.
set OUT_DIR     [file normalize synth/out]
set OUT_DIR_TMP [file join [file normalize $::env(TEMP)] vivado_synth_tinyllama_out]
file mkdir $OUT_DIR
file mkdir $OUT_DIR_TMP
# Clear OneDrive read-only flag on OUT_DIR so the final copy succeeds.
catch { file attributes $OUT_DIR -readonly 0 }

# ── Source files (packages first, then leaves, then top) ─────────────────────
set RTL_SRCS [list \
  rtl/common/tinyllama_pkg.sv \
  rtl/common/tinyllama_bus_pkg.sv \
  rtl/common/stream_fifo.sv \
  rtl/common/skid_buffer.sv \
  rtl/common/descriptor_fifo.sv \
  rtl/control/axi_lite_ctrl_slave.sv \
  rtl/control/kernel_reg_file.sv \
  rtl/control/prefill_decode_controller.sv \
  rtl/control/layer_controller.sv \
  rtl/control/stop_condition_unit.sv \
  rtl/control/host_cmd_status_mgr.sv \
  rtl/memory/hbm_port_router.sv \
  rtl/memory/tile_buffer_bank.sv \
  rtl/memory/kv_cache_manager.sv \
  rtl/memory/scale_metadata_store.sv \
  rtl/memory/prompt_token_reader.sv \
  rtl/memory/generated_token_writer.sv \
  rtl/memory/weight_dma_reader.sv \
  rtl/memory/kv_cache_dma_reader.sv \
  rtl/memory/kv_cache_dma_writer.sv \
  rtl/memory/embedding_lmhead_dma_reader.sv \
  rtl/memory/debug_dma_writer.sv \
  rtl/compute/mac_lane.sv \
  rtl/compute/accumulator_bank.sv \
  rtl/compute/requantize_unit.sv \
  rtl/compute/gemm_operand_router.sv \
  rtl/compute/gemm_result_router.sv \
  rtl/compute/gemm_op_scheduler.sv \
  rtl/compute/shared_gemm_engine.sv \
  rtl/compute/rope_lut_rom.sv \
  rtl/compute/rope_unit.sv \
  rtl/compute/gqa_router.sv \
  rtl/compute/causal_mask_unit.sv \
  rtl/compute/elementwise_mul.sv \
  rtl/compute/residual_add.sv \
  rtl/compute/argmax_reduction.sv \
  rtl/compute/lm_head_controller.sv \
  rtl/compute/embedding_lookup.sv \
  rtl/compute/embedding_quantizer.sv \
  rtl/compute/debug_capture_mux.sv \
  rtl/nonlinear/rmsnorm_wrapper.sv \
  rtl/nonlinear/softmax_wrapper.sv \
  rtl/nonlinear/silu_wrapper.sv \
  rtl/nonlinear/rmsnorm_core_hls_ip_synth.sv \
  rtl/nonlinear/softmax_core_hls_ip_synth.sv \
  rtl/nonlinear/silu_core_hls_ip_synth.sv \
  rtl/top/runtime_embedding_frontend.sv \
  rtl/top/runtime_decoder_datapath.sv \
  rtl/top/runtime_final_rmsnorm_tail.sv \
  rtl/top/runtime_lm_head_tail.sv \
  rtl/top/tinyllama_u55c_kernel_top.sv \
]

# ── Read sources ──────────────────────────────────────────────────────────────
foreach f $RTL_SRCS {
  read_verilog -sv $f
}

# ── Constraints ───────────────────────────────────────────────────────────────
read_xdc synth/kernel_top.xdc

# ── Synthesis ─────────────────────────────────────────────────────────────────
# Disable multithreading: helper process uses absolute project path (with space).
set_param synth.maxThreads 1

# Full technology-mapping synthesis.
# GEMM_LANES=64 via `ifdef SYNTHESIS (min safe: ROPE_CHUNK_TOKENS=64/64=1).
# Peak RAM ~17 GB; design takes ~14 min at RuntimeOptimized directive.
# Wrapped in catch: OneDrive locks .Xil/realtime during cleanup → Designutils
# 20-411 makes synth_design return TCL error even though netlist is complete.
# We collect utilization from get_cells and attempt write_checkpoint manually.
set synth_rc [catch {
  synth_design \
    -top  $TOP \
    -part $PART \
    -mode out_of_context \
    -flatten_hierarchy none \
    -fsm_extraction one_hot \
    -directive RuntimeOptimized
} synth_err]

set dcp_path      [file join $OUT_DIR_TMP kernel_top_synth.dcp]
set rpt_util_synth [file join $OUT_DIR_TMP utilization_synth.rpt]
set rpt_time_synth [file join $OUT_DIR_TMP timing_synth.rpt]

if { $synth_rc != 0 } {
  puts "NOTE: synth_design TCL error (likely OneDrive cleanup): $synth_err"
  set all_cells  [llength [get_cells -hierarchical -quiet]]
  set lut_cells  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ LUT*}]]
  set ff_cells   [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ FD* || REF_NAME =~ LD*}]]
  set dsp_cells  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ DSP*}]]
  set bram_cells [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB*}]]
  puts "UTILIZATION ESTIMATE (GEMM_LANES=64 scaled run):"
  puts "  Total cells : $all_cells"
  puts "  LUTs        : $lut_cells"
  puts "  FFs         : $ff_cells"
  puts "  DSPs        : $dsp_cells"
  puts "  BRAMs       : $bram_cells"
  if { $all_cells == 0 } {
    puts "FATAL: empty netlist — synthesis did not complete."
    exit 1
  }
  puts "Attempting write_checkpoint to $OUT_DIR_TMP ..."
  if { [catch {write_checkpoint -force $dcp_path} cp_err] } {
    puts "NOTE: write_checkpoint also failed: $cp_err"
  } else {
    puts "Checkpoint written: $dcp_path"
    catch { file copy -force $dcp_path [file join $OUT_DIR kernel_top_synth.dcp] }
    puts "Copied DCP to $OUT_DIR"
  }
} else {
  # Clean success path — write full reports then copy to OUT_DIR.
  report_utilization    -file $rpt_util_synth
  report_timing_summary -file $rpt_time_synth -max_paths 20
  write_checkpoint -force $dcp_path
  puts "Reports and DCP written to $OUT_DIR_TMP"
  foreach src [list $dcp_path $rpt_util_synth $rpt_time_synth] {
    catch { file copy -force $src $OUT_DIR }
  }
  puts "Synthesis complete. Outputs copied to $OUT_DIR"
}
