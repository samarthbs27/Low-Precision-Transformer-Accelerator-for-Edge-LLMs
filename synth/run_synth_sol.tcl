# Vivado batch synthesis script — ASU SOL HPC cluster version.
# No OneDrive, no path spaces, multi-thread safe.
# Run via SLURM:  sbatch synth/sol_synth.slurm
# Or directly:    vivado -mode batch -source synth/run_synth_sol.tcl -nojournal

set PART      xcu55c-fsvh2892-2L-e
set TOP       tinyllama_u55c_kernel_top
set PROJ_ROOT [file normalize [file dirname [info script]]/..]

cd $PROJ_ROOT

set OUT_DIR [file normalize synth/out]
file mkdir $OUT_DIR

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
# flatten_hierarchy full + maxThreads 1: proven-working configuration (job
# 51905999).  All ports use `ifdef SYNTHESIS flat-logic shims to avoid the
# NDup::dupNameType(UAggType*) Vivado 2022.1 crash; no ram_style attributes
# (they trigger a different writeAGenome code path that re-crashes).
#
# NOTE: 512-lane synthesis (GEMM_LANES=512, ACC_BUS_W=16463 bits) crashes
# Vivado 2022.1 with NDup::dupNameType(UAggType*) regardless of flatten
# setting (tried: full, rebuilt, none).  This is a width-dependent tool bug.
# Proven-working configuration below is for 64 lanes (job 51905999, Apr 27).
# Use 64-lane synth/out/ reports as the milestone synthesis artifact.
set_param synth.maxThreads 8

synth_design \
  -top  $TOP \
  -part $PART \
  -mode out_of_context \
  -flatten_hierarchy full \
  -directive RuntimeOptimized

# ── Reports and checkpoint ────────────────────────────────────────────────────
set rpt_util [file join $OUT_DIR utilization_synth.rpt]
set rpt_time [file join $OUT_DIR timing_synth.rpt]
set dcp_path [file join $OUT_DIR kernel_top_synth.dcp]

report_utilization    -file $rpt_util
report_timing_summary -file $rpt_time -max_paths 20
write_checkpoint -force $dcp_path

puts "Synthesis complete. Reports in $OUT_DIR"
