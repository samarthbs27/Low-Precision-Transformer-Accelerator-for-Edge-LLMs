`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_shared_gemm_engine;

  localparam string TRACE_META_FILE = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_gemm_smoke_m0_n0_k0.meta.memh";
  localparam string TRACE_ACT_FILE  = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_gemm_smoke_m0_n0_k0.act_steps_packed.memh";
  localparam string TRACE_WT_FILE   = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_gemm_smoke_m0_n0_k0.wt_steps_packed.memh";
  localparam string TRACE_ACC_FILE  = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_gemm_smoke_m0_n0_k0.acc_expected_packed.memh";
  localparam int unsigned TRACE_K_WORDS = 64;

  logic       clk;
  logic       rst_n;
  gemm_mode_e gemm_mode;
  logic       clear_acc;
  logic       mac_valid;
  logic       emit_acc;
  logic       operands_valid;
  logic       operands_ready;
  act_bus_t   act_bus;
  wt_bus_t    wt_bus;
  logic       acc_valid;
  logic       acc_ready;
  acc_bus_t   acc_bus;
  logic       busy;

  logic [31:0]                    trace_meta_mem [0:3];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0]    trace_act_mem [0:TRACE_K_WORDS-1];
  logic [(WEIGHT_VECTOR_ELEMS * WEIGHT_W)-1:0] trace_wt_mem [0:TRACE_K_WORDS-1];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0]    trace_acc_expected_mem [0:0];

  shared_gemm_engine dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .gemm_mode_i(gemm_mode),
    .clear_acc_i(clear_acc),
    .mac_valid_i(mac_valid),
    .emit_acc_i(emit_acc),
    .operands_valid_i(operands_valid),
    .operands_ready_o(operands_ready),
    .act_i(act_bus),
    .wt_i(wt_bus),
    .acc_valid_o(acc_valid),
    .acc_ready_i(acc_ready),
    .acc_o(acc_bus),
    .busy_o(busy)
  );

  always #5 clk = ~clk;

  task automatic clear_operands;
    begin
      act_bus.data = '0;
      wt_bus.data  = '0;
    end
  endtask

  task automatic consume_snapshot_and_expect_idle;
    begin
      if (operands_ready) begin
        $error("shared_gemm_engine should backpressure while snapshot is pending");
        $finish;
      end

      @(negedge clk);
      acc_ready = 1'b1;
      @(negedge clk);
      acc_ready = 1'b0;

      if (acc_valid || !operands_ready || busy) begin
        $error("shared_gemm_engine expected idle after snapshot consume");
        $finish;
      end
    end
  endtask

  task automatic run_directed_smoke_case;
    begin
      act_bus.tag.elem_count = 16'd4;
      wt_bus.tag.elem_count  = 16'd4;
      act_bus.tag.tile_id    = 16'h0033;
      wt_bus.tag.tile_id     = 16'h0033;

      @(negedge clk);
      clear_acc      = 1'b1;
      mac_valid      = 1'b1;
      operands_valid = 1'b1;
      act_bus.data[0] = 8'sd1;
      act_bus.data[1] = 8'sd2;
      act_bus.data[2] = 8'sd3;
      act_bus.data[3] = 8'sd4;
      wt_bus.data[0]  = 8'sd1;
      wt_bus.data[1]  = 8'sd1;
      wt_bus.data[2]  = 8'sd1;
      wt_bus.data[3]  = 8'sd1;

      @(negedge clk);
      clear_acc       = 1'b0;
      act_bus.data[0] = 8'sd10;
      act_bus.data[1] = 8'sd20;
      act_bus.data[2] = 8'sd30;
      act_bus.data[3] = 8'sd40;

      @(negedge clk);
      emit_acc       = 1'b1;
      operands_valid = 1'b0;
      mac_valid      = 1'b0;
      clear_operands();

      @(negedge clk);
      emit_acc = 1'b0;

      if (!acc_valid || (acc_bus.data[0] != 32'sd11) || (acc_bus.data[1] != 32'sd22) ||
          (acc_bus.data[2] != 32'sd33) || (acc_bus.data[3] != 32'sd44) ||
          (acc_bus.tag.block_id != BLOCK_Q)) begin
        $error("shared_gemm_engine directed accumulated output mismatch");
        $finish;
      end

      consume_snapshot_and_expect_idle();
    end
  endtask

  task automatic load_exported_trace_case;
    begin
      $readmemh(TRACE_META_FILE, trace_meta_mem);
      $readmemh(TRACE_ACT_FILE, trace_act_mem);
      $readmemh(TRACE_WT_FILE, trace_wt_mem);
      $readmemh(TRACE_ACC_FILE, trace_acc_expected_mem);
    end
  endtask

  task automatic run_exported_trace_case;
    int k_total;
    int active_lane_count;
    int step_idx;
    begin
      load_exported_trace_case();
      k_total = trace_meta_mem[0];
      active_lane_count = trace_meta_mem[1];

      if ((k_total != K_TILE) || (active_lane_count != ACC_VECTOR_ELEMS)) begin
        $error("shared_gemm_engine trace metadata mismatch");
        $finish;
      end

      gemm_mode              = GEMM_Q;
      clear_acc              = 1'b0;
      mac_valid              = 1'b0;
      emit_acc               = 1'b0;
      operands_valid         = 1'b0;
      acc_ready              = 1'b0;
      act_bus                = '0;
      wt_bus                 = '0;
      act_bus.tag.block_id   = BLOCK_Q;
      act_bus.tag.gemm_mode  = GEMM_Q;
      wt_bus.tag.block_id    = BLOCK_Q;
      wt_bus.tag.gemm_mode   = GEMM_Q;
      act_bus.tag.elem_count = ELEM_COUNT_W'(active_lane_count);
      wt_bus.tag.elem_count  = ELEM_COUNT_W'(active_lane_count);

      for (step_idx = 0; step_idx < k_total; step_idx++) begin
        @(negedge clk);
        clear_acc          = (step_idx == 0);
        mac_valid          = 1'b1;
        emit_acc           = 1'b0;
        operands_valid     = 1'b1;
        act_bus.tag.is_last = (step_idx == (k_total - 1));
        wt_bus.tag.is_last  = (step_idx == (k_total - 1));
        act_bus.data       = trace_act_mem[step_idx];
        wt_bus.data        = trace_wt_mem[step_idx];
      end

      @(negedge clk);
      clear_acc      = 1'b0;
      mac_valid      = 1'b0;
      operands_valid = 1'b0;
      emit_acc       = 1'b1;
      clear_operands();

      @(negedge clk);
      emit_acc = 1'b0;

      if (!acc_valid || (acc_bus.tag.block_id != BLOCK_Q) || (acc_bus.tag.gemm_mode != GEMM_Q) ||
          (acc_bus.tag.elem_count != ELEM_COUNT_W'(active_lane_count))) begin
        $error("shared_gemm_engine trace snapshot/tag mismatch");
        $finish;
      end

      if (acc_bus.data !== trace_acc_expected_mem[0]) begin
        $error("shared_gemm_engine exported trace mismatch");
        $finish;
      end

      consume_snapshot_and_expect_idle();
    end
  endtask

  initial begin
    clk            = 1'b0;
    rst_n          = 1'b0;
    gemm_mode      = GEMM_Q;
    clear_acc      = 1'b0;
    mac_valid      = 1'b0;
    emit_acc       = 1'b0;
    operands_valid = 1'b0;
    act_bus        = '0;
    wt_bus         = '0;
    acc_ready      = 1'b0;

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    run_directed_smoke_case();
    run_exported_trace_case();

    $display("PASS: tb_shared_gemm_engine");
    $finish;
  end

endmodule
