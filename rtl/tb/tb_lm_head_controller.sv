`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_lm_head_controller;

  localparam int unsigned LMHEAD_TILE_COUNT = (VOCAB_SIZE + VOCAB_TILE - 1) / VOCAB_TILE;

  logic                  clk;
  logic                  rst_n;
  logic                  start;
  logic                  hidden_valid;
  logic                  hidden_ready;
  act_bus_t              hidden_bus;
  logic                  hidden_scale_valid;
  logic                  hidden_scale_ready;
  scale_bus_t            hidden_scale_bus;
  logic                  context_valid;
  act_bus_t              context_hidden_bus;
  scale_bus_t            context_hidden_scale_bus;
  logic                  sched_start;
  logic [TILE_ID_W-1:0]  vocab_tile_idx;
  logic                  sched_done;
  logic                  logits_valid;
  logic                  logits_ready;
  acc_bus_t              logits_bus;
  logic                  argmax_valid;
  logic                  argmax_ready;
  acc_bus_t              argmax_bus;
  logic                  busy;
  logic                  done_pulse;

  lm_head_controller dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .hidden_valid_i(hidden_valid),
    .hidden_ready_o(hidden_ready),
    .hidden_i(hidden_bus),
    .hidden_scale_valid_i(hidden_scale_valid),
    .hidden_scale_ready_o(hidden_scale_ready),
    .hidden_scale_i(hidden_scale_bus),
    .context_valid_o(context_valid),
    .hidden_o(context_hidden_bus),
    .hidden_scale_o(context_hidden_scale_bus),
    .sched_start_o(sched_start),
    .vocab_tile_idx_o(vocab_tile_idx),
    .sched_done_i(sched_done),
    .logits_valid_i(logits_valid),
    .logits_ready_o(logits_ready),
    .logits_i(logits_bus),
    .argmax_valid_o(argmax_valid),
    .argmax_ready_i(argmax_ready),
    .argmax_o(argmax_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  task automatic drive_scheduler_tile(
    input int unsigned tile_idx
  );
    begin
      while (!sched_start) begin
        @(posedge clk);
      end

      if (vocab_tile_idx != TILE_ID_W'(tile_idx)) begin
        $error("lm_head_controller issued wrong vocab tile index: got %0d expected %0d", vocab_tile_idx, tile_idx);
        $finish;
      end

      @(negedge clk);
      sched_done = 1'b1;
      @(posedge clk);
      @(negedge clk);
      sched_done = 1'b0;

      logits_bus = '0;
      logits_bus.tag.layer_id = LAYER_ID_W'(5);
      logits_bus.data[0] = ACC_W'(tile_idx);
      @(negedge clk);
      logits_valid = 1'b1;
      @(posedge clk);
      if (!logits_ready || !argmax_valid) begin
        $error("lm_head_controller did not forward logits for tile %0d", tile_idx);
        $finish;
      end

      if ((argmax_bus.tag.block_id != BLOCK_LM_HEAD) ||
          (argmax_bus.tag.gemm_mode != GEMM_LM_HEAD) ||
          (argmax_bus.tag.tile_id != TILE_ID_W'(tile_idx)) ||
          (argmax_bus.tag.elem_count != ELEM_COUNT_W'(VOCAB_TILE)) ||
          (argmax_bus.tag.is_partial != 1'b0) ||
          (argmax_bus.tag.is_last != (tile_idx == (LMHEAD_TILE_COUNT - 1))) ||
          (argmax_bus.data[0] != ACC_W'(tile_idx))) begin
        $error("lm_head_controller output tag/data mismatch for tile %0d", tile_idx);
        $finish;
      end

      @(negedge clk);
      logits_valid = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    hidden_valid = 1'b0;
    hidden_bus = '0;
    hidden_scale_valid = 1'b0;
    hidden_scale_bus = '0;
    sched_done = 1'b0;
    logits_valid = 1'b0;
    logits_bus = '0;
    argmax_ready = 1'b1;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    hidden_bus.tag.layer_id = LAYER_ID_W'(5);
    hidden_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
    hidden_bus.tag.tile_id = TILE_ID_W'(0);
    hidden_bus.tag.elem_count = ELEM_COUNT_W'(ACT_VECTOR_ELEMS);
    hidden_bus.data[0] = 8'sd11;
    hidden_scale_bus.tag.layer_id = LAYER_ID_W'(5);
    hidden_scale_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
    hidden_scale_bus.data = {SCALE_VECTOR_ELEMS{32'h0001_0000}};

    @(negedge clk);
    start = 1'b1;
    hidden_valid = 1'b1;
    hidden_scale_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    start = 1'b0;
    hidden_valid = 1'b0;
    hidden_scale_valid = 1'b0;

    if (!context_valid ||
        (context_hidden_bus.data[0] != 8'sd11) ||
        (context_hidden_scale_bus.data[0] != 32'h0001_0000)) begin
      $error("lm_head_controller failed to capture hidden context");
      $finish;
    end

    for (int tile_idx = 0; tile_idx < LMHEAD_TILE_COUNT; tile_idx++) begin
      drive_scheduler_tile(tile_idx);
    end

    @(posedge clk);
    if (busy || !done_pulse || context_valid) begin
      $error("lm_head_controller did not terminate cleanly after the final tile");
      $finish;
    end

    $display("PASS: tb_lm_head_controller");
    $finish;
  end

endmodule
