`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_argmax_reduction;

  localparam int MAX_ARGMAX_TILES = (VOCAB_SIZE + VOCAB_TILE - 1) / VOCAB_TILE;
  localparam string PREFILL_BASE = "sim/golden_traces/phase6/rtl/phase6_prefill_argmax";
  localparam string DECODE_BASE  = "sim/golden_traces/phase6/rtl/phase6_decode_argmax";

  logic clk;
  logic rst_n;
  logic start;
  logic logits_valid;
  logic logits_ready;
  acc_bus_t logits_bus;
  logic token_valid;
  logic token_ready;
  logic [TOKEN_W-1:0] token_id;
  logic signed [ACC_W-1:0] token_logit;
  logic busy;
  logic done_pulse;
  logic clear_capture;
  logic saw_token;
  logic [TOKEN_W-1:0] captured_token_id;
  logic signed [ACC_W-1:0] captured_token_logit;

  logic [31:0] meta_mem [0:3];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] logits_tiles_mem [0:MAX_ARGMAX_TILES-1];

  argmax_reduction dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .logits_valid_i(logits_valid),
    .logits_ready_o(logits_ready),
    .logits_i(logits_bus),
    .token_valid_o(token_valid),
    .token_ready_i(token_ready),
    .token_id_o(token_id),
    .token_logit_o(token_logit),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n || clear_capture) begin
      saw_token <= 1'b0;
      captured_token_id <= '0;
      captured_token_logit <= '0;
    end else if (token_valid && token_ready) begin
      saw_token <= 1'b1;
      captured_token_id <= token_id;
      captured_token_logit <= token_logit;
    end
  end

  task automatic reset_capture;
    begin
      @(negedge clk);
      clear_capture = 1'b1;
      @(posedge clk);
      @(negedge clk);
      clear_capture = 1'b0;
    end
  endtask

  task automatic wait_for_token(input string case_name);
    int timeout_cycles;
    begin
      timeout_cycles = 0;
      while (!saw_token && (timeout_cycles < 512)) begin
        @(posedge clk);
        timeout_cycles++;
      end

      if (!saw_token) begin
        $error("argmax_reduction timed out waiting for token handshake for %s", case_name);
        $finish;
      end

      @(posedge clk);
      if (busy || token_valid || done_pulse) begin
        $error("argmax_reduction did not return idle after token handshake for %s", case_name);
        $finish;
      end
    end
  endtask

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".logits_tiles_packed.memh"}, logits_tiles_mem);
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int tile_count;
    begin
      load_case(base);
      tile_count = meta_mem[0];
      reset_capture();

      @(negedge clk);
      start = 1'b1;
      @(posedge clk);
      @(negedge clk);
      start = 1'b0;

      for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) begin
        @(negedge clk);
        logits_bus = '0;
        logits_bus.tag.block_id = BLOCK_LM_HEAD;
        logits_bus.tag.gemm_mode = GEMM_LM_HEAD;
        logits_bus.tag.tile_id = TILE_ID_W'(tile_idx);
        logits_bus.tag.elem_count = ELEM_COUNT_W'(VOCAB_TILE);
        logits_bus.tag.is_partial = 1'b0;
        logits_bus.tag.is_last = (tile_idx == (tile_count - 1));
        logits_bus.data = logits_tiles_mem[tile_idx];
        logits_valid = 1'b1;
        do begin
          @(posedge clk);
        end while (!logits_ready);
        @(negedge clk);
        logits_valid = 1'b0;
      end

      wait_for_token(case_name);

      if ((captured_token_id != TOKEN_W'(meta_mem[1])) ||
          (captured_token_logit != $signed(meta_mem[2]))) begin
        $error("argmax_reduction mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    logits_valid = 1'b0;
    logits_bus = '0;
    token_ready = 1'b1;
    clear_capture = 1'b0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    reset_capture();
    @(negedge clk);
    start = 1'b1;
    @(posedge clk);
    @(negedge clk);
    start = 1'b0;

    @(negedge clk);
    logits_bus = '0;
    logits_bus.tag.tile_id = '0;
    logits_bus.tag.elem_count = ELEM_COUNT_W'(VOCAB_TILE);
    logits_bus.tag.is_last = 1'b0;
    logits_bus.data[0] = 32'sd10;
    logits_bus.data[1] = 32'sd55;
    logits_bus.data[2] = 32'sd55;
    logits_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    logits_valid = 1'b0;

    @(negedge clk);
    logits_bus = '0;
    logits_bus.tag.tile_id = TILE_ID_W'(1);
    logits_bus.tag.elem_count = ELEM_COUNT_W'(VOCAB_TILE);
    logits_bus.tag.is_last = 1'b1;
    logits_bus.data[0] = 32'sd40;
    logits_bus.data[1] = 32'sd100;
    logits_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    logits_valid = 1'b0;

    wait_for_token("directed");

    if ((captured_token_id != TOKEN_W'(VOCAB_TILE + 1)) ||
        (captured_token_logit != 32'sd100)) begin
      $error("argmax_reduction directed mismatch");
      $finish;
    end

    run_case("phase6_prefill_argmax", PREFILL_BASE);
    run_case("phase6_decode_argmax", DECODE_BASE);

    $display("PASS: tb_argmax_reduction");
    $finish;
  end

endmodule
