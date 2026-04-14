`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_residual_add;

  localparam string PREFILL_BASE = "sim/golden_traces/phase6/rtl/phase6_prefill_layer0_residual1_m0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase6/rtl/phase6_decode_layer0_residual2_m0";

  logic      clk;
  logic      rst_n;
  block_id_e block_id;
  logic      residual_valid;
  logic      residual_ready;
  acc_bus_t  residual_bus;
  logic      update_valid;
  logic      update_ready;
  acc_bus_t  update_bus;
  logic      sum_valid;
  logic      sum_ready;
  acc_bus_t  sum_bus;
  logic      busy;
  logic      done_pulse;
  logic      saw_sum;
  acc_bus_t  captured_sum_bus;

  logic [31:0]                        meta_mem [0:1];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] residual_mem [0:0];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] update_mem [0:0];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] sum_mem [0:0];

  residual_add dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .block_id_i(block_id),
    .residual_valid_i(residual_valid),
    .residual_ready_o(residual_ready),
    .residual_i(residual_bus),
    .update_valid_i(update_valid),
    .update_ready_o(update_ready),
    .update_i(update_bus),
    .sum_valid_o(sum_valid),
    .sum_ready_i(sum_ready),
    .sum_o(sum_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_sum <= 1'b0;
      captured_sum_bus <= '0;
    end else if (sum_valid && sum_ready) begin
      saw_sum <= 1'b1;
      captured_sum_bus <= sum_bus;
    end
  end

  task automatic reset_capture;
    begin
      @(negedge clk);
      saw_sum = 1'b0;
      captured_sum_bus = '0;
    end
  endtask

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".residual_in_packed.memh"}, residual_mem);
      $readmemh({base, ".update_in_packed.memh"}, update_mem);
      $readmemh({base, ".sum_expected_packed.memh"}, sum_mem);
    end
  endtask

  task automatic wait_for_sum(input string case_name);
    int timeout_cycles;
    begin
      timeout_cycles = 0;
      while (!saw_sum && (timeout_cycles < 32)) begin
        @(posedge clk);
        timeout_cycles++;
      end
      if (!saw_sum) begin
        $error("residual_add timed out waiting for sum for %s", case_name);
        $finish;
      end
      @(posedge clk);
      if (busy || sum_valid || done_pulse) begin
        $error("residual_add did not return idle after output for %s", case_name);
        $finish;
      end
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int elem_count;
    begin
      load_case(base);
      elem_count = meta_mem[1];
      block_id = block_id_e'(meta_mem[0][BLOCK_ID_W-1:0]);

      residual_bus = '0;
      residual_bus.tag.layer_id = '0;
      residual_bus.tag.block_id = BLOCK_NONE;
      residual_bus.tag.gemm_mode = GEMM_NONE;
      residual_bus.tag.tile_id = TILE_ID_W'(0);
      residual_bus.tag.elem_count = ELEM_COUNT_W'(elem_count);
      residual_bus.tag.is_partial = (elem_count != ACC_VECTOR_ELEMS);
      residual_bus.tag.is_last = 1'b1;
      residual_bus.data = residual_mem[0];

      update_bus = residual_bus;
      update_bus.data = update_mem[0];

      reset_capture();
      @(negedge clk);
      residual_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!residual_ready);
      @(negedge clk);
      residual_valid = 1'b0;

      @(negedge clk);
      update_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!update_ready);
      @(negedge clk);
      update_valid = 1'b0;

      wait_for_sum(case_name);

      if (captured_sum_bus.data !== sum_mem[0]) begin
        $error("residual_add data mismatch for %s", case_name);
        $finish;
      end

      if ((captured_sum_bus.tag.block_id != block_id) ||
          (captured_sum_bus.tag.gemm_mode != GEMM_NONE) ||
          (captured_sum_bus.tag.elem_count != ELEM_COUNT_W'(elem_count))) begin
        $error("residual_add tag mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    block_id = BLOCK_RESIDUAL1;
    residual_valid = 1'b0;
    residual_bus = '0;
    update_valid = 1'b0;
    update_bus = '0;
    sum_ready = 1'b1;
    saw_sum = 1'b0;
    captured_sum_bus = '0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    residual_bus.tag.elem_count = 16'd4;
    residual_bus.data[0] = 32'sd10;
    residual_bus.data[1] = -32'sd5;
    residual_bus.data[2] = 32'sd3;
    residual_bus.data[3] = -32'sd7;
    update_bus.tag = residual_bus.tag;
    update_bus.data[0] = 32'sd2;
    update_bus.data[1] = 32'sd8;
    update_bus.data[2] = -32'sd1;
    update_bus.data[3] = 32'sd4;
    reset_capture();
    @(negedge clk);
    residual_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!residual_ready);
    @(negedge clk);
    residual_valid = 1'b0;
    @(negedge clk);
    update_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!update_ready);
    @(negedge clk);
    update_valid = 1'b0;
    wait_for_sum("directed");
    if ((captured_sum_bus.data[0] != 32'sd12) ||
        (captured_sum_bus.data[1] != 32'sd3) ||
        (captured_sum_bus.data[2] != 32'sd2) ||
        (captured_sum_bus.data[3] != -32'sd3)) begin
      $error("residual_add directed mismatch");
      $finish;
    end

    run_case("phase6_prefill_layer0_residual1_m0", PREFILL_BASE);
    run_case("phase6_decode_layer0_residual2_m0", DECODE_BASE);

    $display("PASS: tb_residual_add");
    $finish;
  end

endmodule
