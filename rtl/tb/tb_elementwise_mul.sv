`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_elementwise_mul;

  localparam string PREFILL_BASE = "sim/golden_traces/phase6/rtl/phase6_prefill_layer0_glu_mul_m0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase6/rtl/phase6_decode_layer0_glu_mul_m0";

  logic clk;
  logic rst_n;
  logic silu_valid;
  logic silu_ready;
  act_bus_t silu_bus;
  logic up_valid;
  logic up_ready;
  act_bus_t up_bus;
  logic prod_valid;
  logic prod_ready;
  acc_bus_t prod_bus;
  logic busy;
  logic done_pulse;
  logic clear_capture;
  logic saw_prod;
  acc_bus_t captured_prod_bus;

  logic [31:0] meta_mem [0:1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] silu_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] up_mem [0:0];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] prod_mem [0:0];

  elementwise_mul dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .silu_valid_i(silu_valid),
    .silu_ready_o(silu_ready),
    .silu_i(silu_bus),
    .up_valid_i(up_valid),
    .up_ready_o(up_ready),
    .up_i(up_bus),
    .prod_valid_o(prod_valid),
    .prod_ready_i(prod_ready),
    .prod_o(prod_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n || clear_capture) begin
      saw_prod <= 1'b0;
      captured_prod_bus <= '0;
    end else if (prod_valid && prod_ready) begin
      saw_prod <= 1'b1;
      captured_prod_bus <= prod_bus;
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

  task automatic wait_for_prod(input string case_name);
    int timeout_cycles;
    begin
      timeout_cycles = 0;
      while (!saw_prod && (timeout_cycles < 32)) begin
        @(posedge clk);
        timeout_cycles++;
      end

      if (!saw_prod) begin
        $error("elementwise_mul timed out waiting for output handshake for %s", case_name);
        $finish;
      end

      @(posedge clk);
      if (busy || prod_valid || done_pulse) begin
        $error("elementwise_mul did not return idle after output handshake for %s", case_name);
        $finish;
      end
    end
  endtask

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".silu_in_packed.memh"}, silu_mem);
      $readmemh({base, ".up_in_packed.memh"}, up_mem);
      $readmemh({base, ".prod_out_expected_packed.memh"}, prod_mem);
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int elem_count;
    begin
      load_case(base);
      elem_count = meta_mem[1];

      silu_bus = '0;
      silu_bus.tag.layer_id = '0;
      silu_bus.tag.block_id = BLOCK_SILU;
      silu_bus.tag.gemm_mode = GEMM_NONE;
      silu_bus.tag.tile_id = TILE_ID_W'(meta_mem[0]);
      silu_bus.tag.elem_count = ELEM_COUNT_W'(elem_count);
      silu_bus.tag.is_partial = (elem_count != ACT_VECTOR_ELEMS);
      silu_bus.tag.is_last = 1'b1;
      silu_bus.data = silu_mem[0];

      up_bus = silu_bus;
      up_bus.tag.block_id = BLOCK_UP;
      up_bus.data = up_mem[0];

      reset_capture();
      @(negedge clk);
      silu_valid = 1'b1;
      up_valid = 1'b0;
      prod_ready = 1'b1;

      do begin
        @(posedge clk);
      end while (!silu_ready);
      @(negedge clk);
      silu_valid = 1'b0;

      @(negedge clk);
      up_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!up_ready);
      @(negedge clk);
      up_valid = 1'b0;

      wait_for_prod(case_name);

      if (captured_prod_bus.data !== prod_mem[0]) begin
        $error("elementwise_mul data mismatch for %s", case_name);
        $finish;
      end

      if ((captured_prod_bus.tag.block_id != BLOCK_GLU_MUL) ||
          (captured_prod_bus.tag.gemm_mode != GEMM_NONE) ||
          (captured_prod_bus.tag.tile_id != TILE_ID_W'(meta_mem[0])) ||
          (captured_prod_bus.tag.elem_count != ELEM_COUNT_W'(elem_count))) begin
        $error("elementwise_mul tag mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    silu_valid = 1'b0;
    up_valid = 1'b0;
    silu_bus = '0;
    up_bus = '0;
    prod_ready = 1'b1;
    clear_capture = 1'b0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    reset_capture();
    silu_bus.tag.elem_count = 16'd4;
    up_bus.tag = silu_bus.tag;
    silu_bus.data[0] = 8'sd2;
    silu_bus.data[1] = -8'sd3;
    silu_bus.data[2] = 8'sd4;
    silu_bus.data[3] = -8'sd5;
    up_bus.data[0] = -8'sd6;
    up_bus.data[1] = 8'sd7;
    up_bus.data[2] = 8'sd8;
    up_bus.data[3] = -8'sd9;
    @(negedge clk);
    silu_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!silu_ready);
    @(negedge clk);
    silu_valid = 1'b0;
    up_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!up_ready);
    @(negedge clk);
    up_valid = 1'b0;
    wait_for_prod("directed");

    if ((captured_prod_bus.data[0] != -32'sd12) ||
        (captured_prod_bus.data[1] != -32'sd21) ||
        (captured_prod_bus.data[2] != 32'sd32) ||
        (captured_prod_bus.data[3] != 32'sd45) ||
        (captured_prod_bus.data[4] != '0)) begin
      $error("elementwise_mul directed multiply mismatch");
      $finish;
    end

    run_case("phase6_prefill_layer0_glu_mul_m0", PREFILL_BASE);
    run_case("phase6_decode_layer0_glu_mul_m0", DECODE_BASE);

    $display("PASS: tb_elementwise_mul");
    $finish;
  end

endmodule
