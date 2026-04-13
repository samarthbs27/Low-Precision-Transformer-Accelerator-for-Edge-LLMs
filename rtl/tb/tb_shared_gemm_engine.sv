`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_shared_gemm_engine;

  logic      clk;
  logic      rst_n;
  gemm_mode_e gemm_mode;
  logic      clear_acc;
  logic      mac_valid;
  logic      emit_acc;
  logic      operands_valid;
  logic      operands_ready;
  act_bus_t  act_bus;
  wt_bus_t   wt_bus;
  logic      acc_valid;
  logic      acc_ready;
  acc_bus_t  acc_bus;
  logic      busy;

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

    act_bus.tag.elem_count = 16'd4;
    wt_bus.tag.elem_count  = 16'd4;
    act_bus.tag.tile_id    = 16'h0033;
    wt_bus.tag.tile_id     = 16'h0033;

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    clear_acc          <= 1'b1;
    mac_valid          <= 1'b1;
    operands_valid     <= 1'b1;
    act_bus.data[0]    <= 8'sd1;
    act_bus.data[1]    <= 8'sd2;
    act_bus.data[2]    <= 8'sd3;
    act_bus.data[3]    <= 8'sd4;
    wt_bus.data[0]     <= 8'sd1;
    wt_bus.data[1]     <= 8'sd1;
    wt_bus.data[2]     <= 8'sd1;
    wt_bus.data[3]     <= 8'sd1;

    @(negedge clk);
    clear_acc       <= 1'b0;
    act_bus.data[0] <= 8'sd10;
    act_bus.data[1] <= 8'sd20;
    act_bus.data[2] <= 8'sd30;
    act_bus.data[3] <= 8'sd40;

    @(negedge clk);
    emit_acc        <= 1'b1;
    operands_valid  <= 1'b0;
    mac_valid       <= 1'b0;

    @(negedge clk);
    emit_acc <= 1'b0;

    if (!acc_valid || (acc_bus.data[0] != 32'sd11) || (acc_bus.data[1] != 32'sd22) ||
        (acc_bus.data[2] != 32'sd33) || (acc_bus.data[3] != 32'sd44) ||
        (acc_bus.tag.block_id != BLOCK_Q)) begin
      $error("shared_gemm_engine accumulated output mismatch");
      $finish;
    end

    if (operands_ready) begin
      $error("shared_gemm_engine should backpressure while snapshot is pending");
      $finish;
    end

    @(negedge clk);
    acc_ready <= 1'b1;
    @(negedge clk);
    acc_ready <= 1'b0;

    if (acc_valid || !operands_ready || busy) begin
      $error("shared_gemm_engine expected idle after snapshot consume");
      $finish;
    end

    $display("PASS: tb_shared_gemm_engine");
    $finish;
  end

endmodule
