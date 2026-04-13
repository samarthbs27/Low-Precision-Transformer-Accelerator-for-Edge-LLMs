`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_accumulator_bank;

  logic                                          clk;
  logic                                          rst_n;
  logic                                          clear;
  logic                                          load;
  logic signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] load_data;
  tile_tag_t                                     load_tag;
  acc_bus_t                                      acc_bus;

  accumulator_bank dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .clear_i(clear),
    .load_i(load),
    .load_data_i(load_data),
    .load_tag_i(load_tag),
    .acc_o(acc_bus)
  );

  always #5 clk = ~clk;

  initial begin
    clk       = 1'b0;
    rst_n     = 1'b0;
    clear     = 1'b0;
    load      = 1'b0;
    load_data = '0;
    load_tag  = '0;

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    load                <= 1'b1;
    load_data[0]        <= 32'sd11;
    load_data[1]        <= -32'sd7;
    load_tag.block_id   <= BLOCK_Q;
    load_tag.gemm_mode  <= GEMM_Q;
    load_tag.elem_count <= 16'd2;
    @(negedge clk);
    load <= 1'b0;

    if ((acc_bus.data[0] != 32'sd11) || (acc_bus.data[1] != -32'sd7) ||
        (acc_bus.tag.block_id != BLOCK_Q) || (acc_bus.tag.elem_count != 16'd2)) begin
      $error("accumulator_bank load mismatch");
      $finish;
    end

    @(negedge clk);
    clear              <= 1'b1;
    load_tag.block_id  <= BLOCK_O;
    load_tag.gemm_mode <= GEMM_O;
    @(negedge clk);
    clear <= 1'b0;

    if ((acc_bus.data[0] != '0) || (acc_bus.data[1] != '0) ||
        (acc_bus.tag.block_id != BLOCK_O)) begin
      $error("accumulator_bank clear mismatch");
      $finish;
    end

    $display("PASS: tb_accumulator_bank");
    $finish;
  end

endmodule
