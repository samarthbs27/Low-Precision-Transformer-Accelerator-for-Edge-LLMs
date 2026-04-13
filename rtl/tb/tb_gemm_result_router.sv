`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_gemm_result_router;

  gemm_mode_e gemm_mode;
  logic       acc_valid;
  logic       acc_ready;
  acc_bus_t   acc_bus;
  logic       scale_valid;
  scale_bus_t scale_bus;
  logic       quant_valid;
  logic       quant_ready;
  act_bus_t   quant_bus;
  logic       score_valid;
  logic       score_ready;
  acc_bus_t   score_bus;
  logic       lmhead_valid;
  logic       lmhead_ready;
  acc_bus_t   lmhead_bus;

  gemm_result_router dut (
    .gemm_mode_i(gemm_mode),
    .acc_valid_i(acc_valid),
    .acc_ready_o(acc_ready),
    .acc_i(acc_bus),
    .scale_valid_i(scale_valid),
    .scale_i(scale_bus),
    .quant_valid_o(quant_valid),
    .quant_ready_i(quant_ready),
    .quant_o(quant_bus),
    .score_valid_o(score_valid),
    .score_ready_i(score_ready),
    .score_o(score_bus),
    .lmhead_valid_o(lmhead_valid),
    .lmhead_ready_i(lmhead_ready),
    .lmhead_o(lmhead_bus)
  );

  initial begin
    gemm_mode     = GEMM_Q;
    acc_valid     = 1'b1;
    scale_valid   = 1'b1;
    quant_ready   = 1'b1;
    score_ready   = 1'b1;
    lmhead_ready  = 1'b1;
    acc_bus       = '0;
    scale_bus     = '0;

    acc_bus.data[0]   = 32'sd12;
    scale_bus.data[0] = 32'h0001_0000;

    #1;
    if (!quant_valid || !acc_ready || (quant_bus.data[0] != 8'sd12) || score_valid || lmhead_valid) begin
      $error("gemm_result_router quantized path mismatch");
      $finish;
    end

    scale_valid = 1'b0;
    #1;
    if (quant_valid || acc_ready) begin
      $error("gemm_result_router expected quantized stall without scale metadata");
      $finish;
    end

    gemm_mode   = GEMM_SCORE;
    scale_valid = 1'b1;
    #1;
    if (!score_valid || !acc_ready || (score_bus.data[0] != 32'sd12) || quant_valid || lmhead_valid) begin
      $error("gemm_result_router score path mismatch");
      $finish;
    end

    gemm_mode = GEMM_LM_HEAD;
    #1;
    if (!lmhead_valid || !acc_ready || (lmhead_bus.data[0] != 32'sd12) || quant_valid || score_valid) begin
      $error("gemm_result_router lm-head path mismatch");
      $finish;
    end

    $display("PASS: tb_gemm_result_router");
    $finish;
  end

endmodule
