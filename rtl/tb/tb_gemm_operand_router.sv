`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_gemm_operand_router;

  gemm_mode_e gemm_mode;
  logic       act_valid;
  logic       act_ready;
  act_bus_t   act_bus;
  logic       weight_valid;
  logic       weight_ready;
  wt_bus_t    weight_bus;
  logic       score_valid;
  logic       score_ready;
  act_bus_t   score_bus;
  logic       kv_valid;
  logic       kv_ready;
  act_bus_t   kv_bus;
  logic       operands_valid;
  logic       operands_ready;
  act_bus_t   routed_act;
  wt_bus_t    routed_wt;

  gemm_operand_router dut (
    .gemm_mode_i(gemm_mode),
    .act_valid_i(act_valid),
    .act_ready_o(act_ready),
    .act_i(act_bus),
    .weight_valid_i(weight_valid),
    .weight_ready_o(weight_ready),
    .weight_i(weight_bus),
    .score_valid_i(score_valid),
    .score_ready_o(score_ready),
    .score_i(score_bus),
    .kv_valid_i(kv_valid),
    .kv_ready_o(kv_ready),
    .kv_i(kv_bus),
    .operands_valid_o(operands_valid),
    .operands_ready_i(operands_ready),
    .act_o(routed_act),
    .wt_o(routed_wt)
  );

  initial begin
    gemm_mode      = GEMM_Q;
    act_valid      = 1'b1;
    weight_valid   = 1'b1;
    score_valid    = 1'b1;
    kv_valid       = 1'b1;
    operands_ready = 1'b1;
    act_bus        = '0;
    weight_bus     = '0;
    score_bus      = '0;
    kv_bus         = '0;

    act_bus.data[0]    = 8'sd7;
    weight_bus.data[0] = -8'sd5;
    score_bus.data[0]  = 8'sd22;
    kv_bus.data[0]     = 8'sd9;

    #1;
    if (!operands_valid || !act_ready || !weight_ready ||
        (routed_act.data[0] != 8'sd7) || (routed_wt.data[0] != -8'sd5) ||
        (routed_act.tag.block_id != BLOCK_Q) || (routed_act.tag.gemm_mode != GEMM_Q) ||
        (routed_wt.tag.block_id != BLOCK_Q) || (routed_wt.tag.gemm_mode != GEMM_Q)) begin
      $error("gemm_operand_router Q-path mismatch");
      $finish;
    end

    gemm_mode = GEMM_SCORE;
    #1;
    if (!operands_valid || !act_ready || !kv_ready || weight_ready || score_ready ||
        (routed_act.data[0] != 8'sd7) || (routed_wt.data[0] != 8'sd9) ||
        (routed_act.tag.block_id != BLOCK_SCORE) || (routed_act.tag.gemm_mode != GEMM_SCORE) ||
        (routed_wt.tag.block_id != BLOCK_SCORE) || (routed_wt.tag.gemm_mode != GEMM_SCORE)) begin
      $error("gemm_operand_router SCORE-path mismatch");
      $finish;
    end

    gemm_mode = GEMM_WEIGHTED_SUM;
    #1;
    if (!operands_valid || !score_ready || !kv_ready || act_ready ||
        (routed_act.data[0] != 8'sd22) || (routed_wt.data[0] != 8'sd9) ||
        (routed_act.tag.block_id != BLOCK_WEIGHTED_SUM) || (routed_act.tag.gemm_mode != GEMM_WEIGHTED_SUM) ||
        (routed_wt.tag.block_id != BLOCK_WEIGHTED_SUM) || (routed_wt.tag.gemm_mode != GEMM_WEIGHTED_SUM)) begin
      $error("gemm_operand_router WEIGHTED_SUM-path mismatch");
      $finish;
    end

    $display("PASS: tb_gemm_operand_router");
    $finish;
  end

endmodule
