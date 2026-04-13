`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_gqa_router;

  logic                   select_k;
  logic                   select_v;
  logic [Q_HEAD_ID_W-1:0] q_head_id;
  logic                   k_valid;
  logic                   k_ready;
  act_bus_t               k_bus;
  logic                   v_valid;
  logic                   v_ready;
  act_bus_t               v_bus;
  logic                   routed_valid;
  logic                   routed_ready;
  act_bus_t               routed_bus;
  logic                   route_error;
  logic [KV_HEAD_ID_W-1:0] expected_kv_head;

  gqa_router dut (
    .select_k_i(select_k),
    .select_v_i(select_v),
    .q_head_id_i(q_head_id),
    .k_valid_i(k_valid),
    .k_ready_o(k_ready),
    .k_i(k_bus),
    .v_valid_i(v_valid),
    .v_ready_o(v_ready),
    .v_i(v_bus),
    .routed_valid_o(routed_valid),
    .routed_ready_i(routed_ready),
    .routed_o(routed_bus),
    .route_error_o(route_error),
    .expected_kv_head_o(expected_kv_head)
  );

  initial begin
    select_k = 1'b0;
    select_v = 1'b0;
    q_head_id = '0;
    k_valid = 1'b0;
    v_valid = 1'b0;
    k_bus = '0;
    v_bus = '0;
    routed_ready = 1'b1;

    q_head_id = Q_HEAD_ID_W'(9);
    select_k = 1'b1;
    k_valid = 1'b1;
    k_bus.tag.kv_head_id = KV_HEAD_ID_W'(1);
    k_bus.tag.block_id = BLOCK_K;
    k_bus.tag.gemm_mode = GEMM_K;
    k_bus.data[0] = 8'sd12;

    #1;

    if (!routed_valid || !k_ready || route_error ||
        (expected_kv_head != KV_HEAD_ID_W'(1)) ||
        (routed_bus.tag.block_id != BLOCK_SCORE) ||
        (routed_bus.tag.gemm_mode != GEMM_SCORE) ||
        (routed_bus.tag.q_head_id != Q_HEAD_ID_W'(9)) ||
        (routed_bus.tag.kv_head_id != KV_HEAD_ID_W'(1)) ||
        (routed_bus.data[0] != 8'sd12)) begin
      $error("gqa_router K-route mismatch");
      $finish;
    end

    select_k = 1'b0;
    k_valid = 1'b0;
    q_head_id = Q_HEAD_ID_W'(17);
    select_v = 1'b1;
    v_valid = 1'b1;
    v_bus.tag.kv_head_id = KV_HEAD_ID_W'(2);
    v_bus.tag.block_id = BLOCK_V;
    v_bus.tag.gemm_mode = GEMM_V;
    v_bus.data[3] = -8'sd21;

    #1;

    if (!routed_valid || !v_ready || route_error ||
        (expected_kv_head != KV_HEAD_ID_W'(2)) ||
        (routed_bus.tag.block_id != BLOCK_WEIGHTED_SUM) ||
        (routed_bus.tag.gemm_mode != GEMM_WEIGHTED_SUM) ||
        (routed_bus.tag.q_head_id != Q_HEAD_ID_W'(17)) ||
        (routed_bus.tag.kv_head_id != KV_HEAD_ID_W'(2)) ||
        (routed_bus.data[3] != -8'sd21)) begin
      $error("gqa_router V-route mismatch");
      $finish;
    end

    select_v = 1'b0;
    v_valid = 1'b0;
    q_head_id = Q_HEAD_ID_W'(31);
    select_k = 1'b1;
    k_valid = 1'b1;
    k_bus.tag.kv_head_id = KV_HEAD_ID_W'(2);

    #1;

    if (!routed_valid || !route_error || (expected_kv_head != KV_HEAD_ID_W'(3))) begin
      $error("gqa_router expected mismatch flag");
      $finish;
    end

    select_v = 1'b1;
    v_valid = 1'b1;
    #1;

    if (routed_valid || k_ready || v_ready || !route_error) begin
      $error("gqa_router expected select-conflict error");
      $finish;
    end

    $display("PASS: tb_gqa_router");
    $finish;
  end

endmodule
