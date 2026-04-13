`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_rope_unit;

  localparam string PREFILL_META_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_rope_q0_kv0_t0.meta.memh";
  localparam string PREFILL_Q_IN_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_rope_q0_kv0_t0.q_in_packed.memh";
  localparam string PREFILL_K_IN_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_rope_q0_kv0_t0.k_in_packed.memh";
  localparam string PREFILL_Q_OUT_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_rope_q0_kv0_t0.q_out_expected_packed.memh";
  localparam string PREFILL_K_OUT_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_rope_q0_kv0_t0.k_out_expected_packed.memh";

  localparam string DECODE_META_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_rope_q0_kv0_t15.meta.memh";
  localparam string DECODE_Q_IN_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_rope_q0_kv0_t15.q_in_packed.memh";
  localparam string DECODE_K_IN_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_rope_q0_kv0_t15.k_in_packed.memh";
  localparam string DECODE_Q_OUT_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_rope_q0_kv0_t15.q_out_expected_packed.memh";
  localparam string DECODE_K_OUT_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_rope_q0_kv0_t15.k_out_expected_packed.memh";

  act_bus_t q_in_bus;
  act_bus_t k_in_bus;
  act_bus_t q_out_bus;
  act_bus_t k_out_bus;

  logic [31:0] meta_mem [0:3];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] q_in_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] k_in_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] q_out_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] k_out_mem [0:0];

  rope_unit dut (
    .q_i(q_in_bus),
    .k_i(k_in_bus),
    .q_o(q_out_bus),
    .k_o(k_out_bus)
  );

  task automatic load_trace_case(
    input string meta_file,
    input string q_in_file,
    input string k_in_file,
    input string q_out_file,
    input string k_out_file
  );
    begin
      $readmemh(meta_file, meta_mem);
      $readmemh(q_in_file, q_in_mem);
      $readmemh(k_in_file, k_in_mem);
      $readmemh(q_out_file, q_out_mem);
      $readmemh(k_out_file, k_out_mem);
    end
  endtask

  task automatic run_trace_case(
    input string case_name,
    input string meta_file,
    input string q_in_file,
    input string k_in_file,
    input string q_out_file,
    input string k_out_file
  );
    int token_base;
    int token_count;
    int q_head_id;
    int kv_head_id;
    begin
      load_trace_case(meta_file, q_in_file, k_in_file, q_out_file, k_out_file);
      token_base = meta_mem[0];
      token_count = meta_mem[1];
      q_head_id = meta_mem[2];
      kv_head_id = meta_mem[3];

      q_in_bus = '0;
      k_in_bus = '0;
      q_in_bus.tag.block_id = BLOCK_Q;
      q_in_bus.tag.gemm_mode = GEMM_Q;
      q_in_bus.tag.token_base = POS_W'(token_base);
      q_in_bus.tag.seq_count = COUNT_W'(token_count);
      q_in_bus.tag.elem_count = ELEM_COUNT_W'(token_count * HEAD_DIM);
      q_in_bus.tag.q_head_id = Q_HEAD_ID_W'(q_head_id);
      q_in_bus.data = q_in_mem[0];

      k_in_bus.tag.block_id = BLOCK_K;
      k_in_bus.tag.gemm_mode = GEMM_K;
      k_in_bus.tag.token_base = POS_W'(token_base);
      k_in_bus.tag.seq_count = COUNT_W'(token_count);
      k_in_bus.tag.elem_count = ELEM_COUNT_W'(token_count * HEAD_DIM);
      k_in_bus.tag.kv_head_id = KV_HEAD_ID_W'(kv_head_id);
      k_in_bus.data = k_in_mem[0];

      #1;

      if ((q_out_bus.tag.block_id != BLOCK_ROPE) ||
          (k_out_bus.tag.block_id != BLOCK_ROPE) ||
          (q_out_bus.tag.elem_count != ELEM_COUNT_W'(token_count * HEAD_DIM)) ||
          (k_out_bus.tag.elem_count != ELEM_COUNT_W'(token_count * HEAD_DIM)) ||
          (q_out_bus.tag.seq_count != COUNT_W'(token_count)) ||
          (k_out_bus.tag.seq_count != COUNT_W'(token_count)) ||
          (q_out_bus.tag.q_head_id != Q_HEAD_ID_W'(q_head_id)) ||
          (k_out_bus.tag.kv_head_id != KV_HEAD_ID_W'(kv_head_id))) begin
        $error("rope_unit trace tag mismatch for %s", case_name);
        $finish;
      end

      if ((q_out_bus.data !== q_out_mem[0]) || (k_out_bus.data !== k_out_mem[0])) begin
        $error("rope_unit trace mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    q_in_bus = '0;
    k_in_bus = '0;

    q_in_bus.tag.block_id = BLOCK_Q;
    q_in_bus.tag.gemm_mode = GEMM_Q;
    q_in_bus.tag.seq_count = COUNT_W'(1);
    q_in_bus.tag.elem_count = ELEM_COUNT_W'(HEAD_DIM);
    q_in_bus.tag.token_base = '0;
    q_in_bus.data[0] = 8'sd7;
    q_in_bus.data[31] = -8'sd9;
    q_in_bus.data[32] = 8'sd11;
    q_in_bus.data[63] = -8'sd13;

    k_in_bus.tag.block_id = BLOCK_K;
    k_in_bus.tag.gemm_mode = GEMM_K;
    k_in_bus.tag.seq_count = COUNT_W'(1);
    k_in_bus.tag.elem_count = ELEM_COUNT_W'(HEAD_DIM);
    k_in_bus.tag.token_base = '0;
    k_in_bus.data[0] = -8'sd4;
    k_in_bus.data[31] = 8'sd6;
    k_in_bus.data[32] = 8'sd5;
    k_in_bus.data[63] = -8'sd8;

    #1;

    if ((q_out_bus.data[0] != q_in_bus.data[0]) ||
        (q_out_bus.data[31] != q_in_bus.data[31]) ||
        (q_out_bus.data[32] != q_in_bus.data[32]) ||
        (q_out_bus.data[63] != q_in_bus.data[63]) ||
        (k_out_bus.data[0] != k_in_bus.data[0]) ||
        (k_out_bus.data[31] != k_in_bus.data[31]) ||
        (k_out_bus.data[32] != k_in_bus.data[32]) ||
        (k_out_bus.data[63] != k_in_bus.data[63])) begin
      $error("rope_unit expected identity rotation at token position 0");
      $finish;
    end

    run_trace_case(
      "phase4_prefill_layer0_rope_q0_kv0_t0",
      PREFILL_META_FILE,
      PREFILL_Q_IN_FILE,
      PREFILL_K_IN_FILE,
      PREFILL_Q_OUT_FILE,
      PREFILL_K_OUT_FILE
    );

    run_trace_case(
      "phase4_decode_layer0_rope_q0_kv0_t15",
      DECODE_META_FILE,
      DECODE_Q_IN_FILE,
      DECODE_K_IN_FILE,
      DECODE_Q_OUT_FILE,
      DECODE_K_OUT_FILE
    );

    $display("PASS: tb_rope_unit");
    $finish;
  end

endmodule
