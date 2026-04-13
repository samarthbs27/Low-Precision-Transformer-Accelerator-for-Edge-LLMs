`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_causal_mask_unit;

  localparam string PREFILL_META_FILE = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_causal_mask_q0_kv0_qb8_kb0.meta.memh";
  localparam string PREFILL_IN_FILE   = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_causal_mask_q0_kv0_qb8_kb0.score_in_packed.memh";
  localparam string PREFILL_OUT_FILE  = "sim/golden_traces/phase4/rtl/phase4_prefill_layer0_causal_mask_q0_kv0_qb8_kb0.score_out_expected_packed.memh";

  localparam string DECODE_META_FILE = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_causal_mask_q0_kv0_qb15_kb0.meta.memh";
  localparam string DECODE_IN_FILE   = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_causal_mask_q0_kv0_qb15_kb0.score_in_packed.memh";
  localparam string DECODE_OUT_FILE  = "sim/golden_traces/phase4/rtl/phase4_decode_layer0_causal_mask_q0_kv0_qb15_kb0.score_out_expected_packed.memh";

  runtime_mode_e runtime_mode;
  logic [POS_W-1:0] query_pos_base;
  logic [POS_W-1:0] key_pos_base;
  logic [COUNT_W-1:0] query_row_count;
  logic [COUNT_W-1:0] key_col_count;
  acc_bus_t score_bus;
  acc_bus_t masked_bus;

  logic [31:0] meta_mem [0:5];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] score_in_mem [0:0];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] score_out_mem [0:0];

  causal_mask_unit dut (
    .runtime_mode_i(runtime_mode),
    .query_pos_base_i(query_pos_base),
    .key_pos_base_i(key_pos_base),
    .query_row_count_i(query_row_count),
    .key_col_count_i(key_col_count),
    .score_i(score_bus),
    .masked_o(masked_bus)
  );

  task automatic load_trace_case(
    input string meta_file,
    input string score_in_file,
    input string score_out_file
  );
    begin
      $readmemh(meta_file, meta_mem);
      $readmemh(score_in_file, score_in_mem);
      $readmemh(score_out_file, score_out_mem);
    end
  endtask

  task automatic run_trace_case(
    input string case_name,
    input runtime_mode_e case_mode,
    input string meta_file,
    input string score_in_file,
    input string score_out_file
  );
    int q_head_id;
    int kv_head_id;
    int row_count;
    int col_count;
    begin
      load_trace_case(meta_file, score_in_file, score_out_file);

      runtime_mode = case_mode;
      query_pos_base = POS_W'(meta_mem[0]);
      key_pos_base = POS_W'(meta_mem[1]);
      query_row_count = COUNT_W'(meta_mem[2]);
      key_col_count = COUNT_W'(meta_mem[3]);
      q_head_id = meta_mem[4];
      kv_head_id = meta_mem[5];
      row_count = meta_mem[2];
      col_count = meta_mem[3];

      score_bus = '0;
      score_bus.tag.block_id = BLOCK_SCORE;
      score_bus.tag.gemm_mode = GEMM_SCORE;
      score_bus.tag.q_head_id = Q_HEAD_ID_W'(q_head_id);
      score_bus.tag.kv_head_id = KV_HEAD_ID_W'(kv_head_id);
      score_bus.tag.elem_count = ELEM_COUNT_W'(row_count * SCORE_K_TILE);
      score_bus.data = score_in_mem[0];

      #1;

      if ((masked_bus.tag.block_id != BLOCK_CAUSAL_MASK) ||
          (masked_bus.tag.gemm_mode != GEMM_SCORE) ||
          (masked_bus.tag.elem_count != ELEM_COUNT_W'(row_count * SCORE_K_TILE)) ||
          (masked_bus.tag.q_head_id != Q_HEAD_ID_W'(q_head_id)) ||
          (masked_bus.tag.kv_head_id != KV_HEAD_ID_W'(kv_head_id)) ||
          (masked_bus.tag.is_partial != ((row_count != SCORE_ROWS_PER_CHUNK) || (col_count != SCORE_K_TILE)))) begin
        $error("causal_mask_unit trace tag mismatch for %s", case_name);
        $finish;
      end

      if (masked_bus.data !== score_out_mem[0]) begin
        $error("causal_mask_unit trace mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    runtime_mode = MODE_PREFILL;
    query_pos_base = '0;
    key_pos_base = '0;
    query_row_count = COUNT_W'(1);
    key_col_count = COUNT_W'(4);
    score_bus = '0;
    score_bus.tag.block_id = BLOCK_SCORE;
    score_bus.tag.gemm_mode = GEMM_SCORE;
    score_bus.tag.elem_count = ELEM_COUNT_W'(SCORE_K_TILE);
    score_bus.data[0] = 32'sd11;
    score_bus.data[1] = 32'sd12;
    score_bus.data[2] = 32'sd13;
    score_bus.data[3] = 32'sd14;

    #1;

    if ((masked_bus.data[0] != 32'sd11) ||
        (masked_bus.data[1] != MASK_NEG_INF) ||
        (masked_bus.data[2] != MASK_NEG_INF) ||
        (masked_bus.data[3] != MASK_NEG_INF) ||
        (masked_bus.data[4] != MASK_NEG_INF) ||
        (masked_bus.data[SCORE_K_TILE] != '0)) begin
      $error("causal_mask_unit directed mask mismatch");
      $finish;
    end

    run_trace_case(
      "phase4_prefill_layer0_causal_mask_q0_kv0_qb8_kb0",
      MODE_PREFILL,
      PREFILL_META_FILE,
      PREFILL_IN_FILE,
      PREFILL_OUT_FILE
    );

    run_trace_case(
      "phase4_decode_layer0_causal_mask_q0_kv0_qb15_kb0",
      MODE_DECODE,
      DECODE_META_FILE,
      DECODE_IN_FILE,
      DECODE_OUT_FILE
    );

    $display("PASS: tb_causal_mask_unit");
    $finish;
  end

endmodule
