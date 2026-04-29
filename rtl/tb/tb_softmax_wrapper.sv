`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_softmax_wrapper;

  localparam int unsigned CHUNK_ELEMS = N_TILE;
  localparam int unsigned CHUNK_W = CHUNK_ELEMS * SCALE_W;
  localparam int unsigned CHUNK_COUNT = SCORE_CHUNK_ELEMS / CHUNK_ELEMS;

  localparam string PREFILL_BASE = "sim/golden_traces/phase5/rtl/phase5_prefill_layer0_softmax_q0_kv0_qb8_kb0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase5/rtl/phase5_decode_layer0_softmax_q0_kv0_qb15_kb0";

  logic clk;
  logic rst_n;
  logic score_valid;
  logic score_ready;
  acc_bus_t score_bus;
  logic [SCALE_W-1:0] score_scale;
  logic prob_scale_valid;
  logic prob_scale_ready;
  scale_bus_t prob_scale_bus;
  logic prob_valid;
  logic prob_ready;
  act_bus_t prob_bus;
  logic busy;
  logic done_pulse;

  logic [31:0] meta_mem [0:5];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0] score_in_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] prob_out_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] score_scale_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] prob_scale_mem [0:0];
  logic [CHUNK_W-1:0] core_score_chunks_mem [0:CHUNK_COUNT-1];
  logic [CHUNK_W-1:0] core_prob_chunks_mem [0:CHUNK_COUNT-1];

  logic saw_scale;
  logic saw_prob;
  scale_bus_t captured_scale_bus;
  act_bus_t captured_prob_bus;
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] captured_scale_data;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_prob_data;

  softmax_wrapper dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .score_valid_i(score_valid),
    .score_ready_o(score_ready),
    .score_i(score_bus),
    .score_scale_i(score_scale),
    .prob_scale_valid_o(prob_scale_valid),
    .prob_scale_ready_i(prob_scale_ready),
    .prob_scale_o(prob_scale_bus),
    .prob_valid_o(prob_valid),
    .prob_ready_i(prob_ready),
    .prob_o(prob_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_scale <= 1'b0;
      saw_prob <= 1'b0;
      captured_scale_bus <= '0;
      captured_prob_bus <= '0;
    end else begin
      if (prob_scale_valid && prob_scale_ready) begin
        saw_scale <= 1'b1;
        captured_scale_bus <= prob_scale_bus;
        captured_scale_data <= prob_scale_bus.data;
      end
      if (prob_valid && prob_ready) begin
        saw_prob <= 1'b1;
        captured_prob_bus <= prob_bus;
        captured_prob_data <= prob_bus.data;
      end
    end
  end

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".score_in_packed.memh"}, score_in_mem);
      $readmemh({base, ".prob_out_expected_packed.memh"}, prob_out_mem);
      $readmemh({base, ".score_scale_packed.memh"}, score_scale_mem);
      $readmemh({base, ".prob_scale_packed.memh"}, prob_scale_mem);
      $readmemh({base, ".core_score_chunks_packed.memh"}, core_score_chunks_mem);
      $readmemh({base, ".core_prob_chunks_packed.memh"}, core_prob_chunks_mem);
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int cycle_count;
    begin
      load_case(base);

      score_bus = '0;
      score_bus.tag.block_id = BLOCK_CAUSAL_MASK;
      score_bus.tag.gemm_mode = GEMM_SCORE;
      score_bus.tag.token_base = POS_W'(meta_mem[2]);
      score_bus.tag.seq_count = COUNT_W'(meta_mem[0]);
      score_bus.tag.q_head_id = Q_HEAD_ID_W'(meta_mem[4]);
      score_bus.tag.kv_head_id = KV_HEAD_ID_W'(meta_mem[5]);
      score_bus.tag.elem_count = ELEM_COUNT_W'(meta_mem[0] * SCORE_K_TILE);
      score_bus.tag.is_partial = (meta_mem[0] != SCORE_ROWS_PER_CHUNK) || (meta_mem[1] != SCORE_K_TILE);
      score_bus.tag.is_last = 1'b1;
      score_bus.data = score_in_mem[0];

      score_scale = score_scale_mem[0][SCALE_W-1:0];
      score_valid = 1'b1;
      prob_scale_ready = 1'b1;
      prob_ready = 1'b1;
      saw_scale = 1'b0;
      saw_prob = 1'b0;

      cycle_count = 0;
      while (!done_pulse && (cycle_count < 200)) begin
        @(posedge clk);
        if (score_valid && score_ready) begin
          score_valid <= 1'b0;
        end
        cycle_count = cycle_count + 1;
      end

      if (!done_pulse) begin
        $error("softmax_wrapper timeout for %s", case_name);
        $finish;
      end

      @(posedge clk);

      if (!saw_scale || !saw_prob) begin
        $error("softmax_wrapper missing output handshake for %s", case_name);
        $finish;
      end

      if (captured_prob_data !== prob_out_mem[0]) begin
        $error("softmax_wrapper probability mismatch for %s", case_name);
        $display("expected=%h", prob_out_mem[0]);
        $display("actual  =%h", captured_prob_data);
        $finish;
      end

      if (captured_scale_data !== prob_scale_mem[0]) begin
        $error("softmax_wrapper probability scale mismatch for %s", case_name);
        $finish;
      end

      if ((captured_prob_bus.tag.block_id != BLOCK_SOFTMAX) ||
          (captured_prob_bus.tag.gemm_mode != GEMM_WEIGHTED_SUM) ||
          (captured_prob_bus.tag.q_head_id != Q_HEAD_ID_W'(meta_mem[4])) ||
          (captured_prob_bus.tag.kv_head_id != KV_HEAD_ID_W'(meta_mem[5]))) begin
        $error("softmax_wrapper tag mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    score_valid = 1'b0;
    score_bus = '0;
    score_scale = '0;
    prob_scale_ready = 1'b1;
    prob_ready = 1'b1;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    run_case("phase5_prefill_layer0_softmax_q0_kv0_qb8_kb0", PREFILL_BASE);
    run_case("phase5_decode_layer0_softmax_q0_kv0_qb15_kb0", DECODE_BASE);

    $display("PASS: tb_softmax_wrapper");
    $finish;
  end

endmodule
