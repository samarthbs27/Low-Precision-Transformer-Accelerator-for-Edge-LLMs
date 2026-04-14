`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_gemm_op_scheduler;

  logic                 clk;
  logic                 rst_n;
  logic                 start;
  logic                 abort_req;
  logic                 lm_head_only;
  logic                 block_start;
  block_id_e            block_start_id;
  logic [Q_HEAD_ID_W-1:0]  block_q_head_id;
  logic [KV_HEAD_ID_W-1:0] block_kv_head_id;
  logic                 dma_ready;
  logic                 buffer_ready;
  logic                 step_ready;
  logic [COUNT_W-1:0]   seq_count;
  logic [COUNT_W-1:0]   kv_token_count;
  logic                 busy;
  logic                 done_pulse;
  logic                 step_valid;
  gemm_mode_e           gemm_mode;
  block_id_e            block_id;
  logic                 clear_acc;
  logic                 emit_acc;
  logic [TILE_ID_W-1:0] m_tile_idx;
  logic [TILE_ID_W-1:0] n_tile_idx;
  logic [TILE_ID_W-1:0] k_tile_idx;
  logic [TILE_ID_W-1:0] m_tile_count;
  logic [TILE_ID_W-1:0] n_tile_count;
  logic [TILE_ID_W-1:0] k_tile_count;
  logic [Q_HEAD_ID_W-1:0]  q_head_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;

  int unsigned q_steps;
  int unsigned k_steps;
  int unsigned v_steps;
  int unsigned score_steps;
  int unsigned weighted_sum_steps;
  int unsigned o_steps;
  int unsigned gate_steps;
  int unsigned up_steps;
  int unsigned down_steps;
  int unsigned lmhead_steps;
  integer      guard;

  gemm_op_scheduler dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .abort_req_i(abort_req),
    .lm_head_only_i(lm_head_only),
    .block_start_i(block_start),
    .block_id_i(block_start_id),
    .block_q_head_id_i(block_q_head_id),
    .block_kv_head_id_i(block_kv_head_id),
    .dma_ready_i(dma_ready),
    .buffer_ready_i(buffer_ready),
    .step_ready_i(step_ready),
    .seq_count_i(seq_count),
    .kv_token_count_i(kv_token_count),
    .busy_o(busy),
    .done_pulse_o(done_pulse),
    .step_valid_o(step_valid),
    .gemm_mode_o(gemm_mode),
    .block_id_o(block_id),
    .clear_acc_o(clear_acc),
    .emit_acc_o(emit_acc),
    .m_tile_idx_o(m_tile_idx),
    .n_tile_idx_o(n_tile_idx),
    .k_tile_idx_o(k_tile_idx),
    .m_tile_count_o(m_tile_count),
    .n_tile_count_o(n_tile_count),
    .k_tile_count_o(k_tile_count),
    .q_head_id_o(q_head_id),
    .kv_head_id_o(kv_head_id)
  );

  always #5 clk = ~clk;

  task automatic reset_counts;
    begin
      q_steps            = 0;
      k_steps            = 0;
      v_steps            = 0;
      score_steps        = 0;
      weighted_sum_steps = 0;
      o_steps            = 0;
      gate_steps         = 0;
      up_steps           = 0;
      down_steps         = 0;
      lmhead_steps       = 0;
    end
  endtask

  task automatic sample_step;
    begin
      if (step_valid) begin
        if ((k_tile_idx == 0) && !clear_acc) begin
          $error("gemm_op_scheduler expected clear_acc on first K tile");
          $finish;
        end
        if ((k_tile_idx == (k_tile_count - 1'b1)) && !emit_acc) begin
          $error("gemm_op_scheduler expected emit_acc on final K tile");
          $finish;
        end
        if ((k_tile_idx != (k_tile_count - 1'b1)) && emit_acc) begin
          $error("gemm_op_scheduler asserted emit_acc too early");
          $finish;
        end

        unique case (gemm_mode)
          GEMM_Q:            q_steps++;
          GEMM_K:            k_steps++;
          GEMM_V:            v_steps++;
          GEMM_SCORE: begin
            score_steps++;
            if (kv_head_id != KV_HEAD_ID_W'(q_head_id / KV_GROUPS)) begin
              $error("gemm_op_scheduler kv-head derivation mismatch");
              $finish;
            end
          end
          GEMM_WEIGHTED_SUM: weighted_sum_steps++;
          GEMM_O:            o_steps++;
          GEMM_GATE:         gate_steps++;
          GEMM_UP:           up_steps++;
          GEMM_DOWN:         down_steps++;
          GEMM_LM_HEAD:      lmhead_steps++;
          default: begin
            $error("gemm_op_scheduler issued unexpected mode");
            $finish;
          end
        endcase
      end
    end
  endtask

  initial begin
    clk            = 1'b0;
    rst_n          = 1'b0;
    start          = 1'b0;
    abort_req      = 1'b0;
    lm_head_only   = 1'b0;
    block_start    = 1'b0;
    block_start_id = BLOCK_NONE;
    block_q_head_id = '0;
    block_kv_head_id = '0;
    dma_ready      = 1'b1;
    buffer_ready   = 1'b1;
    step_ready     = 1'b1;
    seq_count      = 16;
    kv_token_count = 64;
    reset_counts();

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;
    sample_step();

    guard = 0;
    while (!done_pulse && (guard < 25000)) begin
      @(negedge clk);
      guard++;
      sample_step();
    end

    if (!done_pulse || busy) begin
      $error("gemm_op_scheduler expected layer schedule completion");
      $finish;
    end

    if ((q_steps != 2048) || (k_steps != 256) || (v_steps != 256) ||
        (score_steps != 32) || (weighted_sum_steps != 64) ||
        (o_steps != 2048) || (gate_steps != 5632) ||
        (up_steps != 5632) || (down_steps != 5632)) begin
      $display("q=%0d k=%0d v=%0d score=%0d wsum=%0d o=%0d gate=%0d up=%0d down=%0d",
        q_steps, k_steps, v_steps, score_steps, weighted_sum_steps, o_steps, gate_steps, up_steps, down_steps);
      $error("gemm_op_scheduler layer schedule count mismatch");
      $finish;
    end

    reset_counts();
    seq_count      = 1;
    kv_token_count = 1;

    @(negedge clk);
    lm_head_only <= 1'b1;
    start        <= 1'b1;
    @(negedge clk);
    start        <= 1'b0;
    lm_head_only <= 1'b0;
    sample_step();

    guard = 0;
    while (!done_pulse && (guard < 400)) begin
      @(negedge clk);
      guard++;
      if (step_valid && (gemm_mode != GEMM_LM_HEAD || block_id != BLOCK_LM_HEAD)) begin
        $error("gemm_op_scheduler lm-head-only mode mismatch");
        $finish;
      end
      sample_step();
    end

    if (!done_pulse || (lmhead_steps != 128)) begin
      $error("gemm_op_scheduler lm-head schedule mismatch");
      $finish;
    end

    @(negedge clk);

    reset_counts();
    seq_count      = 16;
    kv_token_count = 16;

    @(negedge clk);
    block_start_id  <= BLOCK_Q;
    block_q_head_id <= '0;
    block_kv_head_id <= '0;
    block_start     <= 1'b1;
    @(negedge clk);
    block_start <= 1'b0;
    #1;
    sample_step();

    guard = 0;
    while (!done_pulse && (guard < 3000)) begin
      @(negedge clk);
      guard++;
      if (step_valid && ((gemm_mode != GEMM_Q) || (block_id != BLOCK_Q))) begin
        $error("gemm_op_scheduler block-driven Q mode mismatch");
        $finish;
      end
      sample_step();
    end

    if (!done_pulse || (q_steps != 2048)) begin
      $error("gemm_op_scheduler block-driven Q schedule mismatch q_steps=%0d done_pulse=%0b guard=%0d", q_steps, done_pulse, guard);
      $finish;
    end

    @(negedge clk);

    reset_counts();
    @(negedge clk);
    block_start_id   <= BLOCK_WEIGHTED_SUM;
    block_q_head_id  <= Q_HEAD_ID_W'(17);
    block_kv_head_id <= KV_HEAD_ID_W'(2);
    block_start      <= 1'b1;
    @(negedge clk);
    block_start <= 1'b0;
    #1;
    sample_step();

    guard = 0;
    while (!done_pulse && (guard < 64)) begin
      @(negedge clk);
      guard++;
      if (step_valid) begin
        if ((gemm_mode != GEMM_WEIGHTED_SUM) || (block_id != BLOCK_WEIGHTED_SUM)) begin
          $error("gemm_op_scheduler block-driven WEIGHTED_SUM mode mismatch");
          $finish;
        end
        if ((q_head_id != Q_HEAD_ID_W'(17)) || (kv_head_id != KV_HEAD_ID_W'(2))) begin
          $error("gemm_op_scheduler block-driven head forwarding mismatch");
          $finish;
        end
      end
      sample_step();
    end

    if (!done_pulse || (weighted_sum_steps != 2)) begin
      $error("gemm_op_scheduler block-driven WEIGHTED_SUM schedule mismatch weighted_sum_steps=%0d done_pulse=%0b guard=%0d", weighted_sum_steps, done_pulse, guard);
      $finish;
    end

    $display("PASS: tb_gemm_op_scheduler");
    $finish;
  end

endmodule
