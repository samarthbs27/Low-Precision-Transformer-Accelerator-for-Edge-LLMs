import tinyllama_pkg::*;

module gemm_op_scheduler (
  input  logic                 ap_clk,
  input  logic                 ap_rst_n,
  input  logic                 start_i,
  input  logic                 abort_req_i,
  input  logic                 lm_head_only_i,
  input  logic                 dma_ready_i,
  input  logic                 buffer_ready_i,
  input  logic                 step_ready_i,
  input  logic [COUNT_W-1:0]   seq_count_i,
  input  logic [COUNT_W-1:0]   kv_token_count_i,
  output logic                 busy_o,
  output logic                 done_pulse_o,
  output logic                 step_valid_o,
  output gemm_mode_e           gemm_mode_o,
  output block_id_e            block_id_o,
  output logic                 clear_acc_o,
  output logic                 emit_acc_o,
  output logic [TILE_ID_W-1:0] m_tile_idx_o,
  output logic [TILE_ID_W-1:0] n_tile_idx_o,
  output logic [TILE_ID_W-1:0] k_tile_idx_o,
  output logic [TILE_ID_W-1:0] m_tile_count_o,
  output logic [TILE_ID_W-1:0] n_tile_count_o,
  output logic [TILE_ID_W-1:0] k_tile_count_o,
  output logic [Q_HEAD_ID_W-1:0]  q_head_id_o,
  output logic [KV_HEAD_ID_W-1:0] kv_head_id_o
);

  typedef enum logic [3:0] {
    SOP_Q            = 4'd0,
    SOP_K            = 4'd1,
    SOP_V            = 4'd2,
    SOP_SCORE        = 4'd3,
    SOP_WEIGHTED_SUM = 4'd4,
    SOP_O            = 4'd5,
    SOP_GATE         = 4'd6,
    SOP_UP           = 4'd7,
    SOP_DOWN         = 4'd8,
    SOP_LM_HEAD      = 4'd9
  } sched_op_e;

  typedef enum logic {
    SCHED_IDLE = 1'b0,
    SCHED_RUN  = 1'b1
  } sched_state_e;

  sched_state_e          state_q;
  sched_op_e             op_q;
  logic                  lm_head_only_q;
  logic [TILE_ID_W-1:0]  m_tile_q;
  logic [TILE_ID_W-1:0]  n_tile_q;
  logic [TILE_ID_W-1:0]  k_tile_q;
  logic [Q_HEAD_ID_W-1:0] q_head_q;
  logic                  step_fire_d;
  logic                  last_m_d;
  logic                  last_n_d;
  logic                  last_k_d;

  function automatic logic [15:0] max_count_or_one(
    input logic [COUNT_W-1:0] value
  );
    begin
      if (value == '0) begin
        max_count_or_one = 16'd1;
      end else begin
        max_count_or_one = value;
      end
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] ceil_div(
    input int unsigned numerator,
    input int unsigned denominator
  );
    int unsigned quotient;
    begin
      quotient = (numerator + denominator - 1) / denominator;
      if (quotient == 0) begin
        ceil_div = TILE_ID_W'(1);
      end else begin
        ceil_div = TILE_ID_W'(quotient);
      end
    end
  endfunction

  function automatic gemm_mode_e gemm_mode_from_op(
    input sched_op_e op
  );
    begin
      unique case (op)
        SOP_Q:            gemm_mode_from_op = GEMM_Q;
        SOP_K:            gemm_mode_from_op = GEMM_K;
        SOP_V:            gemm_mode_from_op = GEMM_V;
        SOP_SCORE:        gemm_mode_from_op = GEMM_SCORE;
        SOP_WEIGHTED_SUM: gemm_mode_from_op = GEMM_WEIGHTED_SUM;
        SOP_O:            gemm_mode_from_op = GEMM_O;
        SOP_GATE:         gemm_mode_from_op = GEMM_GATE;
        SOP_UP:           gemm_mode_from_op = GEMM_UP;
        SOP_DOWN:         gemm_mode_from_op = GEMM_DOWN;
        SOP_LM_HEAD:      gemm_mode_from_op = GEMM_LM_HEAD;
        default:          gemm_mode_from_op = GEMM_NONE;
      endcase
    end
  endfunction

  function automatic block_id_e block_id_from_op(
    input sched_op_e op
  );
    begin
      unique case (op)
        SOP_Q:            block_id_from_op = BLOCK_Q;
        SOP_K:            block_id_from_op = BLOCK_K;
        SOP_V:            block_id_from_op = BLOCK_V;
        SOP_SCORE:        block_id_from_op = BLOCK_SCORE;
        SOP_WEIGHTED_SUM: block_id_from_op = BLOCK_WEIGHTED_SUM;
        SOP_O:            block_id_from_op = BLOCK_O;
        SOP_GATE:         block_id_from_op = BLOCK_GATE;
        SOP_UP:           block_id_from_op = BLOCK_UP;
        SOP_DOWN:         block_id_from_op = BLOCK_DOWN;
        SOP_LM_HEAD:      block_id_from_op = BLOCK_LM_HEAD;
        default:          block_id_from_op = BLOCK_NONE;
      endcase
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] m_tiles_for_op(
    input sched_op_e op,
    input logic [COUNT_W-1:0] seq_count
  );
    begin
      unique case (op)
        SOP_SCORE:        m_tiles_for_op = ceil_div(max_count_or_one(seq_count), SCORE_Q_TILE);
        default:          m_tiles_for_op = ceil_div(max_count_or_one(seq_count), M_TILE);
      endcase
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] n_tiles_for_op(
    input sched_op_e op,
    input logic [COUNT_W-1:0] kv_token_count
  );
    begin
      unique case (op)
        SOP_Q,
        SOP_O,
        SOP_DOWN:         n_tiles_for_op = ceil_div(D_MODEL, N_TILE);
        SOP_K,
        SOP_V:            n_tiles_for_op = ceil_div(N_KV_HEADS * HEAD_DIM, N_TILE);
        SOP_SCORE:        n_tiles_for_op = ceil_div(max_count_or_one(kv_token_count), SCORE_K_TILE);
        SOP_WEIGHTED_SUM: n_tiles_for_op = ceil_div(HEAD_DIM, N_TILE);
        SOP_GATE,
        SOP_UP:           n_tiles_for_op = ceil_div(D_FF, N_TILE);
        SOP_LM_HEAD:      n_tiles_for_op = ceil_div(VOCAB_TILE, N_TILE);
        default:          n_tiles_for_op = TILE_ID_W'(1);
      endcase
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] k_tiles_for_op(
    input sched_op_e op,
    input logic [COUNT_W-1:0] kv_token_count
  );
    begin
      unique case (op)
        SOP_SCORE:        k_tiles_for_op = ceil_div(HEAD_DIM, K_TILE);
        SOP_WEIGHTED_SUM: k_tiles_for_op = ceil_div(max_count_or_one(kv_token_count), K_TILE);
        SOP_DOWN:         k_tiles_for_op = ceil_div(D_FF, K_TILE);
        SOP_LM_HEAD:      k_tiles_for_op = ceil_div(D_MODEL, K_TILE);
        default:          k_tiles_for_op = ceil_div(D_MODEL, K_TILE);
      endcase
    end
  endfunction

  assign busy_o         = (state_q == SCHED_RUN);
  assign gemm_mode_o    = gemm_mode_from_op(op_q);
  assign block_id_o     = block_id_from_op(op_q);
  assign m_tile_idx_o   = m_tile_q;
  assign n_tile_idx_o   = n_tile_q;
  assign k_tile_idx_o   = k_tile_q;
  assign m_tile_count_o = m_tiles_for_op(op_q, seq_count_i);
  assign n_tile_count_o = n_tiles_for_op(op_q, kv_token_count_i);
  assign k_tile_count_o = k_tiles_for_op(op_q, kv_token_count_i);
  assign q_head_id_o    = (op_q == SOP_SCORE || op_q == SOP_WEIGHTED_SUM) ? q_head_q : '0;
  assign kv_head_id_o   = (op_q == SOP_SCORE || op_q == SOP_WEIGHTED_SUM) ? KV_HEAD_ID_W'(q_head_q / KV_GROUPS) : '0;
  assign step_valid_o   = (state_q == SCHED_RUN) && dma_ready_i && buffer_ready_i;
  assign clear_acc_o    = step_valid_o && (k_tile_q == '0);
  assign emit_acc_o     = step_valid_o && (k_tile_q == (k_tile_count_o - 1'b1));
  assign last_m_d       = (m_tile_q == (m_tile_count_o - 1'b1));
  assign last_n_d       = (n_tile_q == (n_tile_count_o - 1'b1));
  assign last_k_d       = (k_tile_q == (k_tile_count_o - 1'b1));
  assign step_fire_d    = step_valid_o && step_ready_i;

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q         <= SCHED_IDLE;
      op_q            <= SOP_Q;
      lm_head_only_q  <= 1'b0;
      m_tile_q        <= '0;
      n_tile_q        <= '0;
      k_tile_q        <= '0;
      q_head_q        <= '0;
    end else begin
      if ((state_q == SCHED_RUN) && abort_req_i) begin
        state_q        <= SCHED_IDLE;
        done_pulse_o   <= 1'b1;
        op_q           <= SOP_Q;
        lm_head_only_q <= 1'b0;
        m_tile_q       <= '0;
        n_tile_q       <= '0;
        k_tile_q       <= '0;
        q_head_q       <= '0;
      end else begin
        unique case (state_q)
          SCHED_IDLE: begin
            if (start_i) begin
              state_q        <= SCHED_RUN;
              op_q           <= lm_head_only_i ? SOP_LM_HEAD : SOP_Q;
              lm_head_only_q <= lm_head_only_i;
              m_tile_q       <= '0;
              n_tile_q       <= '0;
              k_tile_q       <= '0;
              q_head_q       <= '0;
            end
          end

          SCHED_RUN: begin
            if (step_fire_d) begin
              if (!last_k_d) begin
                k_tile_q <= k_tile_q + 1'b1;
              end else if (!last_n_d) begin
                k_tile_q <= '0;
                n_tile_q <= n_tile_q + 1'b1;
              end else if (!last_m_d) begin
                k_tile_q <= '0;
                n_tile_q <= '0;
                m_tile_q <= m_tile_q + 1'b1;
              end else begin
                k_tile_q <= '0;
                n_tile_q <= '0;
                m_tile_q <= '0;

                unique case (op_q)
                  SOP_Q: begin
                    op_q <= SOP_K;
                  end

                  SOP_K: begin
                    op_q <= SOP_V;
                  end

                  SOP_V: begin
                    op_q    <= SOP_SCORE;
                    q_head_q <= '0;
                  end

                  SOP_SCORE: begin
                    op_q <= SOP_WEIGHTED_SUM;
                  end

                  SOP_WEIGHTED_SUM: begin
                    if (q_head_q == (N_Q_HEADS - 1)) begin
                      op_q     <= SOP_O;
                      q_head_q <= '0;
                    end else begin
                      op_q     <= SOP_SCORE;
                      q_head_q <= q_head_q + 1'b1;
                    end
                  end

                  SOP_O: begin
                    op_q <= SOP_GATE;
                  end

                  SOP_GATE: begin
                    op_q <= SOP_UP;
                  end

                  SOP_UP: begin
                    op_q <= SOP_DOWN;
                  end

                  SOP_DOWN,
                  SOP_LM_HEAD: begin
                    state_q        <= SCHED_IDLE;
                    done_pulse_o   <= 1'b1;
                    op_q           <= SOP_Q;
                    lm_head_only_q <= 1'b0;
                    q_head_q       <= '0;
                  end

                  default: begin
                    state_q      <= SCHED_IDLE;
                    done_pulse_o <= 1'b1;
                    op_q         <= SOP_Q;
                    q_head_q     <= '0;
                  end
                endcase
              end
            end
          end

          default: begin
            state_q <= SCHED_IDLE;
          end
        endcase
      end
    end
  end

endmodule
