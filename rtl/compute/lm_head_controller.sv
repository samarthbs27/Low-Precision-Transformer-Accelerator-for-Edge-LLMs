import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module lm_head_controller (
  input  logic                ap_clk,
  input  logic                ap_rst_n,
  input  logic                start_i,
  input  logic                hidden_valid_i,
  output logic                hidden_ready_o,
  input  act_bus_t            hidden_i,
  input  logic                hidden_scale_valid_i,
  output logic                hidden_scale_ready_o,
  input  scale_bus_t          hidden_scale_i,
  output logic                context_valid_o,
  output act_bus_t            hidden_o,
  output scale_bus_t          hidden_scale_o,
  output logic                sched_start_o,
  output logic [TILE_ID_W-1:0] vocab_tile_idx_o,
  input  logic                sched_done_i,
  input  logic                logits_valid_i,
  output logic                logits_ready_o,
  input  acc_bus_t            logits_i,
  output logic                argmax_valid_o,
  input  logic                argmax_ready_i,
  output acc_bus_t            argmax_o,
  output logic                busy_o,
  output logic                done_pulse_o
);

  localparam int unsigned LMHEAD_TILE_COUNT = (VOCAB_SIZE + VOCAB_TILE - 1) / VOCAB_TILE;
  localparam int unsigned LMHEAD_LAST_ELEMS = VOCAB_SIZE - ((LMHEAD_TILE_COUNT - 1) * VOCAB_TILE);

  typedef enum logic [1:0] {
    LMH_IDLE       = 2'd0,
    LMH_ISSUE      = 2'd1,
    LMH_WAIT_SCHED = 2'd2,
    LMH_WAIT_LOGIT = 2'd3
  } lmh_state_e;

  lmh_state_e            state_q;
  act_bus_t              hidden_q;
  scale_bus_t            hidden_scale_q;
  logic [TILE_ID_W-1:0]  vocab_tile_idx_q;
  acc_bus_t              argmax_bus_d;

  function automatic logic [ELEM_COUNT_W-1:0] elem_count_for_tile(
    input logic [TILE_ID_W-1:0] tile_idx
  );
    begin
      if (tile_idx == TILE_ID_W'(LMHEAD_TILE_COUNT - 1)) begin
        elem_count_for_tile = ELEM_COUNT_W'(LMHEAD_LAST_ELEMS);
      end else begin
        elem_count_for_tile = ELEM_COUNT_W'(VOCAB_TILE);
      end
    end
  endfunction

  assign hidden_ready_o       = (state_q == LMH_IDLE);
  assign hidden_scale_ready_o = (state_q == LMH_IDLE);
  assign context_valid_o      = (state_q != LMH_IDLE);
  assign hidden_o             = hidden_q;
  assign hidden_scale_o       = hidden_scale_q;
  assign sched_start_o        = (state_q == LMH_ISSUE);
  assign vocab_tile_idx_o     = vocab_tile_idx_q;
  assign logits_ready_o       = (state_q == LMH_WAIT_LOGIT) && argmax_ready_i;
  assign argmax_valid_o       = (state_q == LMH_WAIT_LOGIT) && logits_valid_i;
  assign argmax_o             = argmax_bus_d;
  assign busy_o               = (state_q != LMH_IDLE);

  always_comb begin
    argmax_bus_d = logits_i;
    argmax_bus_d.tag.block_id = BLOCK_LM_HEAD;
    argmax_bus_d.tag.gemm_mode = GEMM_LM_HEAD;
    argmax_bus_d.tag.tile_id = vocab_tile_idx_q;
    argmax_bus_d.tag.elem_count = elem_count_for_tile(vocab_tile_idx_q);
    argmax_bus_d.tag.is_partial = (elem_count_for_tile(vocab_tile_idx_q) != VOCAB_TILE);
    argmax_bus_d.tag.is_last = (vocab_tile_idx_q == TILE_ID_W'(LMHEAD_TILE_COUNT - 1));
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q          <= LMH_IDLE;
      hidden_q         <= '0;
      hidden_scale_q   <= '0;
      vocab_tile_idx_q <= '0;
    end else begin
      unique case (state_q)
        LMH_IDLE: begin
          if (start_i && hidden_valid_i && hidden_scale_valid_i) begin
            hidden_q         <= hidden_i;
            hidden_scale_q   <= hidden_scale_i;
            vocab_tile_idx_q <= '0;
            state_q          <= LMH_ISSUE;
          end
        end

        LMH_ISSUE: begin
          state_q <= LMH_WAIT_SCHED;
        end

        LMH_WAIT_SCHED: begin
          if (sched_done_i) begin
            state_q <= LMH_WAIT_LOGIT;
          end
        end

        LMH_WAIT_LOGIT: begin
          if (logits_valid_i && logits_ready_o) begin
            if (vocab_tile_idx_q == TILE_ID_W'(LMHEAD_TILE_COUNT - 1)) begin
              state_q        <= LMH_IDLE;
              done_pulse_o   <= 1'b1;
              vocab_tile_idx_q <= '0;
            end else begin
              vocab_tile_idx_q <= vocab_tile_idx_q + 1'b1;
              state_q          <= LMH_ISSUE;
            end
          end
        end

        default: begin
          state_q <= LMH_IDLE;
        end
      endcase
    end
  end

endmodule
