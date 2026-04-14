import tinyllama_pkg::*;

module layer_controller (
  input  logic                   ap_clk,
  input  logic                   ap_rst_n,
  input  logic                   start_i,
  input  logic                   abort_req_i,
  input  runtime_mode_e          runtime_mode_i,
  input  logic                   block_done_i,
  output logic                   busy_o,
  output logic                   run_done_o,
  output logic                   layer_start_o,
  output logic                   layer_ctx_valid_o,
  output logic                   block_valid_o,
  output logic                   block_start_o,
  output runtime_mode_e          runtime_mode_o,
  output logic [LAYER_ID_W-1:0]  layer_id_o,
  output logic [LAYER_ID_W-1:0]  weight_layer_sel_o,
  output logic [LAYER_ID_W-1:0]  kv_layer_sel_o,
  output block_id_e              block_id_o,
  output logic [Q_HEAD_ID_W-1:0] q_head_id_o,
  output logic [KV_HEAD_ID_W-1:0] kv_head_id_o
);

  typedef enum logic [4:0] {
    LCTRL_RMSNORM1       = 5'd0,
    LCTRL_Q              = 5'd1,
    LCTRL_K              = 5'd2,
    LCTRL_V              = 5'd3,
    LCTRL_ROPE           = 5'd4,
    LCTRL_KV_CACHE_WRITE = 5'd5,
    LCTRL_SCORE          = 5'd6,
    LCTRL_CAUSAL_MASK    = 5'd7,
    LCTRL_SOFTMAX        = 5'd8,
    LCTRL_WEIGHTED_SUM   = 5'd9,
    LCTRL_O              = 5'd10,
    LCTRL_RESIDUAL1      = 5'd11,
    LCTRL_REQUANT1       = 5'd12,
    LCTRL_RMSNORM2       = 5'd13,
    LCTRL_GATE           = 5'd14,
    LCTRL_UP             = 5'd15,
    LCTRL_SILU           = 5'd16,
    LCTRL_GLU_MUL        = 5'd17,
    LCTRL_REQUANT2       = 5'd18,
    LCTRL_DOWN           = 5'd19,
    LCTRL_RESIDUAL2      = 5'd20,
    LCTRL_REQUANT3       = 5'd21
  } layer_step_e;

  runtime_mode_e           runtime_mode_q;
  logic [LAYER_ID_W-1:0]   layer_id_q;
  logic                    busy_q;
  layer_step_e             step_q;
  logic [Q_HEAD_ID_W-1:0]  q_head_q;

  function automatic block_id_e block_from_step(
    input layer_step_e step
  );
    begin
      unique case (step)
        LCTRL_RMSNORM1:       block_from_step = BLOCK_RMSNORM1;
        LCTRL_Q:              block_from_step = BLOCK_Q;
        LCTRL_K:              block_from_step = BLOCK_K;
        LCTRL_V:              block_from_step = BLOCK_V;
        LCTRL_ROPE:           block_from_step = BLOCK_ROPE;
        LCTRL_KV_CACHE_WRITE: block_from_step = BLOCK_KV_CACHE_WRITE;
        LCTRL_SCORE:          block_from_step = BLOCK_SCORE;
        LCTRL_CAUSAL_MASK:    block_from_step = BLOCK_CAUSAL_MASK;
        LCTRL_SOFTMAX:        block_from_step = BLOCK_SOFTMAX;
        LCTRL_WEIGHTED_SUM:   block_from_step = BLOCK_WEIGHTED_SUM;
        LCTRL_O:              block_from_step = BLOCK_O;
        LCTRL_RESIDUAL1:      block_from_step = BLOCK_RESIDUAL1;
        LCTRL_REQUANT1,
        LCTRL_REQUANT2,
        LCTRL_REQUANT3:       block_from_step = BLOCK_REQUANTIZE;
        LCTRL_RMSNORM2:       block_from_step = BLOCK_RMSNORM2;
        LCTRL_GATE:           block_from_step = BLOCK_GATE;
        LCTRL_UP:             block_from_step = BLOCK_UP;
        LCTRL_SILU:           block_from_step = BLOCK_SILU;
        LCTRL_GLU_MUL:        block_from_step = BLOCK_GLU_MUL;
        LCTRL_DOWN:           block_from_step = BLOCK_DOWN;
        LCTRL_RESIDUAL2:      block_from_step = BLOCK_RESIDUAL2;
        default:              block_from_step = BLOCK_NONE;
      endcase
    end
  endfunction

  function automatic layer_step_e next_step_after(
    input layer_step_e            step,
    input logic [Q_HEAD_ID_W-1:0] q_head_id
  );
    begin
      unique case (step)
        LCTRL_RMSNORM1:       next_step_after = LCTRL_Q;
        LCTRL_Q:              next_step_after = LCTRL_K;
        LCTRL_K:              next_step_after = LCTRL_V;
        LCTRL_V:              next_step_after = LCTRL_ROPE;
        LCTRL_ROPE:           next_step_after = LCTRL_KV_CACHE_WRITE;
        LCTRL_KV_CACHE_WRITE: next_step_after = LCTRL_SCORE;
        LCTRL_SCORE:          next_step_after = LCTRL_CAUSAL_MASK;
        LCTRL_CAUSAL_MASK:    next_step_after = LCTRL_SOFTMAX;
        LCTRL_SOFTMAX:        next_step_after = LCTRL_WEIGHTED_SUM;
        LCTRL_WEIGHTED_SUM: begin
          if (q_head_id == (N_Q_HEADS - 1)) begin
            next_step_after = LCTRL_O;
          end else begin
            next_step_after = LCTRL_SCORE;
          end
        end
        LCTRL_O:            next_step_after = LCTRL_RESIDUAL1;
        LCTRL_RESIDUAL1:    next_step_after = LCTRL_REQUANT1;
        LCTRL_REQUANT1:     next_step_after = LCTRL_RMSNORM2;
        LCTRL_RMSNORM2:     next_step_after = LCTRL_GATE;
        LCTRL_GATE:         next_step_after = LCTRL_UP;
        LCTRL_UP:           next_step_after = LCTRL_SILU;
        LCTRL_SILU:         next_step_after = LCTRL_GLU_MUL;
        LCTRL_GLU_MUL:      next_step_after = LCTRL_REQUANT2;
        LCTRL_REQUANT2:     next_step_after = LCTRL_DOWN;
        LCTRL_DOWN:         next_step_after = LCTRL_RESIDUAL2;
        LCTRL_RESIDUAL2:    next_step_after = LCTRL_REQUANT3;
        default:            next_step_after = LCTRL_RMSNORM1;
      endcase
    end
  endfunction

  assign busy_o             = busy_q;
  assign layer_ctx_valid_o  = busy_q;
  assign block_valid_o      = busy_q;
  assign runtime_mode_o     = runtime_mode_q;
  assign layer_id_o         = layer_id_q;
  assign weight_layer_sel_o = layer_id_q;
  assign kv_layer_sel_o     = layer_id_q;
  always_comb begin
    if (busy_q) begin
      block_id_o = block_from_step(step_q);
    end else begin
      block_id_o = BLOCK_NONE;
    end
  end
  assign q_head_id_o = (
    (step_q == LCTRL_SCORE) ||
    (step_q == LCTRL_CAUSAL_MASK) ||
    (step_q == LCTRL_SOFTMAX) ||
    (step_q == LCTRL_WEIGHTED_SUM)
  ) ? q_head_q : '0;
  assign kv_head_id_o = (
    (step_q == LCTRL_SCORE) ||
    (step_q == LCTRL_CAUSAL_MASK) ||
    (step_q == LCTRL_SOFTMAX) ||
    (step_q == LCTRL_WEIGHTED_SUM)
  ) ? KV_HEAD_ID_W'(q_head_q / KV_GROUPS) : '0;

  always_ff @(posedge ap_clk) begin
    run_done_o    <= 1'b0;
    layer_start_o <= 1'b0;
    block_start_o <= 1'b0;

    if (!ap_rst_n) begin
      runtime_mode_q <= MODE_PREFILL;
      layer_id_q     <= '0;
      busy_q         <= 1'b0;
      step_q         <= LCTRL_RMSNORM1;
      q_head_q       <= '0;
    end else begin
      if (busy_q && abort_req_i) begin
        run_done_o <= 1'b1;
        busy_q     <= 1'b0;
        step_q     <= LCTRL_RMSNORM1;
        q_head_q   <= '0;
      end else if (!busy_q && start_i) begin
        runtime_mode_q <= runtime_mode_i;
        layer_id_q     <= '0;
        busy_q         <= 1'b1;
        step_q         <= LCTRL_RMSNORM1;
        q_head_q       <= '0;
        layer_start_o  <= 1'b1;
        block_start_o  <= 1'b1;
      end else if (busy_q && block_done_i) begin
        if (step_q == LCTRL_WEIGHTED_SUM && (q_head_q != (N_Q_HEADS - 1))) begin
          q_head_q      <= q_head_q + 1'b1;
          step_q        <= LCTRL_SCORE;
          block_start_o <= 1'b1;
        end else if (step_q == LCTRL_REQUANT3) begin
          if (layer_id_q == (N_LAYERS - 1)) begin
            run_done_o <= 1'b1;
            busy_q     <= 1'b0;
            step_q     <= LCTRL_RMSNORM1;
            q_head_q   <= '0;
          end else begin
            layer_id_q    <= layer_id_q + 1'b1;
            step_q        <= LCTRL_RMSNORM1;
            q_head_q      <= '0;
            layer_start_o <= 1'b1;
            block_start_o <= 1'b1;
          end
        end else begin
          step_q        <= next_step_after(step_q, q_head_q);
          block_start_o <= 1'b1;
          if (step_q == LCTRL_WEIGHTED_SUM) begin
            q_head_q <= '0;
          end
        end
      end
    end
  end

endmodule
