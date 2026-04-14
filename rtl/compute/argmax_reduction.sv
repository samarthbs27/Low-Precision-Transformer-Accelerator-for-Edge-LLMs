import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module argmax_reduction (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  start_i,
  input  logic                  logits_valid_i,
  output logic                  logits_ready_o,
  input  acc_bus_t              logits_i,
  output logic                  token_valid_o,
  input  logic                  token_ready_i,
  output logic [TOKEN_W-1:0]    token_id_o,
  output logic signed [ACC_W-1:0] token_logit_o,
  output logic                  busy_o,
  output logic                  done_pulse_o
);

  logic                        reduce_active_q;
  logic                        have_best_q;
  logic [TOKEN_W-1:0]          best_token_id_q;
  logic signed [ACC_W-1:0]     best_logit_q;
  logic [TOKEN_W-1:0]          out_token_id_q;
  logic signed [ACC_W-1:0]     out_token_logit_q;

  logic [ELEM_COUNT_W-1:0]     effective_elem_count_w;
  logic [TOKEN_W-1:0]          tile_best_token_id_w;
  logic signed [ACC_W-1:0]     tile_best_logit_w;
  logic [TOKEN_W-1:0]          next_best_token_id_w;
  logic signed [ACC_W-1:0]     next_best_logit_w;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] logits_data_w;

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [ELEM_COUNT_W-1:0] count_d;
    begin
      count_d = (elem_count == '0) ? ELEM_COUNT_W'(VOCAB_TILE) : elem_count;
      if (count_d > VOCAB_TILE) begin
        effective_elem_count = ELEM_COUNT_W'(VOCAB_TILE);
      end else begin
        effective_elem_count = count_d;
      end
    end
  endfunction

  function automatic logic candidate_better(
    input logic signed [ACC_W-1:0] lhs_logit,
    input logic [TOKEN_W-1:0]      lhs_token_id,
    input logic signed [ACC_W-1:0] rhs_logit,
    input logic [TOKEN_W-1:0]      rhs_token_id
  );
    begin
      candidate_better =
        (lhs_logit > rhs_logit) ||
        ((lhs_logit == rhs_logit) && (lhs_token_id < rhs_token_id));
    end
  endfunction

  assign logits_data_w = logits_i.data;

  always_comb begin
    logic [TOKEN_W-1:0] local_best_token_id;
    logic signed [ACC_W-1:0] local_best_logit;
    logic [TOKEN_W-1:0] token_base;

    local_best_token_id = '0;
    local_best_logit = logits_data_w[0];
    token_base = TOKEN_W'(logits_i.tag.tile_id) * TOKEN_W'(VOCAB_TILE);

    for (int lane = 0; lane < VOCAB_TILE; lane++) begin
      if (lane == 0) begin
        local_best_logit = logits_data_w[0];
        local_best_token_id = token_base;
      end else if (lane < effective_elem_count_w) begin
        if (candidate_better(logits_data_w[lane], token_base + TOKEN_W'(lane), local_best_logit, local_best_token_id)) begin
          local_best_logit = logits_data_w[lane];
          local_best_token_id = token_base + TOKEN_W'(lane);
        end
      end
    end

    tile_best_token_id_w = local_best_token_id;
    tile_best_logit_w = local_best_logit;

    if (!have_best_q || candidate_better(tile_best_logit_w, tile_best_token_id_w, best_logit_q, best_token_id_q)) begin
      next_best_token_id_w = tile_best_token_id_w;
      next_best_logit_w = tile_best_logit_w;
    end else begin
      next_best_token_id_w = best_token_id_q;
      next_best_logit_w = best_logit_q;
    end
  end

  assign effective_elem_count_w = effective_elem_count(logits_i.tag.elem_count);
  assign logits_ready_o = reduce_active_q && !token_valid_o;
  assign token_valid_o = !reduce_active_q && have_best_q;
  assign token_id_o = out_token_id_q;
  assign token_logit_o = out_token_logit_q;
  assign busy_o = reduce_active_q || token_valid_o;

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      reduce_active_q <= 1'b0;
      have_best_q     <= 1'b0;
      best_token_id_q <= '0;
      best_logit_q    <= '0;
      out_token_id_q  <= '0;
      out_token_logit_q <= '0;
    end else begin
      if (start_i) begin
        reduce_active_q <= 1'b1;
        have_best_q     <= 1'b0;
        best_token_id_q <= '0;
        best_logit_q    <= '0;
        out_token_id_q  <= '0;
        out_token_logit_q <= '0;
      end

      if (logits_valid_i && logits_ready_o) begin
        have_best_q     <= 1'b1;
        best_token_id_q <= next_best_token_id_w;
        best_logit_q    <= next_best_logit_w;

        if (logits_i.tag.is_last) begin
          reduce_active_q  <= 1'b0;
          out_token_id_q   <= next_best_token_id_w;
          out_token_logit_q <= next_best_logit_w;
        end
      end

      if (token_valid_o && token_ready_i) begin
        have_best_q  <= 1'b0;
        done_pulse_o <= 1'b1;
      end
    end
  end

endmodule
