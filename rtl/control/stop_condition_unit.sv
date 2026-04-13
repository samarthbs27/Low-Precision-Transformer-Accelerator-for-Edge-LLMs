import tinyllama_pkg::*;

module stop_condition_unit (
  input  logic               abort_req_i,
  input  logic               emitted_token_valid_i,
  input  logic [TOKEN_W-1:0] emitted_token_id_i,
  input  logic [COUNT_W-1:0] generated_token_count_i,
  input  logic [COUNT_W-1:0] max_new_tokens_i,
  input  logic [TOKEN_W-1:0] eos_token_id_i,
  output logic               stop_now_o,
  output stop_reason_e       stop_reason_o
);

  logic [COUNT_W-1:0] next_token_count;
  logic               eos_hit;
  logic               max_token_hit;

  assign next_token_count = generated_token_count_i + 1'b1;
  assign eos_hit          = emitted_token_valid_i && (emitted_token_id_i == eos_token_id_i);
  assign max_token_hit    = emitted_token_valid_i && (max_new_tokens_i != '0) &&
                            (next_token_count >= max_new_tokens_i);

  always_comb begin
    stop_now_o    = 1'b0;
    stop_reason_o = STOP_REASON_NONE;

    if (abort_req_i) begin
      stop_now_o    = 1'b1;
      stop_reason_o = STOP_REASON_HOST_ABORT;
    end else if (eos_hit) begin
      stop_now_o    = 1'b1;
      stop_reason_o = STOP_REASON_EOS;
    end else if (max_token_hit) begin
      stop_now_o    = 1'b1;
      stop_reason_o = STOP_REASON_MAX_TOKENS;
    end
  end

endmodule
