import tinyllama_pkg::*;

module prefill_decode_controller (
  input  logic               ap_clk,
  input  logic               ap_rst_n,
  input  logic               start_i,
  input  logic               abort_req_i,
  input  runtime_mode_e      launch_mode_i,
  input  logic [COUNT_W-1:0] prompt_token_count_i,
  input  logic [COUNT_W-1:0] max_new_tokens_i,
  input  logic               command_info_valid_i,
  input  logic               prompt_read_done_i,
  input  logic               layer_pass_done_i,
  input  logic               lm_head_done_i,
  input  logic               token_valid_i,
  input  logic [TOKEN_W-1:0] token_id_i,
  input  logic               stop_now_i,
  input  stop_reason_e       stop_reason_i,
  input  logic               error_valid_i,
  input  error_code_e        error_code_i,
  output logic               busy_o,
  output logic               done_pulse_o,
  output logic               error_pulse_o,
  output logic               stop_valid_o,
  output stop_reason_e       stop_reason_o,
  output error_code_e        error_code_o,
  output logic [COUNT_W-1:0] generated_token_count_o,
  output runtime_mode_e      active_mode_o,
  output logic               prefill_active_o,
  output logic               decode_active_o,
  output logic               prompt_read_start_o,
  output logic               token_writer_start_o,
  output logic               embedding_start_o,
  output logic               layer_start_o,
  output logic               lm_head_start_o,
  output logic               argmax_start_o
);

  typedef enum logic [2:0] {
    CTRL_IDLE       = 3'd0,
    CTRL_WAIT_CMD   = 3'd1,
    CTRL_WAIT_PROMPT = 3'd2,
    CTRL_RUN_LAYERS = 3'd3,
    CTRL_RUN_LMHEAD = 3'd4,
    CTRL_WAIT_TOKEN = 3'd5,
    CTRL_DONE       = 3'd6,
    CTRL_ERROR      = 3'd7
  } ctrl_state_e;

  ctrl_state_e     state_q;
  runtime_mode_e   active_mode_q;
  logic [COUNT_W-1:0] generated_token_count_q;
  stop_reason_e    stop_reason_q;
  error_code_e     error_code_q;

  assign busy_o                  = (state_q != CTRL_IDLE);
  assign active_mode_o           = active_mode_q;
  assign generated_token_count_o = generated_token_count_q;
  assign prefill_active_o        = (state_q != CTRL_IDLE) && (active_mode_q == MODE_PREFILL);
  assign decode_active_o         = (state_q != CTRL_IDLE) && (active_mode_q == MODE_DECODE);
  assign stop_reason_o           = stop_reason_q;
  assign error_code_o            = error_code_q;

  always_ff @(posedge ap_clk) begin
    done_pulse_o      <= 1'b0;
    error_pulse_o     <= 1'b0;
    stop_valid_o      <= 1'b0;
    prompt_read_start_o <= 1'b0;
    token_writer_start_o <= 1'b0;
    embedding_start_o <= 1'b0;
    layer_start_o     <= 1'b0;
    lm_head_start_o   <= 1'b0;
    argmax_start_o    <= 1'b0;

    if (!ap_rst_n) begin
      state_q                  <= CTRL_IDLE;
      active_mode_q            <= MODE_PREFILL;
      generated_token_count_q  <= '0;
      stop_reason_q            <= STOP_REASON_NONE;
      error_code_q             <= ERROR_NONE;
    end else begin
      unique case (state_q)
        CTRL_IDLE: begin
          if (start_i) begin
            if ((launch_mode_i == MODE_PREFILL) && (prompt_token_count_i == '0)) begin
              error_code_q  <= ERROR_BAD_DESCRIPTOR;
              error_pulse_o <= 1'b1;
              state_q       <= CTRL_ERROR;
            end else begin
              active_mode_q           <= launch_mode_i;
              generated_token_count_q <= '0;
              stop_reason_q           <= STOP_REASON_NONE;
              error_code_q            <= ERROR_NONE;
              state_q                 <= CTRL_WAIT_CMD;
            end
          end
        end

        CTRL_WAIT_CMD: begin
          if (error_valid_i) begin
            error_code_q  <= error_code_i;
            error_pulse_o <= 1'b1;
            state_q       <= CTRL_ERROR;
          end else if (abort_req_i) begin
            stop_reason_q <= STOP_REASON_HOST_ABORT;
            stop_valid_o  <= 1'b1;
            done_pulse_o  <= 1'b1;
            state_q       <= CTRL_DONE;
          end else if (command_info_valid_i) begin
            token_writer_start_o <= 1'b1;

            if (active_mode_q == MODE_PREFILL) begin
              prompt_read_start_o <= 1'b1;
              state_q             <= CTRL_WAIT_PROMPT;
            end else begin
              embedding_start_o <= 1'b1;
              layer_start_o     <= 1'b1;
              state_q           <= CTRL_RUN_LAYERS;
            end
          end
        end

        CTRL_WAIT_PROMPT: begin
          if (error_valid_i) begin
            error_code_q  <= error_code_i;
            error_pulse_o <= 1'b1;
            state_q       <= CTRL_ERROR;
          end else if (abort_req_i) begin
            stop_reason_q <= STOP_REASON_HOST_ABORT;
            stop_valid_o  <= 1'b1;
            done_pulse_o  <= 1'b1;
            state_q       <= CTRL_DONE;
          end else if (prompt_read_done_i) begin
            embedding_start_o <= 1'b1;
            layer_start_o     <= 1'b1;
            state_q           <= CTRL_RUN_LAYERS;
          end
        end

        CTRL_RUN_LAYERS: begin
          if (error_valid_i) begin
            error_code_q  <= error_code_i;
            error_pulse_o <= 1'b1;
            state_q       <= CTRL_ERROR;
          end else if (abort_req_i) begin
            stop_reason_q <= STOP_REASON_HOST_ABORT;
            stop_valid_o  <= 1'b1;
            done_pulse_o  <= 1'b1;
            state_q       <= CTRL_DONE;
          end else if (layer_pass_done_i) begin
            if ((active_mode_q == MODE_PREFILL) && (max_new_tokens_i == '0)) begin
              done_pulse_o <= 1'b1;
              state_q      <= CTRL_DONE;
            end else begin
              lm_head_start_o <= 1'b1;
              argmax_start_o  <= 1'b1;
              state_q         <= CTRL_RUN_LMHEAD;
            end
          end
        end

        CTRL_RUN_LMHEAD: begin
          if (error_valid_i) begin
            error_code_q  <= error_code_i;
            error_pulse_o <= 1'b1;
            state_q       <= CTRL_ERROR;
          end else if (abort_req_i) begin
            stop_reason_q <= STOP_REASON_HOST_ABORT;
            stop_valid_o  <= 1'b1;
            done_pulse_o  <= 1'b1;
            state_q       <= CTRL_DONE;
          end else if (lm_head_done_i) begin
            state_q <= CTRL_WAIT_TOKEN;
          end
        end

        CTRL_WAIT_TOKEN: begin
          if (error_valid_i) begin
            error_code_q  <= error_code_i;
            error_pulse_o <= 1'b1;
            state_q       <= CTRL_ERROR;
          end else if (abort_req_i) begin
            stop_reason_q <= STOP_REASON_HOST_ABORT;
            stop_valid_o  <= 1'b1;
            done_pulse_o  <= 1'b1;
            state_q       <= CTRL_DONE;
          end else if (token_valid_i) begin
            generated_token_count_q <= generated_token_count_q + 1'b1;

            if (stop_now_i) begin
              stop_reason_q <= stop_reason_i;
              stop_valid_o  <= 1'b1;
              done_pulse_o  <= 1'b1;
              state_q       <= CTRL_DONE;
            end else begin
              active_mode_q     <= MODE_DECODE;
              embedding_start_o <= 1'b1;
              layer_start_o     <= 1'b1;
              state_q           <= CTRL_RUN_LAYERS;
            end
          end
        end

        CTRL_DONE: begin
          state_q <= CTRL_IDLE;
        end

        CTRL_ERROR: begin
          done_pulse_o <= 1'b1;
          state_q      <= CTRL_IDLE;
        end

        default: begin
          state_q <= CTRL_IDLE;
        end
      endcase

    end
  end

endmodule
