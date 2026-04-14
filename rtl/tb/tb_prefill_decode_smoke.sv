`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_prefill_decode_smoke;

  localparam int unsigned TRACE_META_WORDS = 12;
  localparam int unsigned TRACE_GEN_MAX = 2;

  logic [31:0] trace_meta [0:TRACE_META_WORDS-1];
  logic [31:0] trace_generated_tokens [0:TRACE_GEN_MAX-1];

  logic               clk;
  logic               rst_n;
  logic               start;
  logic               abort_req;
  logic               command_info_valid;
  logic               prompt_read_done;
  logic               layer_pass_done;
  logic               lm_head_done;
  logic               token_valid;
  logic [TOKEN_W-1:0] token_id;
  logic               stop_now;
  stop_reason_e       stop_reason_now;

  logic               busy;
  logic               done_pulse;
  logic               error_pulse;
  logic               stop_valid;
  stop_reason_e       stop_reason_out;
  error_code_e        error_code_out;
  logic [COUNT_W-1:0] generated_token_count;
  runtime_mode_e      active_mode;
  logic               prompt_read_start;
  logic               token_writer_start;
  logic               embedding_start;
  logic               runtime_layer_start;
  logic               lm_head_start;
  logic               argmax_start;

  logic                  layer_ctrl_busy;
  logic                  layer_ctrl_done;
  logic                  per_layer_start;
  logic                  block_valid;
  logic                  block_start;
  logic                  layer_ctx_valid;
  runtime_mode_e         layer_runtime_mode;
  logic [LAYER_ID_W-1:0] layer_id;
  logic [LAYER_ID_W-1:0] weight_layer_sel;
  logic [LAYER_ID_W-1:0] kv_layer_sel;
  block_id_e             block_id;
  logic [Q_HEAD_ID_W-1:0] q_head_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;
  logic                  block_done;

  int unsigned prompt_start_count;
  int unsigned layer_pass_count;
  int unsigned token_emit_count;
  logic        lm_done_pending_q;
  logic        token_emit_pending_q;
  error_code_e no_error_code;

  assign no_error_code = ERROR_NONE;

  prefill_decode_controller dut_ctrl (
    .ap_clk                   (clk),
    .ap_rst_n                 (rst_n),
    .start_i                  (start),
    .abort_req_i              (abort_req),
    .launch_mode_i            (MODE_PREFILL),
    .prompt_token_count_i     (trace_meta[4][COUNT_W-1:0]),
    .max_new_tokens_i         (trace_meta[5][COUNT_W-1:0]),
    .command_info_valid_i     (command_info_valid),
    .prompt_read_done_i       (prompt_read_done),
    .layer_pass_done_i        (layer_pass_done),
    .lm_head_done_i           (lm_head_done),
    .token_valid_i            (token_valid),
    .token_id_i               (token_id),
    .stop_now_i               (stop_now),
    .stop_reason_i            (stop_reason_now),
    .error_valid_i            (1'b0),
    .error_code_i             (no_error_code),
    .busy_o                   (busy),
    .done_pulse_o             (done_pulse),
    .error_pulse_o            (error_pulse),
    .stop_valid_o             (stop_valid),
    .stop_reason_o            (stop_reason_out),
    .error_code_o             (error_code_out),
    .generated_token_count_o  (generated_token_count),
    .active_mode_o            (active_mode),
    .prefill_active_o         (),
    .decode_active_o          (),
    .prompt_read_start_o      (prompt_read_start),
    .token_writer_start_o     (token_writer_start),
    .embedding_start_o        (embedding_start),
    .layer_start_o            (runtime_layer_start),
    .lm_head_start_o          (lm_head_start),
    .argmax_start_o           (argmax_start)
  );

  layer_controller dut_layer (
    .ap_clk            (clk),
    .ap_rst_n          (rst_n),
    .start_i           (runtime_layer_start),
    .abort_req_i       (abort_req),
    .runtime_mode_i    (active_mode),
    .block_done_i      (block_done),
    .busy_o            (layer_ctrl_busy),
    .run_done_o        (layer_ctrl_done),
    .layer_start_o     (per_layer_start),
    .layer_ctx_valid_o (layer_ctx_valid),
    .block_valid_o     (block_valid),
    .block_start_o     (block_start),
    .runtime_mode_o    (layer_runtime_mode),
    .layer_id_o        (layer_id),
    .weight_layer_sel_o(weight_layer_sel),
    .kv_layer_sel_o    (kv_layer_sel),
    .block_id_o        (block_id),
    .q_head_id_o       (q_head_id),
    .kv_head_id_o      (kv_head_id)
  );

  stop_condition_unit dut_stop (
    .abort_req_i             (abort_req),
    .emitted_token_valid_i   (token_valid),
    .emitted_token_id_i      (token_id),
    .generated_token_count_i (generated_token_count),
    .max_new_tokens_i        (trace_meta[5][COUNT_W-1:0]),
    .eos_token_id_i          (trace_meta[6][TOKEN_W-1:0]),
    .stop_now_o              (stop_now),
    .stop_reason_o           (stop_reason_now)
  );

  assign layer_pass_done = layer_ctrl_done;

  always #5 clk = ~clk;

  always @(posedge prompt_read_start) begin
    prompt_start_count <= prompt_start_count + 1;
  end

  initial begin
    $readmemh("sim/golden_traces/phase8/rtl/phase8_prefill_decode_runtime.meta.memh", trace_meta);
    $readmemh("sim/golden_traces/phase8/rtl/phase8_prefill_decode_runtime.generated_tokens_expected.memh", trace_generated_tokens);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      block_done          <= 1'b0;
      command_info_valid  <= 1'b0;
      prompt_read_done    <= 1'b0;
      lm_head_done        <= 1'b0;
      token_valid         <= 1'b0;
      token_id            <= '0;
      prompt_start_count  <= 0;
      layer_pass_count    <= 0;
      token_emit_count    <= 0;
      lm_done_pending_q   <= 1'b0;
      token_emit_pending_q <= 1'b0;
    end else begin
      block_done         <= block_start;
      prompt_read_done   <= 1'b0;
      lm_head_done       <= 1'b0;
      token_valid        <= 1'b0;

      if (prompt_read_start) begin
        prompt_read_done <= 1'b1;
      end

      if (layer_ctrl_done) begin
        layer_pass_count <= layer_pass_count + 1;
      end

      if (lm_head_start) begin
        if (!argmax_start) begin
          $error("expected argmax_start alongside lm_head_start");
          $finish;
        end
        token_id     <= trace_generated_tokens[token_emit_count];
        lm_done_pending_q <= 1'b1;
      end else if (lm_done_pending_q) begin
        lm_done_pending_q   <= 1'b0;
        lm_head_done        <= 1'b1;
        token_emit_pending_q <= 1'b1;
      end else if (token_emit_pending_q) begin
        token_emit_pending_q <= 1'b0;
        token_valid          <= 1'b1;
        token_emit_count     <= token_emit_count + 1;
      end
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    abort_req = 1'b0;
    command_info_valid = 1'b0;
    prompt_read_done = 1'b0;
    lm_head_done = 1'b0;
    token_valid = 1'b0;
    token_id = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    @(negedge clk);
    command_info_valid <= 1'b1;

    begin : wait_done
      int unsigned wait_cycles;
      wait_cycles = 0;
      while (!done_pulse) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 20000) begin
          $error("tb_prefill_decode_smoke timeout waiting for done_pulse");
          $finish;
        end
      end
    end

    if (error_pulse || (error_code_out != ERROR_NONE)) begin
      $error("runtime smoke expected no error path");
      $finish;
    end
    if (!stop_valid || (stop_reason_out != STOP_REASON_MAX_TOKENS)) begin
      $error("runtime smoke expected MAX_TOKENS stop, stop_valid=%0b reason=%0d", stop_valid, stop_reason_out);
      $finish;
    end
    if (generated_token_count != trace_meta[10][COUNT_W-1:0]) begin
      $error("generated_token_count mismatch: expected %0d got %0d", trace_meta[10], generated_token_count);
      $finish;
    end
    if (layer_pass_count != trace_meta[8]) begin
      $error("expected %0d layer passes, got %0d", trace_meta[8], layer_pass_count);
      $finish;
    end
    if (token_emit_count != trace_meta[10]) begin
      $error("expected %0d emitted tokens, got %0d", trace_meta[10], token_emit_count);
      $finish;
    end
    if (active_mode != MODE_DECODE) begin
      $error("expected final active_mode to be MODE_DECODE after first generation");
      $finish;
    end

    $display("PASS: tb_prefill_decode_smoke");
    $finish;
  end

endmodule
