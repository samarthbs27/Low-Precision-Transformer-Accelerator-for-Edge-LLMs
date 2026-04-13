`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_prefill_decode_controller;

  logic               clk;
  logic               rst_n;
  logic               start;
  logic               abort_req;
  runtime_mode_e      launch_mode;
  logic [COUNT_W-1:0] prompt_token_count;
  logic [COUNT_W-1:0] max_new_tokens;
  logic               layer_pass_done;
  logic               lm_head_done;
  logic               token_valid;
  logic [TOKEN_W-1:0] token_id;
  logic               stop_now;
  stop_reason_e       stop_reason;
  logic               error_valid;
  error_code_e        error_code;

  logic               busy;
  logic               done_pulse;
  logic               error_pulse;
  logic               stop_valid;
  stop_reason_e       stop_reason_out;
  error_code_e        error_code_out;
  logic [COUNT_W-1:0] generated_token_count;
  runtime_mode_e      active_mode;
  logic               prefill_active;
  logic               decode_active;
  logic               embedding_start;
  logic               runtime_layer_start;
  logic               lm_head_start;

  logic               layer_ctrl_busy;
  logic               layer_ctrl_done;
  logic               per_layer_start;
  logic               layer_ctx_valid;
  runtime_mode_e      layer_runtime_mode;
  logic [LAYER_ID_W-1:0] layer_id;
  logic [LAYER_ID_W-1:0] weight_layer_sel;
  logic [LAYER_ID_W-1:0] kv_layer_sel;
  logic               layer_step_done;

  prefill_decode_controller dut_ctrl (
    .ap_clk                  (clk),
    .ap_rst_n                (rst_n),
    .start_i                 (start),
    .abort_req_i             (abort_req),
    .launch_mode_i           (launch_mode),
    .prompt_token_count_i    (prompt_token_count),
    .max_new_tokens_i        (max_new_tokens),
    .layer_pass_done_i       (layer_pass_done),
    .lm_head_done_i          (lm_head_done),
    .token_valid_i           (token_valid),
    .token_id_i              (token_id),
    .stop_now_i              (stop_now),
    .stop_reason_i           (stop_reason),
    .error_valid_i           (error_valid),
    .error_code_i            (error_code),
    .busy_o                  (busy),
    .done_pulse_o            (done_pulse),
    .error_pulse_o           (error_pulse),
    .stop_valid_o            (stop_valid),
    .stop_reason_o           (stop_reason_out),
    .error_code_o            (error_code_out),
    .generated_token_count_o (generated_token_count),
    .active_mode_o           (active_mode),
    .prefill_active_o        (prefill_active),
    .decode_active_o         (decode_active),
    .embedding_start_o       (embedding_start),
    .layer_start_o           (runtime_layer_start),
    .lm_head_start_o         (lm_head_start)
  );

  layer_controller dut_layer (
    .ap_clk           (clk),
    .ap_rst_n         (rst_n),
    .start_i          (runtime_layer_start),
    .abort_req_i      (abort_req),
    .runtime_mode_i   (active_mode),
    .layer_step_done_i(layer_step_done),
    .busy_o           (layer_ctrl_busy),
    .run_done_o       (layer_ctrl_done),
    .layer_start_o    (per_layer_start),
    .layer_ctx_valid_o(layer_ctx_valid),
    .runtime_mode_o   (layer_runtime_mode),
    .layer_id_o       (layer_id),
    .weight_layer_sel_o(weight_layer_sel),
    .kv_layer_sel_o   (kv_layer_sel)
  );

  stop_condition_unit dut_stop (
    .abort_req_i             (abort_req),
    .emitted_token_valid_i   (token_valid),
    .emitted_token_id_i      (token_id),
    .generated_token_count_i (generated_token_count),
    .max_new_tokens_i        (max_new_tokens),
    .eos_token_id_i          (32'd2),
    .stop_now_o              (stop_now),
    .stop_reason_o           (stop_reason)
  );

  assign layer_pass_done = layer_ctrl_done;

  always #5 clk = ~clk;

  task automatic drive_layer_pass;
    int unsigned expected_layer;
    int unsigned wait_cycles;
    begin
      for (expected_layer = 0; expected_layer < N_LAYERS; expected_layer++) begin
        wait_cycles = 0;
        while (!per_layer_start) begin
          @(negedge clk);
          wait_cycles++;
          if (wait_cycles > 128) begin
            $error("layer pass timeout waiting for per_layer_start at expected layer %0d", expected_layer);
            $finish;
          end
        end
        if (layer_id !== expected_layer[LAYER_ID_W-1:0]) begin
          $error("layer_controller expected layer_id %0d, got %0d", expected_layer, layer_id);
          $finish;
        end
        if (weight_layer_sel !== layer_id || kv_layer_sel !== layer_id) begin
          $error("layer selectors expected to match layer_id %0d", layer_id);
          $finish;
        end
        layer_step_done <= 1'b1;
        @(negedge clk);
        layer_step_done <= 1'b0;
      end
    end
  endtask

  task automatic emit_token(
    input logic [TOKEN_W-1:0] value
  );
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!lm_head_start) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 128) begin
          $error("token emit timeout waiting for lm_head_start");
          $finish;
        end
      end
      lm_head_done <= 1'b1;
      @(negedge clk);
      lm_head_done <= 1'b0;

      token_id    <= value;
      token_valid <= 1'b1;
      @(negedge clk);
      token_valid <= 1'b0;
    end
  endtask

  initial begin
    clk               = 1'b0;
    rst_n             = 1'b0;
    start             = 1'b0;
    abort_req         = 1'b0;
    launch_mode       = MODE_PREFILL;
    prompt_token_count = 8;
    max_new_tokens    = 2;
    lm_head_done      = 1'b0;
    token_valid       = 1'b0;
    token_id          = '0;
    error_valid       = 1'b0;
    error_code        = ERROR_NONE;
    layer_step_done   = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!busy || !prefill_active || !embedding_start || !runtime_layer_start) begin
      $error("prefill controller did not launch prefill as expected");
      $finish;
    end

    drive_layer_pass();
    emit_token(32'd11);

    if (generated_token_count !== 1) begin
      $error("generated_token_count expected 1 after first token, got %0d", generated_token_count);
      $finish;
    end
    if (active_mode != MODE_DECODE || !decode_active) begin
      $error("controller expected to transition into decode after first token");
      $finish;
    end

    drive_layer_pass();
    emit_token(32'd2);

    if (!done_pulse || !stop_valid || (stop_reason_out != STOP_REASON_EOS)) begin
      $error("controller expected EOS completion, done=%0b stop_valid=%0b reason=%0d", done_pulse, stop_valid, stop_reason_out);
      $finish;
    end

    @(negedge clk);
    if (busy) begin
      $error("controller expected idle after EOS completion");
      $finish;
    end

    launch_mode        = MODE_DECODE;
    prompt_token_count = 4;
    max_new_tokens     = 4;
    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!busy || !decode_active) begin
      $error("decode-only launch expected busy decode-active controller");
      $finish;
    end

    begin : abort_wait
      int unsigned wait_cycles;
      wait_cycles = 0;
      while (!per_layer_start) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 128) begin
          $error("abort scenario timeout waiting for per_layer_start");
          $finish;
        end
      end
    end
    abort_req <= 1'b1;
    @(negedge clk);
    abort_req <= 1'b0;

    if (!done_pulse || !stop_valid || (stop_reason_out != STOP_REASON_HOST_ABORT)) begin
      $error("controller expected HOST_ABORT completion, done=%0b stop_valid=%0b reason=%0d", done_pulse, stop_valid, stop_reason_out);
      $finish;
    end

    if (error_pulse || (error_code_out != ERROR_NONE)) begin
      $error("controller expected no error path during abort scenario");
      $finish;
    end

    launch_mode        = MODE_DECODE;
    prompt_token_count = 4;
    max_new_tokens     = 1;
    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!busy || !decode_active) begin
      $error("max-token scenario expected busy decode-active controller");
      $finish;
    end

    drive_layer_pass();
    emit_token(32'd11);

    if (!done_pulse || !stop_valid || (stop_reason_out != STOP_REASON_MAX_TOKENS)) begin
      $error("controller expected MAX_TOKENS completion, done=%0b stop_valid=%0b reason=%0d", done_pulse, stop_valid, stop_reason_out);
      $finish;
    end
    if (generated_token_count != 1) begin
      $error("generated_token_count expected 1 after max-token stop, got %0d", generated_token_count);
      $finish;
    end

    @(negedge clk);
    if (busy) begin
      $error("controller expected idle after MAX_TOKENS completion");
      $finish;
    end

    launch_mode        = MODE_PREFILL;
    prompt_token_count = '0;
    max_new_tokens     = 1;
    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!error_pulse || (error_code_out != ERROR_BAD_DESCRIPTOR)) begin
      $error("controller expected ERROR_BAD_DESCRIPTOR on zero-token prefill launch, error_pulse=%0b code=%0d", error_pulse, error_code_out);
      $finish;
    end

    @(negedge clk);
    if (!done_pulse || busy) begin
      $error("controller expected one-cycle error completion after bad prefill launch, done=%0b busy=%0b", done_pulse, busy);
      $finish;
    end

    $display("PASS: tb_prefill_decode_controller");
    $finish;
  end

endmodule
