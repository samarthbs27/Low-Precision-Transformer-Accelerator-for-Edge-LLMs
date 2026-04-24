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
  logic               command_info_valid;
  logic               prompt_read_done;
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
  logic               prompt_read_start;
  logic               token_writer_start;
  logic               embedding_start;
  logic               runtime_layer_start;
  logic               lm_head_start;
  logic               argmax_start;

  logic                  layer_ctrl_busy;
  logic                  layer_ctrl_done;
  logic                  per_layer_start;
  logic                  block_start;
  logic                  block_valid;
  logic                  layer_ctx_valid;
  runtime_mode_e         layer_runtime_mode;
  logic [LAYER_ID_W-1:0] layer_id;
  logic [LAYER_ID_W-1:0] weight_layer_sel;
  logic [LAYER_ID_W-1:0] kv_layer_sel;
  block_id_e             block_id;
  logic [Q_HEAD_ID_W-1:0] q_head_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;
  logic                  block_done;

  prefill_decode_controller dut_ctrl (
    .ap_clk                  (clk),
    .ap_rst_n                (rst_n),
    .start_i                 (start),
    .abort_req_i             (abort_req),
    .launch_mode_i           (launch_mode),
    .prompt_token_count_i    (prompt_token_count),
    .max_new_tokens_i        (max_new_tokens),
    .command_info_valid_i    (command_info_valid),
    .prompt_read_done_i      (prompt_read_done),
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
    .prompt_read_start_o     (prompt_read_start),
    .token_writer_start_o    (token_writer_start),
    .embedding_start_o       (embedding_start),
    .layer_start_o           (runtime_layer_start),
    .lm_head_start_o         (lm_head_start),
    .argmax_start_o          (argmax_start)
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
    .max_new_tokens_i        (max_new_tokens),
    .eos_token_id_i          (32'd2),
    .stop_now_o              (stop_now),
    .stop_reason_o           (stop_reason)
  );

  assign layer_pass_done = layer_ctrl_done;

  always #5 clk = ~clk;

  task automatic drive_layer_pass;
    int unsigned expected_layer;
    int unsigned expected_block;
    int unsigned wait_cycles;
    begin
      for (expected_layer = 0; expected_layer < N_LAYERS; expected_layer++) begin
        for (expected_block = 0; expected_block < (6 + (4 * N_Q_HEADS) + 12); expected_block++) begin
          wait_cycles = 0;
          while (!block_start) begin
            @(negedge clk);
            wait_cycles++;
            if (wait_cycles > 256) begin
              $error("layer pass timeout waiting for block_start at expected layer %0d block %0d", expected_layer, expected_block);
              $finish;
            end
          end
          if ((expected_block == 0) && !per_layer_start) begin
            $error("layer_controller expected layer_start on first block of layer %0d", expected_layer);
            $finish;
          end
          if (layer_id !== expected_layer[LAYER_ID_W-1:0]) begin
            $error("layer_controller expected layer_id %0d, got %0d", expected_layer, layer_id);
            $finish;
          end
          if (weight_layer_sel !== layer_id || kv_layer_sel !== layer_id) begin
            $error("layer selectors expected to match layer_id %0d", layer_id);
            $finish;
          end
          if (!block_valid || !layer_ctx_valid) begin
            $error("layer_controller expected valid context on layer %0d block %0d", expected_layer, expected_block);
            $finish;
          end
          block_done <= 1'b1;
          @(negedge clk);
          block_done <= 1'b0;
        end
      end
    end
  endtask

  task automatic wait_for_prompt_start;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!prompt_read_start) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("controller timeout waiting for prompt_read_start");
          $finish;
        end
      end
    end
  endtask

  task automatic wait_for_layer_launch;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!runtime_layer_start) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("controller timeout waiting for runtime_layer_start");
          $finish;
        end
      end
    end
  endtask

  task automatic wait_for_embedding_launch;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!embedding_start) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("controller timeout waiting for embedding_start");
          $finish;
        end
      end
    end
  endtask

  task automatic wait_for_done_reason(
    input stop_reason_e expected_reason
  );
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!done_pulse) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("controller timeout waiting for done_pulse with reason %0d", expected_reason);
          $finish;
        end
      end
      if (!stop_valid || (stop_reason_out != expected_reason)) begin
        $error("controller expected completion reason %0d, done=%0b stop_valid=%0b reason=%0d",
               expected_reason, done_pulse, stop_valid, stop_reason_out);
        $finish;
      end
    end
  endtask

  task automatic wait_for_abort_done_from_run_layers;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!done_pulse) begin
        if (lm_head_start) begin
          $error("controller should not launch LM head after abort from CTRL_RUN_LAYERS");
          $finish;
        end
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("controller timeout waiting for abort completion from CTRL_RUN_LAYERS");
          $finish;
        end
      end
      if (!stop_valid || (stop_reason_out != STOP_REASON_HOST_ABORT)) begin
        $error("controller expected HOST_ABORT completion from CTRL_RUN_LAYERS, done=%0b stop_valid=%0b reason=%0d",
               done_pulse, stop_valid, stop_reason_out);
        $finish;
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
      if (!argmax_start) begin
        $error("controller expected argmax_start alongside lm_head_start");
        $finish;
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
    clk                = 1'b0;
    rst_n              = 1'b0;
    start              = 1'b0;
    abort_req          = 1'b0;
    launch_mode        = MODE_PREFILL;
    prompt_token_count = 8;
    max_new_tokens     = 2;
    command_info_valid = 1'b0;
    prompt_read_done   = 1'b0;
    lm_head_done       = 1'b0;
    token_valid        = 1'b0;
    token_id           = '0;
    error_valid        = 1'b0;
    error_code         = ERROR_NONE;
    block_done         = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!busy || !prefill_active) begin
      $error("prefill controller expected busy prefill-active state after launch");
      $finish;
    end
    if (prompt_read_start || runtime_layer_start || token_writer_start) begin
      $error("controller should wait for command_info_valid before launching runtime children");
      $finish;
    end

    @(negedge clk);
    command_info_valid <= 1'b1;
    wait_for_prompt_start();
    if (!token_writer_start) begin
      $error("controller expected token_writer_start with prompt read launch");
      $finish;
    end

    @(negedge clk);
    prompt_read_done <= 1'b1;
    wait_for_layer_launch();
    if (embedding_start) begin
      $error("controller should not repulse embedding_start when prefill ingress completes");
      $finish;
    end
    @(negedge clk);
    prompt_read_done <= 1'b0;

    drive_layer_pass();
    emit_token(32'd11);

    if (generated_token_count !== 1) begin
      $error("generated_token_count expected 1 after first token, got %0d", generated_token_count);
      $finish;
    end
    if (active_mode != MODE_DECODE || !decode_active) begin
      $error("controller expected decode-active transition after first token");
      $finish;
    end

    wait_for_embedding_launch();
    if (runtime_layer_start) begin
      $error("controller should wait for decode embedding ingress before relaunching layers");
      $finish;
    end
    @(negedge clk);
    prompt_read_done <= 1'b1;
    wait_for_layer_launch();
    @(negedge clk);
    prompt_read_done <= 1'b0;

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

    wait_for_embedding_launch();
    if (!busy || !decode_active || !token_writer_start || prompt_read_start) begin
      $error("decode-only launch expected busy decode-active controller with writer start");
      $finish;
    end
    @(negedge clk);
    prompt_read_done <= 1'b1;
    wait_for_layer_launch();
    @(negedge clk);
    prompt_read_done <= 1'b0;

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
    wait_for_abort_done_from_run_layers();
    @(negedge clk);
    abort_req <= 1'b0;

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

    wait_for_embedding_launch();
    @(negedge clk);
    prompt_read_done <= 1'b1;
    wait_for_layer_launch();
    @(negedge clk);
    prompt_read_done <= 1'b0;
    if (!busy || !decode_active) begin
      $error("max-token scenario expected busy decode-active controller");
      $finish;
    end

    drive_layer_pass();
    emit_token(32'd11);

    wait_for_done_reason(STOP_REASON_MAX_TOKENS);
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

    $display("PASS: tb_prefill_decode_controller");
    $finish;
  end

endmodule
