`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_host_cmd_status_mgr;

  logic                  clk;
  logic                  rst_n;
  logic                  start;
  logic [HBM_ADDR_W-1:0] cmd_base_addr;
  logic [HBM_ADDR_W-1:0] status_base_addr;
  logic                  busy;
  logic                  done_pulse;
  logic                  error_valid;
  error_code_e           error_code;
  logic                  stop_valid;
  stop_reason_e          stop_reason;
  logic [COUNT_W-1:0]    generated_token_count;
  logic [TOKEN_W-1:0]    last_token_id;
  logic [LAYER_ID_W-1:0] current_layer;
  block_id_e             current_block;

  logic                  command_info_valid;
  logic [HBM_ADDR_W-1:0] prompt_tokens_base_addr;
  logic [HBM_ADDR_W-1:0] generated_tokens_base_addr;
  logic [COUNT_W-1:0]    generated_tokens_capacity;
  logic                  cmd_read_desc_valid;
  logic                  cmd_read_desc_ready;
  dma_desc_t             cmd_read_desc;
  logic                  cmd_read_data_valid;
  logic [DMA_BEAT_W-1:0] cmd_read_data;
  logic                  cmd_read_data_ready;
  logic                  status_write_desc_valid;
  logic                  status_write_desc_ready;
  dma_desc_t             status_write_desc;
  logic                  status_write_data_valid;
  logic                  status_write_data_ready;
  logic [DMA_BEAT_W-1:0] status_write_data;

  host_cmd_status_mgr dut (
    .ap_clk                    (clk),
    .ap_rst_n                  (rst_n),
    .start_i                   (start),
    .cmd_base_addr_i           (cmd_base_addr),
    .status_base_addr_i        (status_base_addr),
    .busy_i                    (busy),
    .done_pulse_i              (done_pulse),
    .error_valid_i             (error_valid),
    .error_code_i              (error_code),
    .stop_valid_i              (stop_valid),
    .stop_reason_i             (stop_reason),
    .generated_token_count_i   (generated_token_count),
    .last_token_id_i           (last_token_id),
    .current_layer_i           (current_layer),
    .current_block_i           (current_block),
    .command_info_valid_o      (command_info_valid),
    .prompt_tokens_base_addr_o (prompt_tokens_base_addr),
    .generated_tokens_base_addr_o(generated_tokens_base_addr),
    .generated_tokens_capacity_o(generated_tokens_capacity),
    .cmd_read_desc_valid_o     (cmd_read_desc_valid),
    .cmd_read_desc_ready_i     (cmd_read_desc_ready),
    .cmd_read_desc_o           (cmd_read_desc),
    .cmd_read_data_valid_i     (cmd_read_data_valid),
    .cmd_read_data_i           (cmd_read_data),
    .cmd_read_data_ready_o     (cmd_read_data_ready),
    .status_write_desc_valid_o (status_write_desc_valid),
    .status_write_desc_ready_i (status_write_desc_ready),
    .status_write_desc_o       (status_write_desc),
    .status_write_data_valid_o (status_write_data_valid),
    .status_write_data_ready_i (status_write_data_ready),
    .status_write_data_o       (status_write_data)
  );

  always #5 clk = ~clk;

  task automatic wait_for_cmd_desc;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!cmd_read_desc_valid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("host_cmd_status_mgr timeout waiting for cmd_read_desc_valid");
          $finish;
        end
      end
    end
  endtask

  task automatic wait_for_status_write;
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!(status_write_desc_valid && status_write_data_valid)) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("host_cmd_status_mgr timeout waiting for status write valid");
          $finish;
        end
      end
    end
  endtask

  logic [31:0] status_words [0:HOST_BLOCK_WORDS-1];

  initial begin
    clk                    = 1'b0;
    rst_n                  = 1'b0;
    start                  = 1'b0;
    cmd_base_addr          = 64'h0000_0000_0000_1000;
    status_base_addr       = 64'h0000_0000_0000_2000;
    busy                   = 1'b0;
    done_pulse             = 1'b0;
    error_valid            = 1'b0;
    error_code             = ERROR_NONE;
    stop_valid             = 1'b0;
    stop_reason            = STOP_REASON_NONE;
    generated_token_count  = '0;
    last_token_id          = '0;
    current_layer          = '0;
    current_block          = BLOCK_NONE;
    cmd_read_desc_ready    = 1'b0;
    cmd_read_data_valid    = 1'b0;
    cmd_read_data          = '0;
    status_write_desc_ready = 1'b0;
    status_write_data_ready = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    wait_for_status_write();
    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      status_words[word_idx] = status_write_data[word_idx*32 +: 32];
    end
    if (!status_words[HOST_STATUS_WORD_STATUS][STATUS_BUSY_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_DONE_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_VALID_BIT]) begin
      $error("launch status payload mismatch");
      $finish;
    end
    if (status_words[HOST_STATUS_WORD_GEN_COUNT] != 0 ||
        status_words[HOST_STATUS_WORD_LAST_TOKEN] != 0 ||
        status_words[HOST_STATUS_WORD_VERSION] != RTL_VERSION_WORD) begin
      $error("launch status data mismatch");
      $finish;
    end
    status_write_desc_ready <= 1'b1;
    status_write_data_ready <= 1'b1;
    @(negedge clk);
    status_write_desc_ready <= 1'b0;
    status_write_data_ready <= 1'b0;

    wait_for_cmd_desc();
    if (cmd_read_desc.region != REGION_HOST_IO ||
        cmd_read_desc.write_not_read != 1'b0 ||
        cmd_read_desc.pseudo_channel != HOST_IO_PC_ID ||
        cmd_read_desc.addr != cmd_base_addr ||
        cmd_read_desc.byte_count != HOST_BLOCK_BYTES ||
        cmd_read_desc.burst_len != 1) begin
      $error("cmd_read_desc fields mismatch");
      $finish;
    end

    cmd_read_desc_ready <= 1'b1;
    @(negedge clk);
    cmd_read_desc_ready <= 1'b0;

    if (!cmd_read_data_ready) begin
      $error("cmd_read_data_ready expected HIGH while waiting for command beat");
      $finish;
    end

    cmd_read_data[(HOST_CMD_WORD_PROMPT_BASE_LO*32) +: 32] = 32'hAAA0_0000;
    cmd_read_data[(HOST_CMD_WORD_PROMPT_BASE_HI*32) +: 32] = 32'h0000_0001;
    cmd_read_data[(HOST_CMD_WORD_GEN_BASE_LO*32) +: 32]    = 32'hBBB0_0000;
    cmd_read_data[(HOST_CMD_WORD_GEN_BASE_HI*32) +: 32]    = 32'h0000_0002;
    cmd_read_data[(HOST_CMD_WORD_GEN_CAPACITY*32) +: 32]   = 32'd64;
    cmd_read_data_valid <= 1'b1;
    @(negedge clk);
    cmd_read_data_valid <= 1'b0;

    @(negedge clk);
    if (!command_info_valid ||
        (prompt_tokens_base_addr != 64'h0000_0001_AAA0_0000) ||
        (generated_tokens_base_addr != 64'h0000_0002_BBB0_0000) ||
        (generated_tokens_capacity != 64)) begin
      $error("parsed command info mismatch");
      $finish;
    end

    busy                  = 1'b0;
    generated_token_count = 5;
    last_token_id         = 32'd77;
    current_layer         = 6;
    current_block         = BLOCK_LM_HEAD;
    stop_reason           = STOP_REASON_MAX_TOKENS;
    stop_valid            = 1'b1;
    done_pulse            = 1'b1;

    @(negedge clk);
    stop_valid  <= 1'b0;
    done_pulse  <= 1'b0;

    wait_for_status_write();
    if (status_write_desc.region != REGION_HOST_IO ||
        status_write_desc.write_not_read != 1'b1 ||
        status_write_desc.pseudo_channel != HOST_IO_PC_ID ||
        status_write_desc.addr != status_base_addr ||
        status_write_desc.byte_count != HOST_BLOCK_BYTES ||
        status_write_desc.burst_len != 1) begin
      $error("status_write_desc fields mismatch");
      $finish;
    end

    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      status_words[word_idx] = status_write_data[word_idx*32 +: 32];
    end

    if (!status_words[HOST_STATUS_WORD_STATUS][STATUS_DONE_BIT] ||
        !status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_VALID_BIT] ||
        (status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] != STOP_REASON_MAX_TOKENS)) begin
      $error("status payload status word mismatch");
      $finish;
    end

    if (status_words[HOST_STATUS_WORD_GEN_COUNT] != 5 ||
        status_words[HOST_STATUS_WORD_LAST_TOKEN] != 77 ||
        status_words[HOST_STATUS_WORD_CUR_LAYER] != 6 ||
        status_words[HOST_STATUS_WORD_CUR_BLOCK] != BLOCK_LM_HEAD ||
        status_words[HOST_STATUS_WORD_VERSION] != RTL_VERSION_WORD) begin
      $error("status payload data words mismatch");
      $finish;
    end

    status_write_desc_ready <= 1'b1;
    status_write_data_ready <= 1'b1;
    @(negedge clk);
    status_write_desc_ready <= 1'b0;
    status_write_data_ready <= 1'b0;

    @(negedge clk);
    if (status_write_desc_valid || status_write_data_valid) begin
      $error("status write valids expected LOW after handshake");
      $finish;
    end

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (command_info_valid) begin
      $error("command_info_valid expected LOW immediately after relaunch");
      $finish;
    end

    wait_for_status_write();
    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      status_words[word_idx] = status_write_data[word_idx*32 +: 32];
    end
    if (!status_words[HOST_STATUS_WORD_STATUS][STATUS_BUSY_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_DONE_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_VALID_BIT]) begin
      $error("relaunch status payload mismatch");
      $finish;
    end
    status_write_desc_ready <= 1'b1;
    status_write_data_ready <= 1'b1;
    @(negedge clk);
    status_write_desc_ready <= 1'b0;
    status_write_data_ready <= 1'b0;

    wait_for_cmd_desc();
    if (cmd_read_desc.addr != cmd_base_addr) begin
      $error("relaunch cmd_read_desc expected original cmd_base_addr, got 0x%016h", cmd_read_desc.addr);
      $finish;
    end

    cmd_read_desc_ready <= 1'b1;
    @(negedge clk);
    cmd_read_desc_ready <= 1'b0;

    if (!cmd_read_data_ready) begin
      $error("cmd_read_data_ready expected HIGH during relaunch command fetch");
      $finish;
    end

    cmd_read_data[(HOST_CMD_WORD_PROMPT_BASE_LO*32) +: 32] = 32'hCCC0_0000;
    cmd_read_data[(HOST_CMD_WORD_PROMPT_BASE_HI*32) +: 32] = 32'h0000_0003;
    cmd_read_data[(HOST_CMD_WORD_GEN_BASE_LO*32) +: 32]    = 32'hDDD0_0000;
    cmd_read_data[(HOST_CMD_WORD_GEN_BASE_HI*32) +: 32]    = 32'h0000_0004;
    cmd_read_data[(HOST_CMD_WORD_GEN_CAPACITY*32) +: 32]   = 32'd96;
    cmd_read_data_valid <= 1'b1;
    @(negedge clk);
    cmd_read_data_valid <= 1'b0;

    @(negedge clk);
    if (!command_info_valid ||
        (prompt_tokens_base_addr != 64'h0000_0003_CCC0_0000) ||
        (generated_tokens_base_addr != 64'h0000_0004_DDD0_0000) ||
        (generated_tokens_capacity != 96)) begin
      $error("relaunch parsed command info mismatch");
      $finish;
    end

    generated_token_count = 9;
    last_token_id         = 32'd101;
    current_layer         = 3;
    current_block         = BLOCK_UP;
    error_code            = ERROR_INTERNAL_ASSERT;
    error_valid           = 1'b1;

    @(negedge clk);
    error_valid <= 1'b0;

    wait_for_status_write();

    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      status_words[word_idx] = status_write_data[word_idx*32 +: 32];
    end

    if (status_words[HOST_STATUS_WORD_STATUS][STATUS_DONE_BIT] ||
        !status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_BIT] ||
        status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_VALID_BIT] ||
        (status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] != STOP_REASON_NONE) ||
        (status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB] != ERROR_INTERNAL_ASSERT)) begin
      $error("error-only status payload mismatch");
      $finish;
    end

    if (status_words[HOST_STATUS_WORD_GEN_COUNT] != 9 ||
        status_words[HOST_STATUS_WORD_LAST_TOKEN] != 101 ||
        status_words[HOST_STATUS_WORD_CUR_LAYER] != 3 ||
        status_words[HOST_STATUS_WORD_CUR_BLOCK] != BLOCK_UP ||
        status_words[HOST_STATUS_WORD_VERSION] != RTL_VERSION_WORD) begin
      $error("error-only status payload data mismatch");
      $finish;
    end

    status_write_desc_ready <= 1'b1;
    status_write_data_ready <= 1'b1;
    @(negedge clk);
    status_write_desc_ready <= 1'b0;
    status_write_data_ready <= 1'b0;

    @(negedge clk);
    if (status_write_desc_valid || status_write_data_valid) begin
      $error("error-only status write valids expected LOW after handshake");
      $finish;
    end

    $display("PASS: tb_host_cmd_status_mgr");
    $finish;
  end

endmodule
