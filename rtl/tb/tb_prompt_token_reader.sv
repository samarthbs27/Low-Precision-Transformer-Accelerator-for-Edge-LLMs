`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_prompt_token_reader;

  logic                  clk;
  logic                  rst_n;
  logic                  start;
  logic [HBM_ADDR_W-1:0] prompt_tokens_base_addr;
  logic [COUNT_W-1:0]    prompt_token_count;
  logic                  busy;
  logic                  done_pulse;
  logic                  error_valid;
  error_code_e           error_code;
  logic                  rd_desc_valid;
  logic                  rd_desc_ready;
  dma_desc_t             rd_desc;
  logic                  rd_data_valid;
  logic [DMA_BEAT_W-1:0] rd_data;
  logic                  rd_data_ready;
  logic                  token_valid;
  logic                  token_ready;
  token_bus_t            token_bus;

  prompt_token_reader dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .prompt_tokens_base_addr_i(prompt_tokens_base_addr),
    .prompt_token_count_i(prompt_token_count),
    .busy_o(busy),
    .done_pulse_o(done_pulse),
    .error_valid_o(error_valid),
    .error_code_o(error_code),
    .rd_desc_valid_o(rd_desc_valid),
    .rd_desc_ready_i(rd_desc_ready),
    .rd_desc_o(rd_desc),
    .rd_data_valid_i(rd_data_valid),
    .rd_data_i(rd_data),
    .rd_data_ready_o(rd_data_ready),
    .token_valid_o(token_valid),
    .token_ready_i(token_ready),
    .token_o(token_bus)
  );

  always #5 clk = ~clk;

  task automatic expect_token(input logic [TOKEN_W-1:0] expected);
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!token_valid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("prompt_token_reader timeout waiting for token %0d", expected);
          $finish;
        end
      end
      if (token_bus.token_id != expected) begin
        $error("prompt_token_reader expected token %0d, got %0d", expected, token_bus.token_id);
        $finish;
      end
      @(negedge clk);
    end
  endtask

  initial begin
    clk                    = 1'b0;
    rst_n                  = 1'b0;
    start                  = 1'b0;
    prompt_tokens_base_addr = 64'h0000_0000_0000_1000;
    prompt_token_count     = 10;
    rd_desc_ready          = 1'b0;
    rd_data_valid          = 1'b0;
    rd_data                = '0;
    token_ready            = 1'b1;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    if (!busy || !rd_desc_valid) begin
      $error("prompt_token_reader expected busy launch with descriptor valid");
      $finish;
    end
    if (rd_desc.addr != 64'h0000_0000_0000_1000 || rd_desc.byte_count != DMA_BEAT_BYTES) begin
      $error("prompt_token_reader first descriptor mismatch");
      $finish;
    end

    rd_desc_ready <= 1'b1;
    @(negedge clk);
    rd_desc_ready <= 1'b0;

    if (!rd_data_ready) begin
      $error("prompt_token_reader expected rd_data_ready after descriptor handshake");
      $finish;
    end

    rd_data[(0*TOKEN_W) +: TOKEN_W] = 11;
    rd_data[(1*TOKEN_W) +: TOKEN_W] = 22;
    rd_data[(2*TOKEN_W) +: TOKEN_W] = 33;
    rd_data[(3*TOKEN_W) +: TOKEN_W] = 44;
    rd_data[(4*TOKEN_W) +: TOKEN_W] = 55;
    rd_data[(5*TOKEN_W) +: TOKEN_W] = 66;
    rd_data[(6*TOKEN_W) +: TOKEN_W] = 77;
    rd_data[(7*TOKEN_W) +: TOKEN_W] = 88;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    expect_token(11);
    expect_token(22);
    expect_token(33);
    expect_token(44);
    expect_token(55);
    expect_token(66);
    expect_token(77);
    expect_token(88);

    if (!rd_desc_valid || rd_desc.addr != 64'h0000_0000_0000_1020) begin
      $error("prompt_token_reader expected second descriptor at +32 bytes");
      $finish;
    end

    rd_desc_ready <= 1'b1;
    @(negedge clk);
    rd_desc_ready <= 1'b0;

    rd_data = '0;
    rd_data[(0*TOKEN_W) +: TOKEN_W] = 99;
    rd_data[(1*TOKEN_W) +: TOKEN_W] = 100;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    expect_token(99);
    expect_token(100);

    if (!done_pulse || error_valid || (error_code != ERROR_NONE)) begin
      $error("prompt_token_reader expected clean done after final token");
      $finish;
    end

    @(negedge clk);
    if (busy) begin
      $error("prompt_token_reader expected idle after completion");
      $finish;
    end

    $display("PASS: tb_prompt_token_reader");
    $finish;
  end

endmodule
