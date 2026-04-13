`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_generated_token_writer;

  logic                  clk;
  logic                  rst_n;
  logic                  start;
  logic [HBM_ADDR_W-1:0] generated_tokens_base_addr;
  logic [COUNT_W-1:0]    generated_tokens_capacity;
  logic                  token_valid;
  logic [TOKEN_W-1:0]    token_id;
  logic                  token_ready;
  logic                  busy;
  logic                  wr_desc_valid;
  logic                  wr_desc_ready;
  dma_desc_t             wr_desc;
  logic                  wr_data_valid;
  logic                  wr_data_ready;
  logic [DMA_BEAT_W-1:0] wr_data;

  generated_token_writer dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .generated_tokens_base_addr_i(generated_tokens_base_addr),
    .generated_tokens_capacity_i(generated_tokens_capacity),
    .token_valid_i(token_valid),
    .token_id_i(token_id),
    .token_ready_o(token_ready),
    .busy_o(busy),
    .wr_desc_valid_o(wr_desc_valid),
    .wr_desc_ready_i(wr_desc_ready),
    .wr_desc_o(wr_desc),
    .wr_data_valid_o(wr_data_valid),
    .wr_data_ready_i(wr_data_ready),
    .wr_data_o(wr_data)
  );

  always #5 clk = ~clk;

  task automatic push_token(
    input logic [TOKEN_W-1:0] value,
    input logic [HBM_ADDR_W-1:0] expected_addr
  );
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!token_ready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("generated_token_writer timeout waiting for token_ready");
          $finish;
        end
      end

      token_id    <= value;
      token_valid <= 1'b1;
      @(negedge clk);
      token_valid <= 1'b0;

      if (!wr_desc_valid || !wr_data_valid || (wr_desc.addr != expected_addr) || (wr_data[TOKEN_W-1:0] != value)) begin
        $error("generated_token_writer write payload mismatch for token %0d", value);
        $finish;
      end

      wr_desc_ready <= 1'b1;
      wr_data_ready <= 1'b1;
      @(negedge clk);
      wr_desc_ready <= 1'b0;
      wr_data_ready <= 1'b0;
    end
  endtask

  initial begin
    clk                       = 1'b0;
    rst_n                     = 1'b0;
    start                     = 1'b0;
    generated_tokens_base_addr = 64'h0000_0000_0000_2000;
    generated_tokens_capacity = 2;
    token_valid               = 1'b0;
    token_id                  = '0;
    wr_desc_ready             = 1'b0;
    wr_data_ready             = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    push_token(7, 64'h0000_0000_0000_2000);
    push_token(8, 64'h0000_0000_0000_2020);
    push_token(9, 64'h0000_0000_0000_2000);

    if (!busy) begin
      $error("generated_token_writer expected armed busy state after writes");
      $finish;
    end

    $display("PASS: tb_generated_token_writer");
    $finish;
  end

endmodule
