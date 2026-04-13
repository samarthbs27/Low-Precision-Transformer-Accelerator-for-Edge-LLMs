`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_kv_cache_dma_writer;

  logic                  clk;
  logic                  rst_n;
  logic                  req_valid;
  logic                  req_ready;
  dma_desc_t             req_desc;
  logic                  tile_valid;
  logic                  tile_ready;
  act_bus_t              tile_bus;
  logic                  busy;
  logic                  wr_desc_valid;
  logic                  wr_desc_ready;
  dma_desc_t             wr_desc;
  logic                  wr_data_valid;
  logic                  wr_data_ready;
  logic [DMA_BEAT_W-1:0] wr_data;
  logic [DMA_BEAT_W-1:0] first_pattern;
  logic [DMA_BEAT_W-1:0] second_pattern;

  kv_cache_dma_writer dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .req_valid_i(req_valid),
    .req_ready_o(req_ready),
    .req_desc_i(req_desc),
    .tile_valid_i(tile_valid),
    .tile_ready_o(tile_ready),
    .tile_i(tile_bus),
    .busy_o(busy),
    .wr_desc_valid_o(wr_desc_valid),
    .wr_desc_ready_i(wr_desc_ready),
    .wr_desc_o(wr_desc),
    .wr_data_valid_o(wr_data_valid),
    .wr_data_ready_i(wr_data_ready),
    .wr_data_o(wr_data)
  );

  always #5 clk = ~clk;

  initial begin
    clk            = 1'b0;
    rst_n          = 1'b0;
    req_valid      = 1'b0;
    req_desc       = '0;
    req_desc.addr  = 64'h1234;
    tile_valid     = 1'b0;
    tile_bus       = '0;
    wr_desc_ready  = 1'b0;
    wr_data_ready  = 1'b0;
    first_pattern  = '0;
    second_pattern = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    if (!req_ready || !tile_ready) begin
      $error("kv_cache_dma_writer expected ready when idle");
      $finish;
    end

    for (int idx = 0; idx < DMA_BEAT_BYTES; idx++) begin
      first_pattern[(idx * 8) +: 8] = (8'h20 + idx[7:0]);
      second_pattern[(idx * 8) +: 8] = (8'h80 + idx[7:0]);
    end
    tile_bus.data[DMA_BEAT_BYTES-1:0] = first_pattern;

    @(negedge clk);
    req_valid  <= 1'b1;
    tile_valid <= 1'b1;
    @(negedge clk);
    req_valid  <= 1'b0;
    tile_valid <= 1'b0;

    if (!busy || !wr_desc_valid || !wr_data_valid || (wr_desc.addr != 64'h1234)) begin
      $error("kv_cache_dma_writer expected pending write after handshake");
      $finish;
    end

    tile_bus.data[DMA_BEAT_BYTES-1:0] = second_pattern;

    if (wr_data[7:0] != 8'h20) begin
      $error("kv_cache_dma_writer failed to buffer write data");
      $finish;
    end

    wr_desc_ready <= 1'b1;
    wr_data_ready <= 1'b1;
    @(negedge clk);
    wr_desc_ready <= 1'b0;
    wr_data_ready <= 1'b0;

    if (busy || wr_desc_valid || wr_data_valid) begin
      $error("kv_cache_dma_writer expected idle after sink handshake");
      $finish;
    end

    $display("PASS: tb_kv_cache_dma_writer");
    $finish;
  end

endmodule
