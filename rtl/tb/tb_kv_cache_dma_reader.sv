`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_kv_cache_dma_reader;

  logic                  clk;
  logic                  rst_n;
  logic                  req_valid;
  logic                  req_ready;
  dma_desc_t             req_desc;
  logic                  busy;
  logic                  done_pulse;
  logic                  rd_desc_valid;
  logic                  rd_desc_ready;
  dma_desc_t             rd_desc;
  logic                  rd_data_valid;
  logic [DMA_BEAT_W-1:0] rd_data;
  logic                  rd_data_ready;
  logic                  kv_valid;
  logic                  kv_ready;
  logic                  kv_is_v;
  act_bus_t              kv_tile;

  kv_cache_dma_reader dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .req_valid_i(req_valid),
    .req_ready_o(req_ready),
    .req_desc_i(req_desc),
    .busy_o(busy),
    .done_pulse_o(done_pulse),
    .rd_desc_valid_o(rd_desc_valid),
    .rd_desc_ready_i(rd_desc_ready),
    .rd_desc_o(rd_desc),
    .rd_data_valid_i(rd_data_valid),
    .rd_data_i(rd_data),
    .rd_data_ready_o(rd_data_ready),
    .kv_valid_o(kv_valid),
    .kv_ready_i(kv_ready),
    .kv_is_v_o(kv_is_v),
    .kv_tile_o(kv_tile)
  );

  always #5 clk = ~clk;

  initial begin
    clk             = 1'b0;
    rst_n           = 1'b0;
    req_valid       = 1'b0;
    req_desc        = '0;
    req_desc.region = REGION_V_CACHE;
    req_desc.byte_count = 64;
    req_desc.layer_id = 7;
    req_desc.kv_head_id = 2;
    req_desc.tile_id = 16'h0021;
    rd_desc_ready    = 1'b0;
    rd_data_valid    = 1'b0;
    rd_data          = '0;
    kv_ready         = 1'b1;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    req_valid <= 1'b1;
    @(negedge clk);
    req_valid <= 1'b0;

    if (!rd_desc_valid || (rd_desc.region != REGION_V_CACHE) || (rd_desc.byte_count != 64)) begin
      $error("kv_cache_dma_reader descriptor mismatch");
      $finish;
    end

    rd_desc_ready <= 1'b1;
    @(negedge clk);
    rd_desc_ready <= 1'b0;

    rd_data = '0;
    for (int idx = 0; idx < DMA_BEAT_BYTES; idx++) begin
      rd_data[(idx * 8) +: 8] = idx[7:0];
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    if (!kv_valid || !kv_is_v || (kv_tile.tag.block_id != BLOCK_V) || kv_tile.tag.is_last) begin
      $error("kv_cache_dma_reader first beat mismatch");
      $finish;
    end

    @(negedge clk);
    rd_data = '0;
    for (int idx = 0; idx < DMA_BEAT_BYTES; idx++) begin
      rd_data[(idx * 8) +: 8] = (8'h40 + idx[7:0]);
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    if (!kv_valid || !kv_tile.tag.is_last || (kv_tile.data[0] != 8'h40)) begin
      $error("kv_cache_dma_reader second beat mismatch");
      $finish;
    end

    @(negedge clk);
    if (!done_pulse) begin
      $error("kv_cache_dma_reader expected completion pulse");
      $finish;
    end
    @(negedge clk);
    if (busy) begin
      $error("kv_cache_dma_reader expected idle after completion");
      $finish;
    end

    $display("PASS: tb_kv_cache_dma_reader");
    $finish;
  end

endmodule
