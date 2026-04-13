`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_embedding_lmhead_dma_reader;

  logic                  clk;
  logic                  rst_n;
  logic                  req_valid;
  logic                  req_ready;
  logic [HBM_ADDR_W-1:0] base_addr;
  logic [31:0]           byte_count;
  logic [LAYER_ID_W-1:0] layer_id;
  tensor_id_e            tensor_id;
  logic [TILE_ID_W-1:0]  tile_id;
  logic                  busy;
  logic                  done_pulse;
  logic                  rd_desc_valid;
  logic                  rd_desc_ready;
  dma_desc_t             rd_desc;
  logic                  rd_data_valid;
  logic [DMA_BEAT_W-1:0] rd_data;
  logic                  rd_data_ready;
  logic                  embed_row_valid;
  logic                  embed_row_ready;
  logic [DMA_BEAT_W-1:0] embed_row;
  logic                  embed_row_last;
  logic                  gamma_valid;
  logic                  gamma_ready;
  logic [DMA_BEAT_W-1:0] gamma_data;
  logic                  gamma_last;
  logic                  lmhead_wt_valid;
  logic                  lmhead_wt_ready;
  wt_bus_t               lmhead_wt;
  logic                  scale_valid;
  logic                  scale_ready;
  tensor_id_e            scale_tensor_id;
  scale_bus_t            scale_bus;

  embedding_lmhead_dma_reader dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .req_valid_i(req_valid),
    .req_ready_o(req_ready),
    .base_addr_i(base_addr),
    .byte_count_i(byte_count),
    .layer_id_i(layer_id),
    .tensor_id_i(tensor_id),
    .tile_id_i(tile_id),
    .busy_o(busy),
    .done_pulse_o(done_pulse),
    .rd_desc_valid_o(rd_desc_valid),
    .rd_desc_ready_i(rd_desc_ready),
    .rd_desc_o(rd_desc),
    .rd_data_valid_i(rd_data_valid),
    .rd_data_i(rd_data),
    .rd_data_ready_o(rd_data_ready),
    .embed_row_valid_o(embed_row_valid),
    .embed_row_ready_i(embed_row_ready),
    .embed_row_o(embed_row),
    .embed_row_last_o(embed_row_last),
    .gamma_valid_o(gamma_valid),
    .gamma_ready_i(gamma_ready),
    .gamma_o(gamma_data),
    .gamma_last_o(gamma_last),
    .lmhead_wt_valid_o(lmhead_wt_valid),
    .lmhead_wt_ready_i(lmhead_wt_ready),
    .lmhead_wt_o(lmhead_wt),
    .scale_valid_o(scale_valid),
    .scale_ready_i(scale_ready),
    .scale_tensor_id_o(scale_tensor_id),
    .scale_o(scale_bus)
  );

  always #5 clk = ~clk;

  task automatic issue_req(
    input tensor_id_e            tensor,
    input logic [31:0]           bytes,
    input logic [HBM_ADDR_W-1:0] addr
  );
    begin
      tensor_id  <= tensor;
      byte_count <= bytes;
      base_addr  <= addr;
      @(negedge clk);
      req_valid <= 1'b1;
      @(negedge clk);
      req_valid <= 1'b0;
      if (!rd_desc_valid || (rd_desc.addr != addr) || (rd_desc.byte_count != bytes)) begin
        $error("embedding_lmhead_dma_reader descriptor mismatch for tensor %0d", tensor);
        $finish;
      end
      rd_desc_ready <= 1'b1;
      @(negedge clk);
      rd_desc_ready <= 1'b0;
    end
  endtask

  initial begin
    clk             = 1'b0;
    rst_n           = 1'b0;
    req_valid       = 1'b0;
    base_addr       = 64'h0000_0000_0200_0000;
    byte_count      = DMA_BEAT_BYTES;
    layer_id        = 4;
    tensor_id       = TENSOR_EMBED;
    tile_id         = 16'h0002;
    rd_desc_ready   = 1'b0;
    rd_data_valid   = 1'b0;
    rd_data         = '0;
    embed_row_ready = 1'b1;
    gamma_ready     = 1'b1;
    lmhead_wt_ready = 1'b1;
    scale_ready     = 1'b1;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    issue_req(TENSOR_EMBED, 64, 64'h0000_0000_0200_0000);
    rd_data = 256'h1111;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    if (!embed_row_valid || embed_row_last) begin
      $error("embedding_lmhead_dma_reader expected first embedding beat");
      $finish;
    end
    @(negedge clk);
    rd_data = 256'h2222;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    if (!embed_row_valid || !embed_row_last || (embed_row != 256'h2222)) begin
      $error("embedding_lmhead_dma_reader expected final embedding beat");
      $finish;
    end
    @(negedge clk);

    issue_req(TENSOR_LM_HEAD, 64, 64'h0000_0000_0300_0000);
    rd_data = 256'h3333;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    if (!lmhead_wt_valid || lmhead_wt.tag.is_last) begin
      $error("embedding_lmhead_dma_reader expected first LM-head beat");
      $finish;
    end
    @(negedge clk);
    rd_data = 256'h4444;
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    if (!lmhead_wt_valid || !lmhead_wt.tag.is_last) begin
      $error("embedding_lmhead_dma_reader expected final LM-head beat");
      $finish;
    end
    @(negedge clk);

    issue_req(TENSOR_SCALE_META, 64, 64'h0000_0000_0400_0000);
    rd_data = '0;
    for (int idx = 0; idx < 8; idx++) begin
      rd_data[(idx * SCALE_W) +: SCALE_W] = (32'h100 + idx);
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    @(negedge clk);
    rd_data = '0;
    for (int idx = 0; idx < 8; idx++) begin
      rd_data[(idx * SCALE_W) +: SCALE_W] = (32'h200 + idx);
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;
    if (!scale_valid || (scale_tensor_id != TENSOR_SCALE_META) ||
        (scale_bus.data[0] != 32'h100) || (scale_bus.data[8] != 32'h200)) begin
      $error("embedding_lmhead_dma_reader scale aggregation mismatch");
      $finish;
    end
    @(negedge clk);
    if (!done_pulse) begin
      $error("embedding_lmhead_dma_reader expected completion pulse");
      $finish;
    end
    @(negedge clk);
    if (busy) begin
      $error("embedding_lmhead_dma_reader expected idle after completion");
      $finish;
    end

    $display("PASS: tb_embedding_lmhead_dma_reader");
    $finish;
  end

endmodule
