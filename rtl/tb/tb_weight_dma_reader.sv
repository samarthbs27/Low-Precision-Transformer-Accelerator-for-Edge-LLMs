`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_weight_dma_reader;

  logic                  clk;
  logic                  rst_n;
  logic                  req_valid;
  logic                  req_ready;
  logic [HBM_ADDR_W-1:0] base_addr;
  logic [31:0]           byte_count;
  logic [LAYER_ID_W-1:0] layer_id;
  tensor_id_e            tensor_id;
  logic [TILE_ID_W-1:0]  output_tile_id;
  logic [TILE_ID_W-1:0]  input_tile_id;
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
  logic                  wt_valid;
  logic                  wt_ready;
  wt_bus_t               wt_bus;

  weight_dma_reader dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .req_valid_i(req_valid),
    .req_ready_o(req_ready),
    .base_addr_i(base_addr),
    .byte_count_i(byte_count),
    .layer_id_i(layer_id),
    .tensor_id_i(tensor_id),
    .output_tile_id_i(output_tile_id),
    .input_tile_id_i(input_tile_id),
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
    .wt_valid_o(wt_valid),
    .wt_ready_i(wt_ready),
    .wt_o(wt_bus)
  );

  always #5 clk = ~clk;

  initial begin
    clk            = 1'b0;
    rst_n          = 1'b0;
    req_valid      = 1'b0;
    base_addr      = 64'h0000_0000_0100_0000;
    byte_count     = 64;
    layer_id       = 5;
    tensor_id      = TENSOR_WDOWN;
    output_tile_id = 16'h0003;
    input_tile_id  = 16'h0001;
    rd_desc_ready  = 1'b0;
    rd_data_valid  = 1'b0;
    rd_data        = '0;
    wt_ready       = 1'b1;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    req_valid <= 1'b1;
    @(negedge clk);
    req_valid <= 1'b0;

    if (!busy || !rd_desc_valid || (rd_desc.byte_count != 64) || (rd_desc.burst_len != 2)) begin
      $error("weight_dma_reader descriptor issue mismatch");
      $finish;
    end

    rd_desc_ready <= 1'b1;
    @(negedge clk);
    rd_desc_ready <= 1'b0;

    if (!rd_data_ready) begin
      $error("weight_dma_reader expected rd_data_ready after descriptor handshake");
      $finish;
    end

    rd_data = '0;
    for (int idx = 0; idx < DMA_BEAT_BYTES; idx++) begin
      rd_data[(idx * 8) +: 8] = idx[7:0];
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    if (!wt_valid || (wt_bus.tag.block_id != BLOCK_DOWN) || (wt_bus.tag.gemm_mode != GEMM_DOWN) ||
        wt_bus.tag.is_last || (wt_bus.tag.elem_count != DMA_BEAT_BYTES)) begin
      $error("weight_dma_reader first streamed beat mismatch");
      $finish;
    end

    @(negedge clk);
    rd_data = '0;
    for (int idx = 0; idx < DMA_BEAT_BYTES; idx++) begin
      rd_data[(idx * 8) +: 8] = (8'h80 + idx[7:0]);
    end
    rd_data_valid <= 1'b1;
    @(negedge clk);
    rd_data_valid <= 1'b0;

    if (!wt_valid || !wt_bus.tag.is_last || (wt_bus.data[0] != 8'h80)) begin
      $error("weight_dma_reader second streamed beat mismatch");
      $finish;
    end

    @(negedge clk);
    if (!done_pulse || error_valid || (error_code != ERROR_NONE)) begin
      $error("weight_dma_reader expected completion pulse");
      $finish;
    end
    @(negedge clk);
    if (busy) begin
      $error("weight_dma_reader expected idle after completion");
      $finish;
    end

    $display("PASS: tb_weight_dma_reader");
    $finish;
  end

endmodule
