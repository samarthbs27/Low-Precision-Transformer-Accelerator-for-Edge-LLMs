`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_hbm_port_router;

  logic                  clk;
  logic                  rst_n;
  logic                  host_cmd_rd_desc_valid;
  logic                  host_cmd_rd_desc_ready;
  dma_desc_t             host_cmd_rd_desc;
  logic                  host_cmd_rd_data_valid;
  logic                  host_cmd_rd_data_ready;
  logic [DMA_BEAT_W-1:0] host_cmd_rd_data;
  logic                  prompt_rd_desc_valid;
  logic                  prompt_rd_desc_ready;
  dma_desc_t             prompt_rd_desc;
  logic                  prompt_rd_data_valid;
  logic                  prompt_rd_data_ready;
  logic [DMA_BEAT_W-1:0] prompt_rd_data;
  logic                  weight_rd_desc_valid;
  logic                  weight_rd_desc_ready;
  dma_desc_t             weight_rd_desc;
  logic                  weight_rd_data_valid;
  logic                  weight_rd_data_ready;
  logic [DMA_BEAT_W-1:0] weight_rd_data;
  logic                  embed_lm_rd_desc_valid;
  logic                  embed_lm_rd_desc_ready;
  dma_desc_t             embed_lm_rd_desc;
  logic                  embed_lm_rd_data_valid;
  logic                  embed_lm_rd_data_ready;
  logic [DMA_BEAT_W-1:0] embed_lm_rd_data;
  logic                  kv_rd_desc_valid;
  logic                  kv_rd_desc_ready;
  dma_desc_t             kv_rd_desc;
  logic                  kv_rd_data_valid;
  logic                  kv_rd_data_ready;
  logic [DMA_BEAT_W-1:0] kv_rd_data;
  logic                  host_status_wr_desc_valid;
  logic                  host_status_wr_desc_ready;
  dma_desc_t             host_status_wr_desc;
  logic                  host_status_wr_data_valid;
  logic                  host_status_wr_data_ready;
  logic [DMA_BEAT_W-1:0] host_status_wr_data;
  logic                  gen_token_wr_desc_valid;
  logic                  gen_token_wr_desc_ready;
  dma_desc_t             gen_token_wr_desc;
  logic                  gen_token_wr_data_valid;
  logic                  gen_token_wr_data_ready;
  logic [DMA_BEAT_W-1:0] gen_token_wr_data;
  logic                  kv_wr_desc_valid;
  logic                  kv_wr_desc_ready;
  dma_desc_t             kv_wr_desc;
  logic                  kv_wr_data_valid;
  logic                  kv_wr_data_ready;
  logic [DMA_BEAT_W-1:0] kv_wr_data;
  logic                  debug_wr_desc_valid;
  logic                  debug_wr_desc_ready;
  dma_desc_t             debug_wr_desc;
  logic                  debug_wr_data_valid;
  logic                  debug_wr_data_ready;
  logic [DMA_BEAT_W-1:0] debug_wr_data;
  logic                  shell_rd_desc_valid;
  logic                  shell_rd_desc_ready;
  dma_desc_t             shell_rd_desc;
  logic                  shell_rd_data_valid;
  logic                  shell_rd_data_ready;
  logic [DMA_BEAT_W-1:0] shell_rd_data;
  logic                  shell_wr_desc_valid;
  logic                  shell_wr_desc_ready;
  dma_desc_t             shell_wr_desc;
  logic                  shell_wr_data_valid;
  logic                  shell_wr_data_ready;
  logic [DMA_BEAT_W-1:0] shell_wr_data;

  hbm_port_router dut (
    .ap_clk(clk), .ap_rst_n(rst_n),
    .host_cmd_rd_desc_valid_i(host_cmd_rd_desc_valid), .host_cmd_rd_desc_ready_o(host_cmd_rd_desc_ready), .host_cmd_rd_desc_i(host_cmd_rd_desc), .host_cmd_rd_data_valid_o(host_cmd_rd_data_valid), .host_cmd_rd_data_ready_i(host_cmd_rd_data_ready), .host_cmd_rd_data_o(host_cmd_rd_data),
    .prompt_rd_desc_valid_i(prompt_rd_desc_valid), .prompt_rd_desc_ready_o(prompt_rd_desc_ready), .prompt_rd_desc_i(prompt_rd_desc), .prompt_rd_data_valid_o(prompt_rd_data_valid), .prompt_rd_data_ready_i(prompt_rd_data_ready), .prompt_rd_data_o(prompt_rd_data),
    .weight_rd_desc_valid_i(weight_rd_desc_valid), .weight_rd_desc_ready_o(weight_rd_desc_ready), .weight_rd_desc_i(weight_rd_desc), .weight_rd_data_valid_o(weight_rd_data_valid), .weight_rd_data_ready_i(weight_rd_data_ready), .weight_rd_data_o(weight_rd_data),
    .embed_lm_rd_desc_valid_i(embed_lm_rd_desc_valid), .embed_lm_rd_desc_ready_o(embed_lm_rd_desc_ready), .embed_lm_rd_desc_i(embed_lm_rd_desc), .embed_lm_rd_data_valid_o(embed_lm_rd_data_valid), .embed_lm_rd_data_ready_i(embed_lm_rd_data_ready), .embed_lm_rd_data_o(embed_lm_rd_data),
    .kv_rd_desc_valid_i(kv_rd_desc_valid), .kv_rd_desc_ready_o(kv_rd_desc_ready), .kv_rd_desc_i(kv_rd_desc), .kv_rd_data_valid_o(kv_rd_data_valid), .kv_rd_data_ready_i(kv_rd_data_ready), .kv_rd_data_o(kv_rd_data),
    .host_status_wr_desc_valid_i(host_status_wr_desc_valid), .host_status_wr_desc_ready_o(host_status_wr_desc_ready), .host_status_wr_desc_i(host_status_wr_desc), .host_status_wr_data_valid_i(host_status_wr_data_valid), .host_status_wr_data_ready_o(host_status_wr_data_ready), .host_status_wr_data_i(host_status_wr_data),
    .gen_token_wr_desc_valid_i(gen_token_wr_desc_valid), .gen_token_wr_desc_ready_o(gen_token_wr_desc_ready), .gen_token_wr_desc_i(gen_token_wr_desc), .gen_token_wr_data_valid_i(gen_token_wr_data_valid), .gen_token_wr_data_ready_o(gen_token_wr_data_ready), .gen_token_wr_data_i(gen_token_wr_data),
    .kv_wr_desc_valid_i(kv_wr_desc_valid), .kv_wr_desc_ready_o(kv_wr_desc_ready), .kv_wr_desc_i(kv_wr_desc), .kv_wr_data_valid_i(kv_wr_data_valid), .kv_wr_data_ready_o(kv_wr_data_ready), .kv_wr_data_i(kv_wr_data),
    .debug_wr_desc_valid_i(debug_wr_desc_valid), .debug_wr_desc_ready_o(debug_wr_desc_ready), .debug_wr_desc_i(debug_wr_desc), .debug_wr_data_valid_i(debug_wr_data_valid), .debug_wr_data_ready_o(debug_wr_data_ready), .debug_wr_data_i(debug_wr_data),
    .shell_rd_desc_valid_o(shell_rd_desc_valid), .shell_rd_desc_ready_i(shell_rd_desc_ready), .shell_rd_desc_o(shell_rd_desc), .shell_rd_data_valid_i(shell_rd_data_valid), .shell_rd_data_ready_o(shell_rd_data_ready), .shell_rd_data_i(shell_rd_data),
    .shell_wr_desc_valid_o(shell_wr_desc_valid), .shell_wr_desc_ready_i(shell_wr_desc_ready), .shell_wr_desc_o(shell_wr_desc), .shell_wr_data_valid_o(shell_wr_data_valid), .shell_wr_data_ready_i(shell_wr_data_ready), .shell_wr_data_o(shell_wr_data)
  );

  always #5 clk = ~clk;

  initial begin
    int unsigned wait_cycles;

    clk = 1'b0; rst_n = 1'b0;
    host_cmd_rd_desc_valid = 1'b0; host_cmd_rd_desc = '0; host_cmd_rd_data_ready = 1'b0;
    prompt_rd_desc_valid = 1'b0; prompt_rd_desc = '0; prompt_rd_data_ready = 1'b0;
    weight_rd_desc_valid = 1'b0; weight_rd_desc = '0; weight_rd_data_ready = 1'b0;
    embed_lm_rd_desc_valid = 1'b0; embed_lm_rd_desc = '0; embed_lm_rd_data_ready = 1'b0;
    kv_rd_desc_valid = 1'b0; kv_rd_desc = '0; kv_rd_data_ready = 1'b0;
    host_status_wr_desc_valid = 1'b0; host_status_wr_desc = '0; host_status_wr_data_valid = 1'b0; host_status_wr_data = '0;
    gen_token_wr_desc_valid = 1'b0; gen_token_wr_desc = '0; gen_token_wr_data_valid = 1'b0; gen_token_wr_data = '0;
    kv_wr_desc_valid = 1'b0; kv_wr_desc = '0; kv_wr_data_valid = 1'b0; kv_wr_data = '0;
    debug_wr_desc_valid = 1'b0; debug_wr_desc = '0; debug_wr_data_valid = 1'b0; debug_wr_data = '0;
    shell_rd_desc_ready = 1'b0; shell_rd_data_valid = 1'b0; shell_rd_data = '0;
    shell_wr_desc_ready = 1'b0; shell_wr_data_ready = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    prompt_rd_desc.region = REGION_HOST_IO; prompt_rd_desc.pseudo_channel = HOST_IO_PC_ID; prompt_rd_desc.byte_count = DMA_BEAT_BYTES; prompt_rd_desc_valid = 1'b1;
    weight_rd_desc.region = REGION_LAYER_WEIGHTS; weight_rd_desc.pseudo_channel = PC_ID_W'(3); weight_rd_desc.byte_count = DMA_BEAT_BYTES; weight_rd_desc_valid = 1'b1;
    shell_rd_desc_ready = 1'b1;
    #1;
    if (!prompt_rd_desc_ready || weight_rd_desc_ready) begin
      $error("read arbitration expected prompt client to win first");
      $finish;
    end
    if (!shell_rd_desc_valid || (shell_rd_desc.region != REGION_HOST_IO)) begin
      $error("shell read descriptor expected prompt request first");
      $finish;
    end

    @(negedge clk);
    prompt_rd_desc_valid = 1'b0;
    prompt_rd_data_ready = 1'b1;
    shell_rd_data = 256'h0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    shell_rd_data_valid = 1'b1;
    #1;
    if (!prompt_rd_data_valid || (prompt_rd_data != 256'h0123_4567_89AB_CDEF_FEDC_BA98_7654_3210)) begin
      $error("prompt read data routing mismatch");
      $finish;
    end
    @(negedge clk);
    shell_rd_data_valid = 1'b0;

    prompt_rd_data_ready = 1'b0;
    wait_cycles = 0;
    while (!(weight_rd_desc_ready && (shell_rd_desc.region == REGION_LAYER_WEIGHTS)) && (wait_cycles < 8)) begin
      @(negedge clk);
      wait_cycles++;
    end
    if (!weight_rd_desc_ready || (shell_rd_desc.region != REGION_LAYER_WEIGHTS)) begin
      $error("weight descriptor expected after prompt read completed");
      $finish;
    end
    weight_rd_desc_valid = 1'b0;

    host_status_wr_desc.region = REGION_HOST_IO; host_status_wr_desc.pseudo_channel = HOST_IO_PC_ID; host_status_wr_data = 256'hAAAA; host_status_wr_desc_valid = 1'b1; host_status_wr_data_valid = 1'b1;
    gen_token_wr_desc.region = REGION_HOST_IO; gen_token_wr_desc.pseudo_channel = HOST_IO_PC_ID; gen_token_wr_data = 256'hBBBB; gen_token_wr_desc_valid = 1'b1; gen_token_wr_data_valid = 1'b1;
    shell_wr_desc_ready = 1'b1; shell_wr_data_ready = 1'b1;
    #1;
    if (!host_status_wr_desc_ready || !host_status_wr_data_ready || gen_token_wr_desc_ready || gen_token_wr_data_ready) begin
      $error("write arbitration expected host-status client to win first");
      $finish;
    end
    if (!shell_wr_desc_valid || !shell_wr_data_valid || (shell_wr_data != 256'hAAAA)) begin
      $error("shell write routing mismatch for highest-priority client");
      $finish;
    end

    host_status_wr_desc_valid = 1'b0; host_status_wr_data_valid = 1'b0;
    #1;
    if (!gen_token_wr_desc_ready || !gen_token_wr_data_ready || (shell_wr_data != 256'hBBBB)) begin
      $error("gen-token write expected after host-status write completed");
      $finish;
    end
    @(negedge clk);
    if (!gen_token_wr_desc_ready || !gen_token_wr_data_ready || (shell_wr_data != 256'hBBBB)) begin
      $error("gen-token write expected after host-status write completed");
      $finish;
    end

    $display("PASS: tb_hbm_port_router");
    $finish;
  end

endmodule
