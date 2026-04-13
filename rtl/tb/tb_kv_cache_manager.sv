`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_kv_cache_manager;

  logic                    clk;
  logic                    rst_n;
  logic                    issue_read;
  logic                    issue_write;
  hbm_region_e             region;
  logic [HBM_ADDR_W-1:0]   base_addr;
  logic [LAYER_ID_W-1:0]   layer_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;
  logic [POS_W-1:0]        token_base;
  logic [COUNT_W-1:0]      seq_count;
  logic                    read_desc_valid;
  logic                    read_desc_ready;
  dma_desc_t               read_desc;
  logic                    write_desc_valid;
  logic                    write_desc_ready;
  dma_desc_t               write_desc;
  logic [HBM_ADDR_W-1:0]   expected_addr;

  kv_cache_manager dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .issue_read_i(issue_read),
    .issue_write_i(issue_write),
    .region_i(region),
    .base_addr_i(base_addr),
    .layer_id_i(layer_id),
    .kv_head_id_i(kv_head_id),
    .token_base_i(token_base),
    .seq_count_i(seq_count),
    .read_desc_valid_o(read_desc_valid),
    .read_desc_ready_i(read_desc_ready),
    .read_desc_o(read_desc),
    .write_desc_valid_o(write_desc_valid),
    .write_desc_ready_i(write_desc_ready),
    .write_desc_o(write_desc)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    issue_read = 1'b0;
    issue_write = 1'b0;
    region = REGION_K_CACHE;
    base_addr = 64'h0000_0000_0010_0000;
    layer_id = 2;
    kv_head_id = 1;
    token_base = 4;
    seq_count = 3;
    read_desc_ready = 1'b0;
    write_desc_ready = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    expected_addr = base_addr + (((((layer_id * N_KV_HEADS) + kv_head_id) * MAX_POS) + token_base) << 6);

    @(negedge clk);
    issue_read <= 1'b1;
    @(negedge clk);
    issue_read <= 1'b0;

    if (!read_desc_valid || (read_desc.region != REGION_K_CACHE) || (read_desc.pseudo_channel != PC_ID_W'(23)) ||
        (read_desc.addr != expected_addr) || (read_desc.byte_count != 32'd192)) begin
      $error("kv_cache_manager read descriptor mismatch");
      $finish;
    end

    read_desc_ready <= 1'b1;
    @(negedge clk);
    read_desc_ready <= 1'b0;

    region = REGION_V_CACHE;
    @(negedge clk);
    issue_write <= 1'b1;
    @(negedge clk);
    issue_write <= 1'b0;

    if (!write_desc_valid || (write_desc.region != REGION_V_CACHE) || (write_desc.pseudo_channel != PC_ID_W'(27)) ||
        (write_desc.addr != expected_addr) || (write_desc.byte_count != 32'd192) || !write_desc.write_not_read) begin
      $error("kv_cache_manager write descriptor mismatch");
      $finish;
    end

    $display("PASS: tb_kv_cache_manager");
    $finish;
  end

endmodule
