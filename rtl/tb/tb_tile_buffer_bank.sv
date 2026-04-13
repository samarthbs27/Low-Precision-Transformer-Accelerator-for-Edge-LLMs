`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_tile_buffer_bank;

  logic                 clk;
  logic                 rst_n;
  logic                 wr_valid;
  logic                 wr_ready;
  logic                 wr_ping;
  logic [BANK_ID_W-1:0] wr_bank_id;
  logic [TILE_ID_W-1:0] wr_addr;
  logic [63:0]          wr_data;
  logic                 rd_cmd_valid;
  logic                 rd_cmd_ready;
  logic                 rd_ping;
  logic [BANK_ID_W-1:0] rd_bank_id;
  logic [TILE_ID_W-1:0] rd_addr;
  logic                 rd_data_valid;
  logic                 rd_data_ready;
  logic [63:0]          rd_data;

  tile_buffer_bank #(.DATA_W(64), .BUFFER_DEPTH(16)) dut (
    .ap_clk(clk), .ap_rst_n(rst_n),
    .wr_valid_i(wr_valid), .wr_ready_o(wr_ready), .wr_ping_i(wr_ping), .wr_bank_id_i(wr_bank_id), .wr_addr_i(wr_addr), .wr_data_i(wr_data),
    .rd_cmd_valid_i(rd_cmd_valid), .rd_cmd_ready_o(rd_cmd_ready), .rd_ping_i(rd_ping), .rd_bank_id_i(rd_bank_id), .rd_addr_i(rd_addr),
    .rd_data_valid_o(rd_data_valid), .rd_data_ready_i(rd_data_ready), .rd_data_o(rd_data)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 1'b0; rst_n = 1'b0;
    wr_valid = 1'b0; wr_ping = 1'b0; wr_bank_id = '0; wr_addr = '0; wr_data = '0;
    rd_cmd_valid = 1'b0; rd_ping = 1'b0; rd_bank_id = '0; rd_addr = '0; rd_data_ready = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    wr_valid <= 1'b1; wr_ping <= 1'b1; wr_bank_id <= 2; wr_addr <= 3; wr_data <= 64'h1122_3344_5566_7788;
    @(negedge clk);
    wr_valid <= 1'b0;

    @(negedge clk);
    wr_valid <= 1'b1; wr_ping <= 1'b0; wr_bank_id <= 2; wr_addr <= 3; wr_data <= 64'h99AA_BBCC_DDEE_FF00;
    @(negedge clk);
    wr_valid <= 1'b0;

    @(negedge clk);
    rd_cmd_valid <= 1'b1; rd_ping <= 1'b1; rd_bank_id <= 2; rd_addr <= 3;
    @(negedge clk);
    rd_cmd_valid <= 1'b0;
    if (!rd_data_valid || (rd_data != 64'h1122_3344_5566_7788)) begin
      $error("ping-bank readback mismatch, got 0x%016h", rd_data);
      $finish;
    end
    rd_data_ready <= 1'b1;
    @(negedge clk);
    rd_data_ready <= 1'b0;

    @(negedge clk);
    rd_cmd_valid <= 1'b1; rd_ping <= 1'b0; rd_bank_id <= 2; rd_addr <= 3;
    @(negedge clk);
    rd_cmd_valid <= 1'b0;
    if (!rd_data_valid || (rd_data != 64'h99AA_BBCC_DDEE_FF00)) begin
      $error("pong-bank readback mismatch, got 0x%016h", rd_data);
      $finish;
    end
    rd_data_ready <= 1'b1;
    @(negedge clk);
    rd_data_ready <= 1'b0;

    $display("PASS: tb_tile_buffer_bank");
    $finish;
  end

endmodule
