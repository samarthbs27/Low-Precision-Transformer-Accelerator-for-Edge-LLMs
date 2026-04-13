`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_scale_metadata_store;

  logic                                       clk;
  logic                                       rst_n;
  logic                                       wr_valid;
  logic                                       wr_ready;
  tensor_id_e                                 wr_tensor_id;
  scale_bus_t                                 wr_scale;
  logic                                       rd0_en;
  tensor_id_e                                 rd0_tensor_id;
  logic [LAYER_ID_W-1:0]                      rd0_layer_id;
  logic [KV_HEAD_ID_W-1:0]                    rd0_kv_head_id;
  logic                                       rd0_valid;
  logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] rd0_data;

  scale_metadata_store dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .wr_valid_i(wr_valid),
    .wr_ready_o(wr_ready),
    .wr_tensor_id_i(wr_tensor_id),
    .wr_scale_i(wr_scale),
    .rd0_en_i(rd0_en),
    .rd0_tensor_id_i(rd0_tensor_id),
    .rd0_layer_id_i(rd0_layer_id),
    .rd0_kv_head_id_i(rd0_kv_head_id),
    .rd0_valid_o(rd0_valid),
    .rd0_data_o(rd0_data),
    .rd1_en_i(1'b0),
    .rd1_tensor_id_i(tensor_id_e'('0)),
    .rd1_layer_id_i({LAYER_ID_W{1'b0}}),
    .rd1_kv_head_id_i({KV_HEAD_ID_W{1'b0}}),
    .rd1_valid_o(),
    .rd1_data_o(),
    .rd2_en_i(1'b0),
    .rd2_tensor_id_i(tensor_id_e'('0)),
    .rd2_layer_id_i({LAYER_ID_W{1'b0}}),
    .rd2_kv_head_id_i({KV_HEAD_ID_W{1'b0}}),
    .rd2_valid_o(),
    .rd2_data_o(),
    .rd3_en_i(1'b0),
    .rd3_tensor_id_i(tensor_id_e'('0)),
    .rd3_layer_id_i({LAYER_ID_W{1'b0}}),
    .rd3_kv_head_id_i({KV_HEAD_ID_W{1'b0}}),
    .rd3_valid_o(),
    .rd3_data_o()
  );

  always #5 clk = ~clk;

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    wr_valid = 1'b0;
    wr_tensor_id = TENSOR_SCALE_META;
    wr_scale = '0;
    rd0_en = 1'b0;
    rd0_tensor_id = TENSOR_SCALE_META;
    rd0_layer_id = '0;
    rd0_kv_head_id = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    wr_scale.tag.layer_id = 3;
    wr_scale.tag.kv_head_id = 1;
    wr_scale.data[0] = 32'h10;
    wr_scale.data[1] = 32'h20;
    wr_scale.data[2] = 32'h30;

    @(negedge clk);
    wr_valid <= 1'b1;
    @(negedge clk);
    wr_valid <= 1'b0;

    rd0_tensor_id  <= TENSOR_SCALE_META;
    rd0_layer_id   <= 3;
    rd0_kv_head_id <= 1;
    rd0_en         <= 1'b1;
    @(negedge clk);

    if (!rd0_valid || (rd0_data[0] != 32'h10) || (rd0_data[1] != 32'h20) || (rd0_data[2] != 32'h30)) begin
      $error("scale_metadata_store readback mismatch");
      $finish;
    end

    $display("PASS: tb_scale_metadata_store");
    $finish;
  end

endmodule
