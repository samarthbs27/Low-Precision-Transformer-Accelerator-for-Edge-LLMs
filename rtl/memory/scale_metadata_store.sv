import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module scale_metadata_store (
  input  logic                                       ap_clk,
  input  logic                                       ap_rst_n,
  input  logic                                       wr_valid_i,
  output logic                                       wr_ready_o,
  input  tensor_id_e                                 wr_tensor_id_i,
  input  scale_bus_t                                 wr_scale_i,
  input  logic                                       rd0_en_i,
  input  tensor_id_e                                 rd0_tensor_id_i,
  input  logic [LAYER_ID_W-1:0]                      rd0_layer_id_i,
  input  logic [KV_HEAD_ID_W-1:0]                    rd0_kv_head_id_i,
  output logic                                       rd0_valid_o,
  output logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] rd0_data_o,
  input  logic                                       rd1_en_i,
  input  tensor_id_e                                 rd1_tensor_id_i,
  input  logic [LAYER_ID_W-1:0]                      rd1_layer_id_i,
  input  logic [KV_HEAD_ID_W-1:0]                    rd1_kv_head_id_i,
  output logic                                       rd1_valid_o,
  output logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] rd1_data_o,
  input  logic                                       rd2_en_i,
  input  tensor_id_e                                 rd2_tensor_id_i,
  input  logic [LAYER_ID_W-1:0]                      rd2_layer_id_i,
  input  logic [KV_HEAD_ID_W-1:0]                    rd2_kv_head_id_i,
  output logic                                       rd2_valid_o,
  output logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] rd2_data_o,
  input  logic                                       rd3_en_i,
  input  tensor_id_e                                 rd3_tensor_id_i,
  input  logic [LAYER_ID_W-1:0]                      rd3_layer_id_i,
  input  logic [KV_HEAD_ID_W-1:0]                    rd3_kv_head_id_i,
  output logic                                       rd3_valid_o,
  output logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] rd3_data_o
);

  localparam int unsigned SCALE_TENSOR_SLOTS = 16;
  localparam int unsigned SCALE_STORE_DEPTH = SCALE_TENSOR_SLOTS * N_LAYERS * N_KV_HEADS;

  logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] scale_mem [0:SCALE_STORE_DEPTH-1];

  function automatic int unsigned scale_index(
    input tensor_id_e              tensor_id,
    input logic [LAYER_ID_W-1:0]   layer_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id
  );
    int unsigned tensor_slot;
    begin
      tensor_slot = int'(tensor_id);
      scale_index = ((tensor_slot * N_LAYERS) + layer_id) * N_KV_HEADS + kv_head_id;
    end
  endfunction

  assign wr_ready_o  = 1'b1;
  assign rd0_valid_o = rd0_en_i;
  assign rd1_valid_o = rd1_en_i;
  assign rd2_valid_o = rd2_en_i;
  assign rd3_valid_o = rd3_en_i;

  assign rd0_data_o = scale_mem[scale_index(rd0_tensor_id_i, rd0_layer_id_i, rd0_kv_head_id_i)];
  assign rd1_data_o = scale_mem[scale_index(rd1_tensor_id_i, rd1_layer_id_i, rd1_kv_head_id_i)];
  assign rd2_data_o = scale_mem[scale_index(rd2_tensor_id_i, rd2_layer_id_i, rd2_kv_head_id_i)];
  assign rd3_data_o = scale_mem[scale_index(rd3_tensor_id_i, rd3_layer_id_i, rd3_kv_head_id_i)];

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      for (int idx = 0; idx < SCALE_STORE_DEPTH; idx++) begin
        scale_mem[idx] <= '0;
      end
    end else if (wr_valid_i) begin
      scale_mem[scale_index(wr_tensor_id_i, wr_scale_i.tag.layer_id, wr_scale_i.tag.kv_head_id)] <= wr_scale_i.data;
    end
  end

endmodule
