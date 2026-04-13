import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module kv_cache_manager (
  input  logic                    ap_clk,
  input  logic                    ap_rst_n,
  input  logic                    issue_read_i,
  input  logic                    issue_write_i,
  input  hbm_region_e             region_i,
  input  logic [HBM_ADDR_W-1:0]   base_addr_i,
  input  logic [LAYER_ID_W-1:0]   layer_id_i,
  input  logic [KV_HEAD_ID_W-1:0] kv_head_id_i,
  input  logic [POS_W-1:0]        token_base_i,
  input  logic [COUNT_W-1:0]      seq_count_i,
  output logic                    read_desc_valid_o,
  input  logic                    read_desc_ready_i,
  output dma_desc_t               read_desc_o,
  output logic                    write_desc_valid_o,
  input  logic                    write_desc_ready_i,
  output dma_desc_t               write_desc_o
);

  logic                  read_pending_q;
  logic                  write_pending_q;
  logic [HBM_ADDR_W-1:0] row_index;
  logic [HBM_ADDR_W-1:0] byte_addr;
  logic [31:0]           byte_count;
  logic [15:0]           burst_len;

  function automatic logic [PC_ID_W-1:0] cache_pc_id(
    input hbm_region_e             region,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id
  );
    begin
      if (region == REGION_V_CACHE) begin
        cache_pc_id = PC_ID_W'(26) + kv_head_id;
      end else begin
        cache_pc_id = PC_ID_W'(22) + kv_head_id;
      end
    end
  endfunction

  assign row_index  = (((layer_id_i * N_KV_HEADS) + kv_head_id_i) * MAX_POS) + token_base_i;
  assign byte_addr  = base_addr_i + (row_index << 6);
  assign byte_count = seq_count_i << 6;
  assign burst_len  = (byte_count == '0) ? 16'd1 :
                      ((byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES);

  assign read_desc_valid_o  = read_pending_q;
  assign write_desc_valid_o = write_pending_q;

  always @* begin
    read_desc_o = '0;
    read_desc_o.region         = region_i;
    read_desc_o.tensor_id      = TENSOR_NONE;
    read_desc_o.write_not_read = 1'b0;
    read_desc_o.pseudo_channel = cache_pc_id(region_i, kv_head_id_i);
    read_desc_o.addr           = byte_addr;
    read_desc_o.burst_len      = burst_len;
    read_desc_o.byte_count     = byte_count;
    read_desc_o.layer_id       = layer_id_i;
    read_desc_o.kv_head_id     = kv_head_id_i;
    read_desc_o.tile_id        = token_base_i[TILE_ID_W-1:0];

    write_desc_o               = read_desc_o;
    write_desc_o.write_not_read = 1'b1;
  end

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      read_pending_q  <= 1'b0;
      write_pending_q <= 1'b0;
    end else begin
      if (issue_read_i) begin
        read_pending_q <= 1'b1;
      end

      if (issue_write_i) begin
        write_pending_q <= 1'b1;
      end

      if (read_pending_q && read_desc_ready_i) begin
        read_pending_q <= 1'b0;
      end

      if (write_pending_q && write_desc_ready_i) begin
        write_pending_q <= 1'b0;
      end
    end
  end

endmodule
