import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module debug_dma_writer (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  start_i,
  input  logic                  debug_enable_i,
  input  logic [HBM_ADDR_W-1:0] debug_base_addr_i,
  input  logic                  dbg_valid_i,
  output logic                  dbg_ready_o,
  input  dbg_bus_t              dbg_i,
  output logic                  busy_o,
  output logic                  wr_desc_valid_o,
  input  logic                  wr_desc_ready_i,
  output dma_desc_t             wr_desc_o,
  output logic                  wr_data_valid_o,
  input  logic                  wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0] wr_data_o
);

  logic [COUNT_W-1:0]     write_index_q;
  logic                   pending_q;
  logic [HBM_ADDR_W-1:0]  base_addr_q;
  dbg_bus_t               pending_dbg_q;

  assign busy_o          = pending_q;
  assign dbg_ready_o     = debug_enable_i && !pending_q;
  assign wr_desc_valid_o = pending_q;
  assign wr_data_valid_o = pending_q;
  assign wr_data_o       = pending_dbg_q.data;

  assign wr_desc_o.region         = REGION_DEBUG;
  assign wr_desc_o.tensor_id      = TENSOR_NONE;
  assign wr_desc_o.write_not_read = 1'b1;
  assign wr_desc_o.pseudo_channel = PC_ID_W'(31);
  assign wr_desc_o.addr           = base_addr_q + (write_index_q * DMA_BEAT_BYTES);
  assign wr_desc_o.burst_len      = 16'd1;
  assign wr_desc_o.byte_count     = DMA_BEAT_BYTES;
  assign wr_desc_o.layer_id       = pending_dbg_q.tag.layer_id;
  assign wr_desc_o.kv_head_id     = pending_dbg_q.tag.kv_head_id;
  assign wr_desc_o.tile_id        = pending_dbg_q.tag.tile_id;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      write_index_q <= '0;
      pending_q     <= 1'b0;
      base_addr_q   <= '0;
      pending_dbg_q <= '0;
    end else begin
      if (start_i) begin
        write_index_q <= '0;
        base_addr_q   <= debug_base_addr_i;
        pending_q     <= 1'b0;
      end

      if (debug_enable_i && dbg_valid_i && dbg_ready_o) begin
        pending_q     <= 1'b1;
        pending_dbg_q <= dbg_i;
      end

      if (pending_q && wr_desc_ready_i && wr_data_ready_i) begin
        pending_q     <= 1'b0;
        write_index_q <= write_index_q + 1'b1;
      end
    end
  end

endmodule
