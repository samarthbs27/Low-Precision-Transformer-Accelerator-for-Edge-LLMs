import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module kv_cache_dma_writer (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  req_valid_i,
  output logic                  req_ready_o,
  input  dma_desc_t             req_desc_i,
  input  logic                  tile_valid_i,
  output logic                  tile_ready_o,
  input  act_bus_t              tile_i,
  output logic                  busy_o,
  output logic                  wr_desc_valid_o,
  input  logic                  wr_desc_ready_i,
  output dma_desc_t             wr_desc_o,
  output logic                  wr_data_valid_o,
  input  logic                  wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0] wr_data_o
);

  logic                   pending_q;
  dma_desc_t              pending_desc_q;
  logic [DMA_BEAT_W-1:0]  pending_data_q;

  assign busy_o          = pending_q;
  assign req_ready_o     = !pending_q;
  assign tile_ready_o    = !pending_q;
  assign wr_desc_valid_o = pending_q;
  assign wr_data_valid_o = pending_q;
  assign wr_desc_o       = pending_desc_q;
  assign wr_data_o       = pending_data_q;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      pending_q      <= 1'b0;
      pending_desc_q <= '0;
      pending_data_q <= '0;
    end else begin
      if (req_valid_i && tile_valid_i && !pending_q) begin
        pending_q      <= 1'b1;
        pending_desc_q <= req_desc_i;
        pending_data_q <= tile_i.data[DMA_BEAT_BYTES-1:0];
      end

      if (pending_q && wr_desc_ready_i && wr_data_ready_i) begin
        pending_q <= 1'b0;
      end
    end
  end

endmodule
