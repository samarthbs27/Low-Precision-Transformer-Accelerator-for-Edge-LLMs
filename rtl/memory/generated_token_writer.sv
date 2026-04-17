import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module generated_token_writer (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  start_i,
  input  logic [HBM_ADDR_W-1:0] generated_tokens_base_addr_i,
  input  logic [COUNT_W-1:0]    generated_tokens_capacity_i,
  input  logic                  token_valid_i,
  input  logic [TOKEN_W-1:0]    token_id_i,
  output logic                  token_ready_o,
  output logic                  busy_o,
  output logic                  wr_desc_valid_o,
  input  logic                  wr_desc_ready_i,
  output dma_desc_t             wr_desc_o,
  output logic                  wr_data_valid_o,
  input  logic                  wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0] wr_data_o
);

  logic                   armed_q;
  logic                   pending_q;
  logic [HBM_ADDR_W-1:0]  base_addr_q;
  logic [COUNT_W-1:0]     capacity_q;
  logic [COUNT_W-1:0]     write_index_q;
  logic [TOKEN_W-1:0]     pending_token_q;
  logic [HBM_ADDR_W-1:0]  pending_addr_q;

  assign busy_o          = armed_q || pending_q;
  assign token_ready_o   = armed_q && !pending_q;
  assign wr_desc_valid_o = pending_q;
  assign wr_data_valid_o = pending_q;

  assign wr_desc_o.region         = REGION_HOST_IO;
  assign wr_desc_o.tensor_id      = TENSOR_NONE;
  assign wr_desc_o.write_not_read = 1'b1;
  assign wr_desc_o.pseudo_channel = HOST_IO_PC_ID;
  assign wr_desc_o.addr           = pending_addr_q;
  assign wr_desc_o.burst_len      = 16'd1;
  assign wr_desc_o.byte_count     = DMA_BEAT_BYTES;
  assign wr_desc_o.layer_id       = '0;
  assign wr_desc_o.kv_head_id     = '0;
  assign wr_desc_o.tile_id        = TILE_ID_W'(write_index_q);

  always_comb begin
    wr_data_o              = '0;
    wr_data_o[TOKEN_W-1:0] = pending_token_q;
  end

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      armed_q         <= 1'b0;
      pending_q       <= 1'b0;
      base_addr_q     <= '0;
      capacity_q      <= '0;
      write_index_q   <= '0;
      pending_token_q <= '0;
      pending_addr_q  <= '0;
    end else begin
      if (start_i) begin
        armed_q       <= 1'b1;
        pending_q     <= 1'b0;
        base_addr_q   <= generated_tokens_base_addr_i;
        capacity_q    <= generated_tokens_capacity_i;
        write_index_q <= '0;
      end

      if (token_valid_i && token_ready_o) begin
        pending_q       <= 1'b1;
        pending_token_q <= token_id_i;
        pending_addr_q  <= base_addr_q + (write_index_q * DMA_BEAT_BYTES);
      end

      if (pending_q && wr_desc_ready_i && wr_data_ready_i) begin
        pending_q <= 1'b0;

        if ((capacity_q != '0) && ((write_index_q + 1'b1) >= capacity_q)) begin
          write_index_q <= '0;
        end else begin
          write_index_q <= write_index_q + 1'b1;
        end
      end
    end
  end

endmodule
