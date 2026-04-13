import tinyllama_pkg::*;

module tile_buffer_bank #(
  parameter int unsigned DATA_W = DMA_BEAT_W,
  parameter int unsigned BUFFER_DEPTH = 64
) (
  input  logic                         ap_clk,
  input  logic                         ap_rst_n,
  input  logic                         wr_valid_i,
  output logic                         wr_ready_o,
  input  logic                         wr_ping_i,
  input  logic [BANK_ID_W-1:0]         wr_bank_id_i,
  input  logic [TILE_ID_W-1:0]         wr_addr_i,
  input  logic [DATA_W-1:0]            wr_data_i,
  input  logic                         rd_cmd_valid_i,
  output logic                         rd_cmd_ready_o,
  input  logic                         rd_ping_i,
  input  logic [BANK_ID_W-1:0]         rd_bank_id_i,
  input  logic [TILE_ID_W-1:0]         rd_addr_i,
  output logic                         rd_data_valid_o,
  input  logic                         rd_data_ready_i,
  output logic [DATA_W-1:0]            rd_data_o
);

  localparam int unsigned BUF_ADDR_W = (BUFFER_DEPTH > 1) ? $clog2(BUFFER_DEPTH) : 1;

  logic [DATA_W-1:0] mem_ping [0:TILE_BUFFER_BANKS-1][0:BUFFER_DEPTH-1];
  logic [DATA_W-1:0] mem_pong [0:TILE_BUFFER_BANKS-1][0:BUFFER_DEPTH-1];
  logic [DATA_W-1:0] rd_data_q;
  logic              rd_valid_q;

  assign wr_ready_o      = 1'b1;
  assign rd_cmd_ready_o  = !rd_valid_q || rd_data_ready_i;
  assign rd_data_valid_o = rd_valid_q;
  assign rd_data_o       = rd_data_q;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      rd_data_q  <= '0;
      rd_valid_q <= 1'b0;
    end else begin
      if (wr_valid_i) begin
        if (wr_ping_i) begin
          mem_ping[wr_bank_id_i][wr_addr_i[BUF_ADDR_W-1:0]] <= wr_data_i;
        end else begin
          mem_pong[wr_bank_id_i][wr_addr_i[BUF_ADDR_W-1:0]] <= wr_data_i;
        end
      end

      if (rd_cmd_valid_i && rd_cmd_ready_o) begin
        if (rd_ping_i) begin
          rd_data_q <= mem_ping[rd_bank_id_i][rd_addr_i[BUF_ADDR_W-1:0]];
        end else begin
          rd_data_q <= mem_pong[rd_bank_id_i][rd_addr_i[BUF_ADDR_W-1:0]];
        end
        rd_valid_q <= 1'b1;
      end else if (rd_valid_q && rd_data_ready_i) begin
        rd_valid_q <= 1'b0;
      end
    end
  end

endmodule
