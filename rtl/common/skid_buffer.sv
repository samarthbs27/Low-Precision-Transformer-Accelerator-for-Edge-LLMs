module skid_buffer #(
  parameter int unsigned DATA_W = 8
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic [DATA_W-1:0]            in_data,
  input  logic                         in_valid,
  output logic                         in_ready,
  output logic [DATA_W-1:0]            out_data,
  output logic                         out_valid,
  input  logic                         out_ready,
  output logic [$clog2(2 + 1)-1:0]     occupancy
);

  stream_fifo #(
    .DATA_W(DATA_W),
    .DEPTH(2)
  ) u_stream_fifo (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_data   (in_data),
    .in_valid  (in_valid),
    .in_ready  (in_ready),
    .out_data  (out_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .occupancy (occupancy)
  );

endmodule
