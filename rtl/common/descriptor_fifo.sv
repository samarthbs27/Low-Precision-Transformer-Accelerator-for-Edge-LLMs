module descriptor_fifo #(
  parameter int unsigned DESC_W = 64,
  parameter int unsigned DEPTH  = 8
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic [DESC_W-1:0]            desc_in,
  input  logic                         desc_in_valid,
  output logic                         desc_in_ready,
  output logic [DESC_W-1:0]            desc_out,
  output logic                         desc_out_valid,
  input  logic                         desc_out_ready,
  output logic [$clog2(DEPTH + 1)-1:0] occupancy
);

  stream_fifo #(
    .DATA_W(DESC_W),
    .DEPTH (DEPTH)
  ) u_stream_fifo (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_data   (desc_in),
    .in_valid  (desc_in_valid),
    .in_ready  (desc_in_ready),
    .out_data  (desc_out),
    .out_valid (desc_out_valid),
    .out_ready (desc_out_ready),
    .occupancy (occupancy)
  );

endmodule
