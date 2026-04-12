module stream_fifo #(
  parameter int unsigned DATA_W = 8,
  parameter int unsigned DEPTH  = 4
) (
  input  logic                             clk,
  input  logic                             rst_n,
  input  logic [DATA_W-1:0]                in_data,
  input  logic                             in_valid,
  output logic                             in_ready,
  output logic [DATA_W-1:0]                out_data,
  output logic                             out_valid,
  input  logic                             out_ready,
  output logic [$clog2(DEPTH + 1)-1:0]     occupancy
);

  localparam int unsigned PTR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;

  logic [DATA_W-1:0] mem [0:DEPTH-1];
  logic [PTR_W-1:0]  wr_ptr;
  logic [PTR_W-1:0]  rd_ptr;
  logic [PTR_W-1:0]  wr_ptr_next;
  logic [PTR_W-1:0]  rd_ptr_next;
  logic              push;
  logic              pop;

  initial begin
    if (DEPTH == 0) begin
      $error("stream_fifo DEPTH must be >= 1");
      $finish;
    end
  end

  assign out_valid  = (occupancy != '0);
  assign in_ready   = (occupancy != DEPTH) || (out_valid && out_ready);
  assign out_data   = mem[rd_ptr];
  assign push       = in_valid && in_ready;
  assign pop        = out_valid && out_ready;
  assign wr_ptr_next = (wr_ptr == DEPTH - 1) ? '0 : (wr_ptr + 1'b1);
  assign rd_ptr_next = (rd_ptr == DEPTH - 1) ? '0 : (rd_ptr + 1'b1);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_ptr    <= '0;
      rd_ptr    <= '0;
      occupancy <= '0;
    end else begin
      if (push) begin
        mem[wr_ptr] <= in_data;
        wr_ptr      <= wr_ptr_next;
      end

      if (pop) begin
        rd_ptr <= rd_ptr_next;
      end

      unique case ({push, pop})
        2'b10: occupancy <= occupancy + 1'b1;
        2'b01: occupancy <= occupancy - 1'b1;
        default: occupancy <= occupancy;
      endcase
    end
  end

endmodule
