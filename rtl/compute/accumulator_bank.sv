import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module accumulator_bank (
  input  logic                                             ap_clk,
  input  logic                                             ap_rst_n,
  input  logic                                             clear_i,
  input  logic                                             load_i,
  input  logic signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0]    load_data_i,
  input  tile_tag_t                                        load_tag_i,
  output acc_bus_t                                         acc_o
);

  logic signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] acc_q;
  tile_tag_t                                      tag_q;

  assign acc_o.data = acc_q;
  assign acc_o.tag  = tag_q;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      acc_q <= '0;
      tag_q <= '0;
    end else if (load_i) begin
      acc_q <= load_data_i;
      tag_q <= load_tag_i;
    end else if (clear_i) begin
      acc_q <= '0;
      tag_q <= load_tag_i;
    end
  end

endmodule
