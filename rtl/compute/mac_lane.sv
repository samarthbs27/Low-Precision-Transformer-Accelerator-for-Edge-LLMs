import tinyllama_pkg::*;

module mac_lane (
  input  logic signed [ACT_W-1:0]    act_i,
  input  logic signed [WEIGHT_W-1:0] wt_i,
  input  logic signed [ACC_W-1:0]    acc_i,
  input  logic                       mac_valid_i,
  output logic signed [ACC_W-1:0]    acc_o
);

  logic signed [ACC_W-1:0] product_q;

  always_comb begin
    product_q = $signed(act_i) * $signed(wt_i);
    acc_o     = acc_i;

    if (mac_valid_i) begin
      acc_o = acc_i + product_q;
    end
  end

endmodule
