`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_mac_lane;

  logic signed [ACT_W-1:0]    act;
  logic signed [WEIGHT_W-1:0] wt;
  logic signed [ACC_W-1:0]    acc_in;
  logic                       mac_valid;
  logic signed [ACC_W-1:0]    acc_out;

  mac_lane dut (
    .act_i(act),
    .wt_i(wt),
    .acc_i(acc_in),
    .mac_valid_i(mac_valid),
    .acc_o(acc_out)
  );

  initial begin
    act       = 8'sd3;
    wt        = -8'sd4;
    acc_in    = 32'sd10;
    mac_valid = 1'b1;
    #1;
    if (acc_out != -32'sd2) begin
      $error("mac_lane signed multiply/add mismatch");
      $finish;
    end

    mac_valid = 1'b0;
    #1;
    if (acc_out != acc_in) begin
      $error("mac_lane expected accumulator hold when invalid");
      $finish;
    end

    act       = -8'sd7;
    wt        = -8'sd6;
    acc_in    = -32'sd5;
    mac_valid = 1'b1;
    #1;
    if (acc_out != 32'sd37) begin
      $error("mac_lane negative-times-negative mismatch");
      $finish;
    end

    $display("PASS: tb_mac_lane");
    $finish;
  end

endmodule
