`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_requantize_unit;

  acc_bus_t   acc_bus;
  scale_bus_t scale_bus;
  logic       nonnegative_only;
  act_bus_t   act_bus;

  requantize_unit dut (
    .acc_i(acc_bus),
    .scale_i(scale_bus),
    .nonnegative_only_i(nonnegative_only),
    .act_o(act_bus)
  );

  initial begin
    acc_bus           = '0;
    scale_bus         = '0;
    nonnegative_only  = 1'b0;

    acc_bus.tag.elem_count = 16'd5;
    scale_bus.data[0] = 32'h0001_0000; // 1.0 in Q16.16
    acc_bus.data[0]   = 32'sd10;
    acc_bus.data[1]   = 32'sd300;
    acc_bus.data[2]   = -32'sd300;
    #1;

    if ((act_bus.data[0] != 8'sd10) ||
        (act_bus.data[1] != 8'sd127) ||
        (act_bus.data[2] != -8'sd127) ||
        (act_bus.data[5] != '0)) begin
      $error("requantize_unit identity/clamp mismatch");
      $finish;
    end

    scale_bus.data[0] = 32'h0000_8000; // 0.5 in Q16.16
    acc_bus.data[3]   = 32'sd3;
    acc_bus.data[4]   = 32'sd5;
    #1;

    if ((act_bus.data[3] != 8'sd2) || (act_bus.data[4] != 8'sd2)) begin
      $error("requantize_unit round-to-nearest-even mismatch");
      $finish;
    end

    nonnegative_only = 1'b1;
    acc_bus.data[2]  = -32'sd4;
    #1;

    if (act_bus.data[2] != '0) begin
      $error("requantize_unit expected nonnegative clamp");
      $finish;
    end

    $display("PASS: tb_requantize_unit");
    $finish;
  end

endmodule
