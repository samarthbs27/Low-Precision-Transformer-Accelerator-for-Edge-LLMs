`timescale 1ns/1ps

module tb_descriptor_fifo;

  localparam int unsigned DESC_W = 32;
  localparam int unsigned DEPTH  = 8;

  logic clk;
  logic rst_n;
  logic [DESC_W-1:0] desc_in;
  logic desc_in_valid;
  logic desc_in_ready;
  logic [DESC_W-1:0] desc_out;
  logic desc_out_valid;
  logic desc_out_ready;
  logic [$clog2(DEPTH + 1)-1:0] occupancy;

  descriptor_fifo #(
    .DESC_W(DESC_W),
    .DEPTH (DEPTH)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .desc_in        (desc_in),
    .desc_in_valid  (desc_in_valid),
    .desc_in_ready  (desc_in_ready),
    .desc_out       (desc_out),
    .desc_out_valid (desc_out_valid),
    .desc_out_ready (desc_out_ready),
    .occupancy      (occupancy)
  );

  always #5 clk = ~clk;

  task automatic push_desc(input logic [DESC_W-1:0] value);
    begin
      @(negedge clk);
      desc_in       <= value;
      desc_in_valid <= 1'b1;
      do @(negedge clk); while (!desc_in_ready);
      desc_in_valid <= 1'b0;
    end
  endtask

  task automatic pop_desc_and_check(input logic [DESC_W-1:0] expected);
    begin
      do @(negedge clk); while (!desc_out_valid);
      if (desc_out !== expected) begin
        $error("descriptor_fifo expected 0x%0h, got 0x%0h", expected, desc_out);
        $finish;
      end
      desc_out_ready <= 1'b1;
      @(negedge clk);
      desc_out_ready <= 1'b0;
    end
  endtask

  initial begin
    clk            = 1'b0;
    rst_n          = 1'b0;
    desc_in        = '0;
    desc_in_valid  = 1'b0;
    desc_out_ready = 1'b0;

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    push_desc(32'hDEADBEEF);
    push_desc(32'hCAFEBABE);

    if (occupancy !== 2) begin
      $error("descriptor_fifo occupancy expected 2, got %0d", occupancy);
      $finish;
    end

    pop_desc_and_check(32'hDEADBEEF);
    pop_desc_and_check(32'hCAFEBABE);

    if (occupancy !== 0) begin
      $error("descriptor_fifo occupancy expected 0, got %0d", occupancy);
      $finish;
    end

    $display("PASS: tb_descriptor_fifo");
    $finish;
  end

endmodule
