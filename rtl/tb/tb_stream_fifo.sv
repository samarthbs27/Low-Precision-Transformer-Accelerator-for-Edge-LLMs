`timescale 1ns/1ps

module tb_stream_fifo;

  localparam int unsigned DATA_W = 8;
  localparam int unsigned DEPTH  = 4;

  logic clk;
  logic rst_n;
  logic [DATA_W-1:0] in_data;
  logic in_valid;
  logic in_ready;
  logic [DATA_W-1:0] out_data;
  logic out_valid;
  logic out_ready;
  logic [$clog2(DEPTH + 1)-1:0] occupancy;

  stream_fifo #(
    .DATA_W(DATA_W),
    .DEPTH (DEPTH)
  ) dut (
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

  always #5 clk = ~clk;

  task automatic push_byte(input logic [DATA_W-1:0] value);
    int unsigned wait_cycles;
    begin
      @(negedge clk);
      in_data  <= value;
      in_valid <= 1'b1;
      wait_cycles = 0;
      while (!in_ready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 32) begin
          $error("stream_fifo push timeout waiting for in_ready while pushing 0x%0h", value);
          $finish;
        end
      end
      @(negedge clk);
      in_valid <= 1'b0;
    end
  endtask

  task automatic pop_and_check(input logic [DATA_W-1:0] expected);
    int unsigned wait_cycles;
    begin
      wait_cycles = 0;
      while (!out_valid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 32) begin
          $error("stream_fifo pop timeout waiting for out_valid");
          $finish;
        end
      end
      if (out_data !== expected) begin
        $error("stream_fifo expected 0x%0h, got 0x%0h", expected, out_data);
        $finish;
      end
      out_ready <= 1'b1;
      @(negedge clk);
      out_ready <= 1'b0;
    end
  endtask

  initial begin
    clk       = 1'b0;
    rst_n     = 1'b0;
    in_data   = '0;
    in_valid  = 1'b0;
    out_ready = 1'b0;

    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    push_byte(8'h11);
    push_byte(8'h22);
    push_byte(8'h33);

    if (occupancy !== 3) begin
      $error("stream_fifo occupancy expected 3, got %0d", occupancy);
      $finish;
    end

    pop_and_check(8'h11);
    pop_and_check(8'h22);
    pop_and_check(8'h33);

    if (occupancy !== 0) begin
      $error("stream_fifo occupancy expected 0, got %0d", occupancy);
      $finish;
    end

    push_byte(8'hA1);

    @(negedge clk);
    in_data   = 8'hB2;
    in_valid  = 1'b1;
    out_ready = 1'b1;
    @(negedge clk);
    in_valid  = 1'b0;
    out_ready = 1'b0;

    if (occupancy !== 1) begin
      $error("stream_fifo occupancy expected 1 after simultaneous push/pop, got %0d", occupancy);
      $finish;
    end

    if (!out_valid || (out_data !== 8'hB2)) begin
      $error("stream_fifo expected 0xB2 at head after simultaneous push/pop, got valid=%0b data=0x%0h", out_valid, out_data);
      $finish;
    end

    pop_and_check(8'hB2);

    push_byte(8'h10);
    push_byte(8'h20);
    push_byte(8'h30);
    push_byte(8'h40);

    @(negedge clk);
    if (occupancy !== DEPTH) begin
      $error("stream_fifo occupancy expected %0d when full, got %0d", DEPTH, occupancy);
      $finish;
    end

    if (in_ready !== 1'b0) begin
      $error("stream_fifo in_ready expected LOW when full without pop, got %0b", in_ready);
      $finish;
    end

    @(negedge clk);
    in_data   = 8'h50;
    in_valid  = 1'b1;
    out_ready = 1'b1;
    #1;
    if (in_ready !== 1'b1) begin
      $error("stream_fifo in_ready expected HIGH when full and pop is active, got %0b", in_ready);
      $finish;
    end
    @(negedge clk);
    in_valid  = 1'b0;
    out_ready = 1'b0;

    if (occupancy !== DEPTH) begin
      $error("stream_fifo occupancy expected %0d after full simultaneous push/pop, got %0d", DEPTH, occupancy);
      $finish;
    end

    pop_and_check(8'h20);
    pop_and_check(8'h30);
    pop_and_check(8'h40);
    pop_and_check(8'h50);

    push_byte(8'h61);
    push_byte(8'h62);
    push_byte(8'h63);
    push_byte(8'h64);

    @(negedge clk);
    if (occupancy !== DEPTH) begin
      $error("stream_fifo occupancy expected %0d before blocked push, got %0d", DEPTH, occupancy);
      $finish;
    end

    @(negedge clk);
    in_data   = 8'h99;
    in_valid  = 1'b1;
    out_ready = 1'b0;
    #1;
    if (in_ready !== 1'b0) begin
      $error("stream_fifo in_ready expected LOW during blocked push, got %0b", in_ready);
      $finish;
    end
    @(negedge clk);
    in_valid = 1'b0;

    if (occupancy !== DEPTH) begin
      $error("stream_fifo occupancy expected %0d after blocked push, got %0d", DEPTH, occupancy);
      $finish;
    end

    pop_and_check(8'h61);
    pop_and_check(8'h62);
    pop_and_check(8'h63);
    pop_and_check(8'h64);

    $display("PASS: tb_stream_fifo");
    $finish;
  end

endmodule
