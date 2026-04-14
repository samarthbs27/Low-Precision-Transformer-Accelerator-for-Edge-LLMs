`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_silu_wrapper;

  localparam int unsigned CHUNK_ELEMS = N_TILE;
  localparam int unsigned CHUNK_W = CHUNK_ELEMS * SCALE_W;

  localparam string PREFILL_BASE = "sim/golden_traces/phase5/rtl/phase5_prefill_layer0_silu_gate_m0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase5/rtl/phase5_decode_layer0_silu_gate_m0";

  logic clk;
  logic rst_n;
  logic gate_valid;
  logic gate_ready;
  act_bus_t gate_bus;
  logic [SCALE_W-1:0] input_scale;
  logic [SCALE_W-1:0] output_scale;
  logic scale_valid;
  logic scale_ready;
  scale_bus_t scale_bus;
  logic silu_valid;
  logic silu_ready;
  act_bus_t silu_bus;
  logic busy;
  logic done_pulse;

  logic [31:0] meta_mem [0:1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] x_in_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] y_out_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] input_scale_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] output_scale_mem [0:0];
  logic [CHUNK_W-1:0] core_x_chunks_mem [0:15];
  logic [CHUNK_W-1:0] core_y_chunks_mem [0:15];

  logic saw_scale;
  logic saw_silu;
  scale_bus_t captured_scale_bus;
  act_bus_t captured_silu_bus;
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] captured_scale_data;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_silu_data;

  silu_wrapper dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .gate_valid_i(gate_valid),
    .gate_ready_o(gate_ready),
    .gate_i(gate_bus),
    .input_scale_i(input_scale),
    .output_scale_i(output_scale),
    .scale_valid_o(scale_valid),
    .scale_ready_i(scale_ready),
    .scale_o(scale_bus),
    .silu_valid_o(silu_valid),
    .silu_ready_i(silu_ready),
    .silu_o(silu_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_scale <= 1'b0;
      saw_silu <= 1'b0;
      captured_scale_bus <= '0;
      captured_silu_bus <= '0;
    end else begin
      if (scale_valid && scale_ready) begin
        saw_scale <= 1'b1;
        captured_scale_bus <= scale_bus;
        captured_scale_data <= scale_bus.data;
      end
      if (silu_valid && silu_ready) begin
        saw_silu <= 1'b1;
        captured_silu_bus <= silu_bus;
        captured_silu_data <= silu_bus.data;
      end
    end
  end

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".x_in_packed.memh"}, x_in_mem);
      $readmemh({base, ".y_out_expected_packed.memh"}, y_out_mem);
      $readmemh({base, ".input_scale_packed.memh"}, input_scale_mem);
      $readmemh({base, ".output_scale_packed.memh"}, output_scale_mem);
      $readmemh({base, ".core_x_chunks_packed.memh"}, core_x_chunks_mem);
      $readmemh({base, ".core_y_chunks_packed.memh"}, core_y_chunks_mem);
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int cycle_count;
    begin
      load_case(base);

      gate_bus = '0;
      gate_bus.tag.block_id = BLOCK_GATE;
      gate_bus.tag.gemm_mode = GEMM_GATE;
      gate_bus.tag.elem_count = ELEM_COUNT_W'(meta_mem[1]);
      gate_bus.tag.is_partial = (meta_mem[1] != ACT_VECTOR_ELEMS);
      gate_bus.tag.is_last = 1'b1;
      gate_bus.data = x_in_mem[0];

      input_scale = input_scale_mem[0][SCALE_W-1:0];
      output_scale = output_scale_mem[0][SCALE_W-1:0];
      gate_valid = 1'b1;
      scale_ready = 1'b1;
      silu_ready = 1'b1;
      saw_scale = 1'b0;
      saw_silu = 1'b0;

      cycle_count = 0;
      while (!done_pulse && (cycle_count < 200)) begin
        @(posedge clk);
        if (gate_valid && gate_ready) begin
          gate_valid <= 1'b0;
        end
        cycle_count = cycle_count + 1;
      end

      if (!done_pulse) begin
        $error("silu_wrapper timeout for %s", case_name);
        $finish;
      end

      @(posedge clk);

      if (!saw_scale || !saw_silu) begin
        $error("silu_wrapper missing output handshake for %s", case_name);
        $finish;
      end

      if (captured_silu_data !== y_out_mem[0]) begin
        $error("silu_wrapper output mismatch for %s", case_name);
        $finish;
      end

      if (captured_scale_data !== output_scale_mem[0]) begin
        $error("silu_wrapper scale mismatch for %s", case_name);
        $finish;
      end

      if ((captured_silu_bus.tag.block_id != BLOCK_SILU) ||
          (captured_silu_bus.tag.gemm_mode != GEMM_GATE)) begin
        $error("silu_wrapper tag mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    gate_valid = 1'b0;
    gate_bus = '0;
    input_scale = '0;
    output_scale = '0;
    scale_ready = 1'b1;
    silu_ready = 1'b1;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    run_case("phase5_prefill_layer0_silu_gate_m0", PREFILL_BASE);
    run_case("phase5_decode_layer0_silu_gate_m0", DECODE_BASE);

    $display("PASS: tb_silu_wrapper");
    $finish;
  end

endmodule

module silu_core_hls_ip (
  input  logic                ap_clk,
  input  logic                ap_rst_n,
  input  logic                start_i,
  input  logic [ELEM_COUNT_W-1:0] elem_count_i,
  input  logic                in_valid_i,
  output logic                in_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] in_chunk_i,
  output logic                out_valid_o,
  input  logic                out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_o,
  output logic                busy_o,
  output logic                done_pulse_o
);

  typedef enum logic [1:0] {
    CORE_IDLE  = 2'd0,
    CORE_INPUT = 2'd1,
    CORE_OUT   = 2'd2
  } core_state_e;

  core_state_e state_q;
  logic [4:0] in_idx_q;
  logic [4:0] out_idx_q;
  logic [4:0] chunk_count_q;

  assign in_ready_o = (state_q == CORE_INPUT);
  assign out_valid_o = (state_q == CORE_OUT);
  assign out_chunk_o = tb_silu_wrapper.core_y_chunks_mem[out_idx_q];
  assign busy_o = (state_q != CORE_IDLE);

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q <= CORE_IDLE;
      in_idx_q <= '0;
      out_idx_q <= '0;
      chunk_count_q <= '0;
    end else begin
      unique case (state_q)
        CORE_IDLE: begin
          if (start_i) begin
            chunk_count_q <= (elem_count_i + N_TILE - 1) / N_TILE;
            in_idx_q <= '0;
            out_idx_q <= '0;
            state_q <= CORE_INPUT;
          end
        end

        CORE_INPUT: begin
          if (in_valid_i && in_ready_o) begin
            if (in_chunk_i !== tb_silu_wrapper.core_x_chunks_mem[in_idx_q]) begin
              $error("silu_core_hls_ip input chunk mismatch at %0d", in_idx_q);
              $display("expected=%h", tb_silu_wrapper.core_x_chunks_mem[in_idx_q]);
              $display("actual  =%h", in_chunk_i);
              $finish;
            end
            if (in_idx_q == (chunk_count_q - 1'b1)) begin
              state_q <= CORE_OUT;
            end
            in_idx_q <= in_idx_q + 1'b1;
          end
        end

        CORE_OUT: begin
          if (out_valid_o && out_ready_i) begin
            if (out_idx_q == (chunk_count_q - 1'b1)) begin
              done_pulse_o <= 1'b1;
              state_q <= CORE_IDLE;
            end
            out_idx_q <= out_idx_q + 1'b1;
          end
        end

        default: begin
          state_q <= CORE_IDLE;
        end
      endcase
    end
  end

endmodule
