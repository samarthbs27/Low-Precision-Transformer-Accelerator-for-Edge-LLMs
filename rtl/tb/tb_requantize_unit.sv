`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_requantize_unit;

  localparam string PREFILL_META_FILE  = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_requant_m0_n0.meta.memh";
  localparam string PREFILL_ACC_FILE   = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_requant_m0_n0.acc_packed.memh";
  localparam string PREFILL_SCALE_FILE = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_requant_m0_n0.scale_packed.memh";
  localparam string PREFILL_OUT_FILE   = "sim/golden_traces/phase3/rtl/phase3_prefill_layer0_q_proj_requant_m0_n0.out_expected_packed.memh";

  localparam string DECODE_META_FILE  = "sim/golden_traces/phase3/rtl/phase3_decode_layer0_q_proj_requant_m0_n0.meta.memh";
  localparam string DECODE_ACC_FILE   = "sim/golden_traces/phase3/rtl/phase3_decode_layer0_q_proj_requant_m0_n0.acc_packed.memh";
  localparam string DECODE_SCALE_FILE = "sim/golden_traces/phase3/rtl/phase3_decode_layer0_q_proj_requant_m0_n0.scale_packed.memh";
  localparam string DECODE_OUT_FILE   = "sim/golden_traces/phase3/rtl/phase3_decode_layer0_q_proj_requant_m0_n0.out_expected_packed.memh";

  acc_bus_t   acc_bus;
  scale_bus_t scale_bus;
  logic       nonnegative_only;
  act_bus_t   act_bus;

  logic [31:0]                trace_meta_mem [0:0];
  logic [(ACC_VECTOR_ELEMS * ACC_W)-1:0]      trace_acc_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0]  trace_scale_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0]      trace_out_mem [0:0];

  requantize_unit dut (
    .acc_i(acc_bus),
    .scale_i(scale_bus),
    .nonnegative_only_i(nonnegative_only),
    .act_o(act_bus)
  );

  task automatic load_trace_case(
    input string meta_file,
    input string acc_file,
    input string scale_file,
    input string out_file
  );
    begin
      $readmemh(meta_file, trace_meta_mem);
      $readmemh(acc_file, trace_acc_mem);
      $readmemh(scale_file, trace_scale_mem);
      $readmemh(out_file, trace_out_mem);
    end
  endtask

  task automatic run_trace_case(
    input string case_name,
    input string meta_file,
    input string acc_file,
    input string scale_file,
    input string out_file
  );
    int elem_count;
    begin
      load_trace_case(meta_file, acc_file, scale_file, out_file);
      elem_count = trace_meta_mem[0];

      acc_bus          = '0;
      scale_bus        = '0;
      nonnegative_only = 1'b0;

      acc_bus.tag.block_id   = BLOCK_REQUANTIZE;
      acc_bus.tag.gemm_mode  = GEMM_Q;
      acc_bus.tag.elem_count = ELEM_COUNT_W'(elem_count);
      acc_bus.tag.is_last    = 1'b1;
      acc_bus.data           = trace_acc_mem[0];
      scale_bus.data         = trace_scale_mem[0];

      #1;

      if ((act_bus.tag.block_id != BLOCK_REQUANTIZE) ||
          (act_bus.tag.gemm_mode != GEMM_Q) ||
          (act_bus.tag.elem_count != ELEM_COUNT_W'(elem_count))) begin
        $error("requantize_unit trace tag mismatch for %s", case_name);
        $finish;
      end

      if (act_bus.data !== trace_out_mem[0]) begin
        $error("requantize_unit trace mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

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

    nonnegative_only       = 1'b0;
    acc_bus                = '0;
    scale_bus              = '0;
    acc_bus.tag.elem_count = 16'd64;
    scale_bus.data[0]      = 32'h0001_0000; // 1.0 for lanes 0..31
    scale_bus.data[1]      = 32'h0000_8000; // 0.5 for lanes 32..63
    acc_bus.data[0]        = 32'sd10;
    acc_bus.data[1]        = -32'sd11;
    acc_bus.data[32]       = 32'sd10;
    acc_bus.data[33]       = 32'sd11;
    #1;

    if ((act_bus.data[0] != 8'sd10) ||
        (act_bus.data[1] != -8'sd11) ||
        (act_bus.data[32] != 8'sd5) ||
        (act_bus.data[33] != 8'sd6)) begin
      $error("requantize_unit per-bank scale indexing mismatch");
      $finish;
    end

    run_trace_case(
      "phase3_prefill_layer0_q_proj_requant_m0_n0",
      PREFILL_META_FILE,
      PREFILL_ACC_FILE,
      PREFILL_SCALE_FILE,
      PREFILL_OUT_FILE
    );

    run_trace_case(
      "phase3_decode_layer0_q_proj_requant_m0_n0",
      DECODE_META_FILE,
      DECODE_ACC_FILE,
      DECODE_SCALE_FILE,
      DECODE_OUT_FILE
    );

    $display("PASS: tb_requantize_unit");
    $finish;
  end

endmodule
