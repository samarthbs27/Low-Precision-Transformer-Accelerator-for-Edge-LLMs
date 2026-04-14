`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_embedding_quantizer;

  localparam int unsigned EMBED_ROW_W = D_MODEL * 16;
  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam string PREFILL_BASE = "sim/golden_traces/phase6/rtl/phase6_prefill_embedding_quantizer_batch0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase6/rtl/phase6_decode_embedding_quantizer_batch0";

  logic                               clk;
  logic                               rst_n;
  logic                               row_valid;
  logic                               row_ready;
  logic [(D_MODEL * 16)-1:0]          row_fp16;
  token_bus_t                         row_meta;
  logic                               scale_valid;
  logic                               scale_ready;
  scale_bus_t                         scale_bus;
  logic                               scale_out_valid;
  logic                               scale_out_ready;
  scale_bus_t                         scale_out_bus;
  logic                               act_valid;
  logic                               act_ready;
  act_bus_t                           act_bus;
  logic                               busy;
  logic                               done_pulse;

  logic [(D_MODEL * 16)-1:0]          rows_mem [0:M_TILE-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] act_tiles_mem [0:FEATURE_TILE_COUNT-1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_mem [0:0];
  logic [31:0]                        meta_mem [0:1];

  logic                               saw_scale;
  scale_bus_t                         captured_scale_bus;
  int                                 captured_tile_count;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_act_data [0:FEATURE_TILE_COUNT-1];
  block_id_e                          captured_act_block_id [0:FEATURE_TILE_COUNT-1];
  gemm_mode_e                         captured_act_gemm_mode [0:FEATURE_TILE_COUNT-1];
  logic [TILE_ID_W-1:0]               captured_act_tile_id [0:FEATURE_TILE_COUNT-1];
  logic [POS_W-1:0]                   captured_act_token_base [0:FEATURE_TILE_COUNT-1];
  logic [COUNT_W-1:0]                 captured_act_seq_count [0:FEATURE_TILE_COUNT-1];
  logic [ELEM_COUNT_W-1:0]            captured_act_elem_count [0:FEATURE_TILE_COUNT-1];
  logic                               captured_act_is_partial [0:FEATURE_TILE_COUNT-1];
  logic                               captured_act_is_last [0:FEATURE_TILE_COUNT-1];

  embedding_quantizer dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .row_valid_i(row_valid),
    .row_ready_o(row_ready),
    .row_fp16_i(row_fp16),
    .row_meta_i(row_meta),
    .scale_valid_i(scale_valid),
    .scale_ready_o(scale_ready),
    .scale_i(scale_bus),
    .scale_out_valid_o(scale_out_valid),
    .scale_out_ready_i(scale_out_ready),
    .scale_out_o(scale_out_bus),
    .act_valid_o(act_valid),
    .act_ready_i(act_ready),
    .act_o(act_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_scale <= 1'b0;
      captured_scale_bus <= '0;
      captured_tile_count <= 0;
      for (int idx = 0; idx < FEATURE_TILE_COUNT; idx++) begin
        captured_act_data[idx] <= '0;
        captured_act_block_id[idx] <= BLOCK_NONE;
        captured_act_gemm_mode[idx] <= GEMM_NONE;
        captured_act_tile_id[idx] <= '0;
        captured_act_token_base[idx] <= '0;
        captured_act_seq_count[idx] <= '0;
        captured_act_elem_count[idx] <= '0;
        captured_act_is_partial[idx] <= 1'b0;
        captured_act_is_last[idx] <= 1'b0;
      end
    end else begin
      if (scale_out_valid && scale_out_ready) begin
        saw_scale <= 1'b1;
        captured_scale_bus <= scale_out_bus;
      end
      if (act_valid && act_ready && (captured_tile_count < FEATURE_TILE_COUNT)) begin
        captured_act_data[captured_tile_count] <= act_bus.data;
        captured_act_block_id[captured_tile_count] <= act_bus.tag.block_id;
        captured_act_gemm_mode[captured_tile_count] <= act_bus.tag.gemm_mode;
        captured_act_tile_id[captured_tile_count] <= act_bus.tag.tile_id;
        captured_act_token_base[captured_tile_count] <= act_bus.tag.token_base;
        captured_act_seq_count[captured_tile_count] <= act_bus.tag.seq_count;
        captured_act_elem_count[captured_tile_count] <= act_bus.tag.elem_count;
        captured_act_is_partial[captured_tile_count] <= act_bus.tag.is_partial;
        captured_act_is_last[captured_tile_count] <= act_bus.tag.is_last;
        captured_tile_count <= captured_tile_count + 1;
      end
    end
  end

  task automatic reset_capture;
    begin
      @(negedge clk);
      saw_scale = 1'b0;
      captured_scale_bus = '0;
      captured_tile_count = 0;
      for (int idx = 0; idx < FEATURE_TILE_COUNT; idx++) begin
        captured_act_data[idx] = '0;
        captured_act_block_id[idx] = BLOCK_NONE;
        captured_act_gemm_mode[idx] = GEMM_NONE;
        captured_act_tile_id[idx] = '0;
        captured_act_token_base[idx] = '0;
        captured_act_seq_count[idx] = '0;
        captured_act_elem_count[idx] = '0;
        captured_act_is_partial[idx] = 1'b0;
        captured_act_is_last[idx] = 1'b0;
      end
    end
  endtask

  task automatic load_case(input string base);
    begin
      for (int idx = 0; idx < M_TILE; idx++) begin
        rows_mem[idx] = '0;
      end
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".rows_fp16_packed.memh"}, rows_mem);
      $readmemh({base, ".scale_packed.memh"}, scale_mem);
      $readmemh({base, ".act_tiles_expected_packed.memh"}, act_tiles_mem);
    end
  endtask

  task automatic wait_for_tiles(input string case_name);
    int timeout_cycles;
    begin
      timeout_cycles = 0;
      while ((captured_tile_count < FEATURE_TILE_COUNT) && (timeout_cycles < 512)) begin
        @(posedge clk);
        timeout_cycles++;
      end
      if (captured_tile_count != FEATURE_TILE_COUNT) begin
        $error("embedding_quantizer timed out waiting for all act tiles for %s", case_name);
        $finish;
      end
      @(posedge clk);
      if (busy || act_valid || done_pulse) begin
        $error("embedding_quantizer did not return idle after output for %s", case_name);
        $finish;
      end
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int row_count;
    int token_base;
    int mismatch_count;
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] actual_tile;
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] expected_tile;
    begin
      load_case(base);
      row_count = meta_mem[0];
      token_base = meta_mem[1];

      reset_capture();
      scale_bus = '0;
      scale_bus.tag.block_id = BLOCK_EMBED;
      scale_bus.tag.gemm_mode = GEMM_NONE;
      scale_bus.data = scale_mem[0];
      @(negedge clk);
      scale_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!scale_ready);
      @(negedge clk);
      scale_valid = 1'b0;

      for (int row_idx = 0; row_idx < row_count; row_idx++) begin
        row_fp16 = rows_mem[row_idx];
        row_meta = '0;
        row_meta.token_id = 32'd100 + row_idx;
        row_meta.token_count = row_idx + 1;
        row_meta.tag.block_id = BLOCK_EMBED;
        row_meta.tag.gemm_mode = GEMM_NONE;
        row_meta.tag.tile_id = TILE_ID_W'(row_idx);
        row_meta.tag.token_base = POS_W'(token_base + row_idx);
        row_meta.tag.seq_count = COUNT_W'(row_count - row_idx);
        row_meta.tag.elem_count = 16'd1;
        row_meta.tag.is_last = (row_idx == (row_count - 1));
        row_meta.tag.is_partial = 1'b1;
        @(negedge clk);
        row_valid = 1'b1;
        do begin
          @(posedge clk);
        end while (!row_ready);
        @(negedge clk);
        row_valid = 1'b0;
      end

      wait_for_tiles(case_name);

      if (!saw_scale ||
          (captured_scale_bus.tag.block_id != BLOCK_EMBED) ||
          (captured_scale_bus.tag.token_base != POS_W'(token_base)) ||
          (captured_scale_bus.tag.seq_count != COUNT_W'(row_count)) ||
          (captured_scale_bus.data != scale_mem[0])) begin
        $error("embedding_quantizer scale output mismatch for %s", case_name);
        $finish;
      end

      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        if (captured_act_data[tile_idx] !== act_tiles_mem[tile_idx]) begin
          actual_tile = captured_act_data[tile_idx];
          expected_tile = act_tiles_mem[tile_idx];
          mismatch_count = 0;
          for (int lane_idx = 0; lane_idx < ACT_VECTOR_ELEMS; lane_idx++) begin
            if (actual_tile[lane_idx] !== expected_tile[lane_idx]) begin
              if (mismatch_count < 8) begin
                $display(
                  "embedding_quantizer mismatch %s tile %0d lane %0d actual=%0d expected=%0d",
                  case_name,
                  tile_idx,
                  lane_idx,
                  actual_tile[lane_idx],
                  expected_tile[lane_idx]
                );
              end
              mismatch_count++;
            end
          end
          $error("embedding_quantizer tile data mismatch for %s at tile %0d", case_name, tile_idx);
          $finish;
        end
        if ((captured_act_block_id[tile_idx] != BLOCK_EMBED) ||
            (captured_act_gemm_mode[tile_idx] != GEMM_NONE) ||
            (captured_act_tile_id[tile_idx] != TILE_ID_W'(tile_idx)) ||
            (captured_act_token_base[tile_idx] != POS_W'(token_base)) ||
            (captured_act_seq_count[tile_idx] != COUNT_W'(row_count)) ||
            (captured_act_elem_count[tile_idx] != ELEM_COUNT_W'(row_count * N_TILE)) ||
            (captured_act_is_partial[tile_idx] != (row_count != M_TILE)) ||
            (captured_act_is_last[tile_idx] != (tile_idx == FEATURE_TILE_COUNT - 1))) begin
          $error("embedding_quantizer tile tag mismatch for %s at tile %0d", case_name, tile_idx);
          $finish;
        end
      end
    end
  endtask

  initial begin
    logic [15:0] directed_rows [0:1][0:D_MODEL-1];
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] expected_directed_tile0;

    clk = 1'b0;
    rst_n = 1'b0;
    row_valid = 1'b0;
    row_fp16 = '0;
    row_meta = '0;
    scale_valid = 1'b0;
    scale_bus = '0;
    scale_out_ready = 1'b1;
    act_ready = 1'b1;
    saw_scale = 1'b0;
    captured_scale_bus = '0;
    captured_tile_count = 0;
    for (int idx = 0; idx < FEATURE_TILE_COUNT; idx++) begin
      captured_act_data[idx] = '0;
      captured_act_block_id[idx] = BLOCK_NONE;
      captured_act_gemm_mode[idx] = GEMM_NONE;
      captured_act_tile_id[idx] = '0;
      captured_act_token_base[idx] = '0;
      captured_act_seq_count[idx] = '0;
      captured_act_elem_count[idx] = '0;
      captured_act_is_partial[idx] = 1'b0;
      captured_act_is_last[idx] = 1'b0;
    end

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (int row_idx = 0; row_idx < 2; row_idx++) begin
      for (int col_idx = 0; col_idx < D_MODEL; col_idx++) begin
        directed_rows[row_idx][col_idx] = 16'h0000;
      end
    end
    directed_rows[0][0] = 16'h3c00; // 1.0
    directed_rows[0][1] = 16'hc000; // -2.0
    directed_rows[1][0] = 16'h4200; // 3.0
    directed_rows[1][1] = 16'hbc00; // -1.0
    expected_directed_tile0 = '0;
    expected_directed_tile0[0] = 8'sd1;
    expected_directed_tile0[1] = -8'sd2;
    expected_directed_tile0[32] = 8'sd3;
    expected_directed_tile0[33] = -8'sd1;

    reset_capture();
    scale_bus.data = {SCALE_VECTOR_ELEMS{32'h0001_0000}};
    @(negedge clk);
    scale_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!scale_ready);
    @(negedge clk);
    scale_valid = 1'b0;

    for (int row_idx = 0; row_idx < 2; row_idx++) begin
      row_fp16 = '0;
      for (int col_idx = 0; col_idx < D_MODEL; col_idx++) begin
        row_fp16[(col_idx * 16) +: 16] = directed_rows[row_idx][col_idx];
      end
      row_meta = '0;
      row_meta.tag.block_id = BLOCK_EMBED;
      row_meta.tag.token_base = POS_W'(row_idx);
      row_meta.tag.is_last = (row_idx == 1);
      @(negedge clk);
      row_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!row_ready);
      @(negedge clk);
      row_valid = 1'b0;
    end

    wait_for_tiles("directed");
    if (captured_act_data[0] !== expected_directed_tile0) begin
      $error("embedding_quantizer directed first-tile mismatch");
      $finish;
    end

    run_case("phase6_prefill_embedding_quantizer_batch0", PREFILL_BASE);
    run_case("phase6_decode_embedding_quantizer_batch0", DECODE_BASE);

    $display("PASS: tb_embedding_quantizer");
    $finish;
  end

endmodule
