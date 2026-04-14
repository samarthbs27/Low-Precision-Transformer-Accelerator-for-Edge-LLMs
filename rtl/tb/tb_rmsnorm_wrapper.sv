`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_rmsnorm_wrapper;

  localparam int unsigned CHUNK_ELEMS = N_TILE;
  localparam int unsigned CHUNK_W = CHUNK_ELEMS * SCALE_W;
  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned GAMMA_BEAT_COUNT = D_MODEL / (DMA_BEAT_W / 16);

  localparam string PREFILL_BASE = "sim/golden_traces/phase5/rtl/phase5_prefill_layer0_rmsnorm1";
  localparam string DECODE_BASE  = "sim/golden_traces/phase5/rtl/phase5_decode_layer0_rmsnorm1";

  logic clk;
  logic rst_n;
  block_id_e block_id;
  logic act_valid;
  logic act_ready;
  act_bus_t act_bus;
  logic [SCALE_W-1:0] input_scale;
  logic [SCALE_W-1:0] output_scale;
  logic gamma_valid;
  logic gamma_ready;
  logic [DMA_BEAT_W-1:0] gamma_bus;
  logic gamma_last;
  logic scale_valid;
  logic scale_ready;
  scale_bus_t scale_bus;
  logic norm_valid;
  logic norm_ready;
  act_bus_t norm_bus;
  logic busy;
  logic done_pulse;

  logic [31:0] meta_mem [0:4];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] x_tiles_mem [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] y_tiles_mem [0:FEATURE_TILE_COUNT-1];
  logic [DMA_BEAT_W-1:0] gamma_beats_mem [0:GAMMA_BEAT_COUNT-1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] input_scale_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] output_scale_mem [0:0];
  logic [CHUNK_W-1:0] core_x_chunks_mem [0:(FEATURE_TILE_COUNT * M_TILE)-1];
  logic [CHUNK_W-1:0] core_gamma_chunks_mem [0:FEATURE_TILE_COUNT-1];
  logic [CHUNK_W-1:0] core_y_chunks_mem [0:(FEATURE_TILE_COUNT * M_TILE)-1];

  logic saw_scale;
  integer captured_tile_count;
  scale_bus_t captured_scale_bus;
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] captured_scale_data;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_norm_data [0:FEATURE_TILE_COUNT-1];
  block_id_e captured_norm_block_id [0:FEATURE_TILE_COUNT-1];
  gemm_mode_e captured_norm_gemm_mode [0:FEATURE_TILE_COUNT-1];
  logic [TILE_ID_W-1:0] captured_norm_tile_id [0:FEATURE_TILE_COUNT-1];
  logic [ELEM_COUNT_W-1:0] captured_norm_elem_count [0:FEATURE_TILE_COUNT-1];
  logic captured_norm_is_partial [0:FEATURE_TILE_COUNT-1];
  logic captured_norm_is_last [0:FEATURE_TILE_COUNT-1];

  rmsnorm_wrapper dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .block_id_i(block_id),
    .act_valid_i(act_valid),
    .act_ready_o(act_ready),
    .act_i(act_bus),
    .input_scale_i(input_scale),
    .output_scale_i(output_scale),
    .gamma_valid_i(gamma_valid),
    .gamma_ready_o(gamma_ready),
    .gamma_i(gamma_bus),
    .gamma_last_i(gamma_last),
    .scale_valid_o(scale_valid),
    .scale_ready_i(scale_ready),
    .scale_o(scale_bus),
    .norm_valid_o(norm_valid),
    .norm_ready_i(norm_ready),
    .norm_o(norm_bus),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_scale <= 1'b0;
      captured_tile_count <= 0;
      captured_scale_bus <= '0;
      for (int idx = 0; idx < FEATURE_TILE_COUNT; idx++) begin
        captured_norm_data[idx] <= '0;
        captured_norm_block_id[idx] <= BLOCK_NONE;
        captured_norm_gemm_mode[idx] <= GEMM_NONE;
        captured_norm_tile_id[idx] <= '0;
        captured_norm_elem_count[idx] <= '0;
        captured_norm_is_partial[idx] <= 1'b0;
        captured_norm_is_last[idx] <= 1'b0;
      end
    end else begin
      if (scale_valid && scale_ready) begin
        saw_scale <= 1'b1;
        captured_scale_bus <= scale_bus;
        captured_scale_data <= scale_bus.data;
      end
      if (norm_valid && norm_ready) begin
        captured_norm_data[captured_tile_count] <= norm_bus.data;
        captured_norm_block_id[captured_tile_count] <= norm_bus.tag.block_id;
        captured_norm_gemm_mode[captured_tile_count] <= norm_bus.tag.gemm_mode;
        captured_norm_tile_id[captured_tile_count] <= norm_bus.tag.tile_id;
        captured_norm_elem_count[captured_tile_count] <= norm_bus.tag.elem_count;
        captured_norm_is_partial[captured_tile_count] <= norm_bus.tag.is_partial;
        captured_norm_is_last[captured_tile_count] <= norm_bus.tag.is_last;
        captured_tile_count <= captured_tile_count + 1;
      end
    end
  end

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".x_tiles_packed.memh"}, x_tiles_mem);
      $readmemh({base, ".y_tiles_expected_packed.memh"}, y_tiles_mem);
      $readmemh({base, ".gamma_beats_packed.memh"}, gamma_beats_mem);
      $readmemh({base, ".input_scale_packed.memh"}, input_scale_mem);
      $readmemh({base, ".output_scale_packed.memh"}, output_scale_mem);
      $readmemh({base, ".core_x_chunks_packed.memh"}, core_x_chunks_mem);
      $readmemh({base, ".core_gamma_chunks_packed.memh"}, core_gamma_chunks_mem);
      $readmemh({base, ".core_y_chunks_packed.memh"}, core_y_chunks_mem);
    end
  endtask

  task automatic send_gamma_stream;
    begin
      for (int beat_idx = 0; beat_idx < GAMMA_BEAT_COUNT; beat_idx++) begin
        @(negedge clk);
        gamma_bus = gamma_beats_mem[beat_idx];
        gamma_last = (beat_idx == (GAMMA_BEAT_COUNT - 1));
        gamma_valid = 1'b1;
        do begin
          @(posedge clk);
        end while (!gamma_ready);
        @(negedge clk);
        gamma_valid = 1'b0;
      end
    end
  endtask

  task automatic send_act_stream(input int row_count);
    begin
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        @(negedge clk);
        act_bus = '0;
        act_bus.tag.block_id = BLOCK_REQUANTIZE;
        act_bus.tag.gemm_mode = GEMM_NONE;
        act_bus.tag.tile_id = TILE_ID_W'(tile_idx);
        act_bus.tag.elem_count = ELEM_COUNT_W'(row_count * N_TILE);
        act_bus.tag.is_partial = (row_count != M_TILE);
        act_bus.tag.is_last = (tile_idx == (FEATURE_TILE_COUNT - 1));
        act_bus.data = x_tiles_mem[tile_idx];
        act_valid = 1'b1;
        do begin
          @(posedge clk);
        end while (!act_ready);
        @(negedge clk);
        act_valid = 1'b0;
      end
    end
  endtask

  task automatic run_case(
    input string case_name,
    input string base,
    input block_id_e expected_block_id
  );
    int row_count;
    int cycle_count;
    begin
      load_case(base);
      row_count = meta_mem[0];
      block_id = expected_block_id;
      input_scale = input_scale_mem[0][SCALE_W-1:0];
      output_scale = output_scale_mem[0][SCALE_W-1:0];
      act_valid = 1'b0;
      gamma_valid = 1'b0;
      gamma_last = 1'b0;
      act_bus = '0;
      gamma_bus = '0;
      scale_ready = 1'b1;
      norm_ready = 1'b1;
      saw_scale = 1'b0;
      captured_tile_count = 0;

      fork
        send_gamma_stream();
        send_act_stream(row_count);
      join

      cycle_count = 0;
      while (!done_pulse && (cycle_count < 3000)) begin
        @(posedge clk);
        cycle_count = cycle_count + 1;
      end

      if (!done_pulse) begin
        $error("rmsnorm_wrapper timeout for %s", case_name);
        $finish;
      end

      @(posedge clk);

      if (!saw_scale) begin
        $error("rmsnorm_wrapper missing scale output for %s", case_name);
        $finish;
      end

      if (captured_scale_data !== output_scale_mem[0]) begin
        $error("rmsnorm_wrapper scale mismatch for %s", case_name);
        $finish;
      end

      if ((captured_scale_bus.tag.block_id != expected_block_id) ||
          (captured_scale_bus.tag.gemm_mode != GEMM_NONE) ||
          (captured_scale_bus.tag.tile_id != '0) ||
          (captured_scale_bus.tag.elem_count != ELEM_COUNT_W'(SCALE_VECTOR_ELEMS)) ||
          (captured_scale_bus.tag.is_partial != 1'b0) ||
          (captured_scale_bus.tag.is_last != 1'b1)) begin
        $error("rmsnorm_wrapper scale tag mismatch for %s", case_name);
        $finish;
      end

      if (captured_tile_count != FEATURE_TILE_COUNT) begin
        $error("rmsnorm_wrapper expected %0d output tiles, got %0d for %s",
               FEATURE_TILE_COUNT, captured_tile_count, case_name);
        $finish;
      end

      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        if (captured_norm_data[tile_idx] !== y_tiles_mem[tile_idx]) begin
          $error("rmsnorm_wrapper output tile mismatch at %0d for %s", tile_idx, case_name);
          $display("expected=%h", y_tiles_mem[tile_idx]);
          $display("actual  =%h", captured_norm_data[tile_idx]);
          $finish;
        end

        if ((captured_norm_block_id[tile_idx] != expected_block_id) ||
            (captured_norm_gemm_mode[tile_idx] != GEMM_NONE) ||
            (captured_norm_tile_id[tile_idx] != TILE_ID_W'(tile_idx)) ||
            (captured_norm_elem_count[tile_idx] != ELEM_COUNT_W'(row_count * N_TILE)) ||
            (captured_norm_is_partial[tile_idx] != (row_count != M_TILE)) ||
            (captured_norm_is_last[tile_idx] != (tile_idx == (FEATURE_TILE_COUNT - 1)))) begin
          $error("rmsnorm_wrapper output tag mismatch at %0d for %s", tile_idx, case_name);
          $finish;
        end
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    block_id = BLOCK_RMSNORM1;
    act_valid = 1'b0;
    act_bus = '0;
    input_scale = '0;
    output_scale = '0;
    gamma_valid = 1'b0;
    gamma_bus = '0;
    gamma_last = 1'b0;
    scale_ready = 1'b1;
    norm_ready = 1'b1;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    run_case("phase5_prefill_layer0_rmsnorm1", PREFILL_BASE, BLOCK_RMSNORM1);
    run_case("phase5_decode_layer0_rmsnorm1", DECODE_BASE, BLOCK_RMSNORM1);
    run_case("phase5_prefill_layer0_rmsnorm2", PREFILL_BASE, BLOCK_RMSNORM2);

    $display("PASS: tb_rmsnorm_wrapper");
    $finish;
  end

endmodule

module rmsnorm_core_hls_ip (
  input  logic                ap_clk,
  input  logic                ap_rst_n,
  input  logic                start_i,
  input  logic [COUNT_W-1:0]  row_count_i,
  input  logic [15:0]         feature_count_i,
  input  logic [31:0]         epsilon_q16_i,
  input  logic                act_valid_i,
  output logic                act_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] act_chunk_i,
  input  logic                gamma_valid_i,
  output logic                gamma_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] gamma_chunk_i,
  output logic                out_valid_o,
  input  logic                out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_o,
  output logic                busy_o,
  output logic                done_pulse_o
);

  typedef enum logic [1:0] {
    CORE_IDLE  = 2'd0,
    CORE_GAMMA = 2'd1,
    CORE_ACT   = 2'd2,
    CORE_OUT   = 2'd3
  } core_state_e;

  core_state_e state_q;
  logic [6:0] gamma_idx_q;
  logic [6:0] feature_idx_q;
  logic [COUNT_W-1:0] row_idx_q;

  assign gamma_ready_o = (state_q == CORE_GAMMA);
  assign act_ready_o   = (state_q == CORE_ACT);
  assign out_valid_o   = (state_q == CORE_OUT);
  assign out_chunk_o   = tb_rmsnorm_wrapper.core_y_chunks_mem[(feature_idx_q * row_count_i) + row_idx_q];
  assign busy_o        = (state_q != CORE_IDLE);

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q <= CORE_IDLE;
      gamma_idx_q <= '0;
      feature_idx_q <= '0;
      row_idx_q <= '0;
    end else begin
      unique case (state_q)
        CORE_IDLE: begin
          if (start_i) begin
            if ((feature_count_i != 16'd2048) || (epsilon_q16_i == '0)) begin
              $error("rmsnorm_core_hls_ip unexpected wrapper configuration");
              $finish;
            end
            gamma_idx_q <= '0;
            feature_idx_q <= '0;
            row_idx_q <= '0;
            state_q <= CORE_GAMMA;
          end
        end

        CORE_GAMMA: begin
          if (gamma_valid_i && gamma_ready_o) begin
            if (gamma_chunk_i !== tb_rmsnorm_wrapper.core_gamma_chunks_mem[gamma_idx_q]) begin
              $error("rmsnorm_core_hls_ip gamma chunk mismatch at %0d", gamma_idx_q);
              $display("expected=%h", tb_rmsnorm_wrapper.core_gamma_chunks_mem[gamma_idx_q]);
              $display("actual  =%h", gamma_chunk_i);
              $finish;
            end
            if (gamma_idx_q == (tb_rmsnorm_wrapper.FEATURE_TILE_COUNT - 1)) begin
              state_q <= CORE_ACT;
            end
            gamma_idx_q <= gamma_idx_q + 1'b1;
          end
        end

        CORE_ACT: begin
          if (act_valid_i && act_ready_o) begin
            if (act_chunk_i !== tb_rmsnorm_wrapper.core_x_chunks_mem[(feature_idx_q * row_count_i) + row_idx_q]) begin
              $error("rmsnorm_core_hls_ip act chunk mismatch at feature=%0d row=%0d", feature_idx_q, row_idx_q);
              $display("expected=%h", tb_rmsnorm_wrapper.core_x_chunks_mem[(feature_idx_q * row_count_i) + row_idx_q]);
              $display("actual  =%h", act_chunk_i);
              $finish;
            end
            if (row_idx_q == (row_count_i - 1'b1)) begin
              row_idx_q <= '0;
              if (feature_idx_q == (tb_rmsnorm_wrapper.FEATURE_TILE_COUNT - 1)) begin
                feature_idx_q <= '0;
                row_idx_q <= '0;
                state_q <= CORE_OUT;
              end else begin
                feature_idx_q <= feature_idx_q + 1'b1;
              end
            end else begin
              row_idx_q <= row_idx_q + 1'b1;
            end
          end
        end

        CORE_OUT: begin
          if (out_valid_o && out_ready_i) begin
            if (row_idx_q == (row_count_i - 1'b1)) begin
              row_idx_q <= '0;
              if (feature_idx_q == (tb_rmsnorm_wrapper.FEATURE_TILE_COUNT - 1)) begin
                done_pulse_o <= 1'b1;
                state_q <= CORE_IDLE;
              end
              feature_idx_q <= feature_idx_q + 1'b1;
            end else begin
              row_idx_q <= row_idx_q + 1'b1;
            end
          end
        end

        default: begin
          state_q <= CORE_IDLE;
        end
      endcase
    end
  end

endmodule
