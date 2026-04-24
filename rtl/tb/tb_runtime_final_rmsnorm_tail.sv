`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_runtime_final_rmsnorm_tail;

  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned GAMMA_BEAT_COUNT = D_MODEL / (DMA_BEAT_W / 16);
  localparam logic [HBM_ADDR_W-1:0] GAMMA_BASE_ADDR = 64'h0000_0000_0800_0000;
  localparam string PREFILL_BASE = "sim/golden_traces/phase5/rtl/phase5_prefill_layer0_rmsnorm1";
  localparam string DECODE_BASE = "sim/golden_traces/phase5/rtl/phase5_decode_layer0_rmsnorm1";

  logic clk;
  logic rst_n;
  logic launch;
  logic abort_req;
  logic rd_desc_valid;
  logic rd_desc_ready;
  dma_desc_t rd_desc;
  logic rd_data_valid;
  logic rd_data_ready;
  logic [DMA_BEAT_W-1:0] rd_data;
  logic hidden_scale_valid;
  logic hidden_scale_ready;
  scale_bus_t hidden_scale_bus;
  logic hidden_act_valid;
  logic hidden_act_ready;
  act_bus_t hidden_act_bus;
  logic norm_scale_valid;
  logic norm_scale_ready;
  scale_bus_t norm_scale_bus;
  logic norm_act_valid;
  logic norm_act_ready;
  act_bus_t norm_act_bus;
  logic norm_done_pulse;
  logic busy;

  logic [31:0] meta_mem [0:4];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] x_tiles_mem [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] y_tiles_mem [0:FEATURE_TILE_COUNT-1];
  logic [DMA_BEAT_W-1:0] gamma_beats_mem [0:GAMMA_BEAT_COUNT-1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] input_scale_mem [0:0];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] output_scale_mem [0:0];

  logic                  rd_burst_active_q;
  logic [7:0]            rd_beat_idx_q;
  logic [7:0]            rd_beats_total_q;
  int                    captured_tile_count;
  logic                  saw_scale;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_tiles [0:FEATURE_TILE_COUNT-1];
  logic [LAYER_ID_W-1:0] captured_layer_id [0:FEATURE_TILE_COUNT-1];
  block_id_e             captured_block_id [0:FEATURE_TILE_COUNT-1];
  gemm_mode_e            captured_gemm_mode [0:FEATURE_TILE_COUNT-1];
  logic [TILE_ID_W-1:0]  captured_tile_id [0:FEATURE_TILE_COUNT-1];
  logic [TOKEN_W-1:0]    captured_token_base [0:FEATURE_TILE_COUNT-1];
  logic [COUNT_W-1:0]    captured_seq_count [0:FEATURE_TILE_COUNT-1];
  logic [Q_HEAD_ID_W-1:0] captured_q_head_id [0:FEATURE_TILE_COUNT-1];
  logic [KV_HEAD_ID_W-1:0] captured_kv_head_id [0:FEATURE_TILE_COUNT-1];
  logic [ELEM_COUNT_W-1:0] captured_elem_count [0:FEATURE_TILE_COUNT-1];
  logic                  captured_is_last [0:FEATURE_TILE_COUNT-1];
  logic                  captured_is_partial [0:FEATURE_TILE_COUNT-1];
  scale_bus_t            captured_scale;

  runtime_final_rmsnorm_tail dut (
    .ap_clk              (clk),
    .ap_rst_n            (rst_n),
    .launch_i            (launch),
    .abort_req_i         (abort_req),
    .gamma_base_addr_i   (GAMMA_BASE_ADDR),
    .output_scale_i      (output_scale_mem[0][SCALE_W-1:0]),
    .rd_desc_valid_o     (rd_desc_valid),
    .rd_desc_ready_i     (rd_desc_ready),
    .rd_desc_o           (rd_desc),
    .rd_data_valid_i     (rd_data_valid),
    .rd_data_ready_o     (rd_data_ready),
    .rd_data_i           (rd_data),
    .hidden_scale_valid_i(hidden_scale_valid),
    .hidden_scale_ready_o(hidden_scale_ready),
    .hidden_scale_i      (hidden_scale_bus),
    .hidden_act_valid_i  (hidden_act_valid),
    .hidden_act_ready_o  (hidden_act_ready),
    .hidden_act_i        (hidden_act_bus),
    .norm_scale_valid_o  (norm_scale_valid),
    .norm_scale_ready_i  (norm_scale_ready),
    .norm_scale_o        (norm_scale_bus),
    .norm_act_valid_o    (norm_act_valid),
    .norm_act_ready_i    (norm_act_ready),
    .norm_act_o          (norm_act_bus),
    .norm_done_pulse_o   (norm_done_pulse),
    .busy_o              (busy)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_burst_active_q <= 1'b0;
      rd_data_valid <= 1'b0;
      rd_data <= '0;
      rd_beat_idx_q <= '0;
      rd_beats_total_q <= '0;
      captured_tile_count <= 0;
      saw_scale <= 1'b0;
      captured_scale <= '0;
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        captured_tiles[tile_idx] <= '0;
        captured_layer_id[tile_idx] <= '0;
        captured_block_id[tile_idx] <= BLOCK_NONE;
        captured_gemm_mode[tile_idx] <= GEMM_NONE;
        captured_tile_id[tile_idx] <= '0;
        captured_token_base[tile_idx] <= '0;
        captured_seq_count[tile_idx] <= '0;
        captured_q_head_id[tile_idx] <= '0;
        captured_kv_head_id[tile_idx] <= '0;
        captured_elem_count[tile_idx] <= '0;
        captured_is_last[tile_idx] <= 1'b0;
        captured_is_partial[tile_idx] <= 1'b0;
      end
    end else begin
      if (rd_desc_valid && rd_desc_ready) begin
        if ((rd_desc.tensor_id != TENSOR_FINAL_RMS_GAMMA) ||
            (rd_desc.addr != GAMMA_BASE_ADDR)) begin
          $error("runtime_final_rmsnorm_tail unexpected gamma descriptor tensor=%0d addr=0x%0h",
                 rd_desc.tensor_id,
                 rd_desc.addr);
          $finish;
        end
        rd_burst_active_q <= 1'b1;
        rd_beat_idx_q <= '0;
        rd_beats_total_q <= (rd_desc.byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
        rd_data_valid <= 1'b0;
      end

      if (rd_burst_active_q && !rd_data_valid) begin
        rd_data_valid <= 1'b1;
        rd_data <= gamma_beats_mem[rd_beat_idx_q];
      end else if (rd_data_valid && rd_data_ready) begin
        if (rd_beat_idx_q + 1 >= rd_beats_total_q) begin
          rd_burst_active_q <= 1'b0;
          rd_data_valid <= 1'b0;
          rd_beat_idx_q <= '0;
        end else begin
          rd_beat_idx_q <= rd_beat_idx_q + 1'b1;
          rd_data <= gamma_beats_mem[rd_beat_idx_q + 1'b1];
        end
      end

      if (norm_scale_valid && norm_scale_ready) begin
        saw_scale <= 1'b1;
        captured_scale <= norm_scale_bus;
      end

      if (norm_act_valid && norm_act_ready &&
          (captured_tile_count < FEATURE_TILE_COUNT)) begin
        captured_tiles[captured_tile_count] <= norm_act_bus.data;
        captured_layer_id[captured_tile_count] <= norm_act_bus.tag.layer_id;
        captured_block_id[captured_tile_count] <= norm_act_bus.tag.block_id;
        captured_gemm_mode[captured_tile_count] <= norm_act_bus.tag.gemm_mode;
        captured_tile_id[captured_tile_count] <= norm_act_bus.tag.tile_id;
        captured_token_base[captured_tile_count] <= norm_act_bus.tag.token_base;
        captured_seq_count[captured_tile_count] <= norm_act_bus.tag.seq_count;
        captured_q_head_id[captured_tile_count] <= norm_act_bus.tag.q_head_id;
        captured_kv_head_id[captured_tile_count] <= norm_act_bus.tag.kv_head_id;
        captured_elem_count[captured_tile_count] <= norm_act_bus.tag.elem_count;
        captured_is_last[captured_tile_count] <= norm_act_bus.tag.is_last;
        captured_is_partial[captured_tile_count] <= norm_act_bus.tag.is_partial;
        captured_tile_count <= captured_tile_count + 1;
      end
    end
  end

  assign rd_desc_ready = 1'b1;
  assign norm_scale_ready = 1'b1;
  assign norm_act_ready = 1'b1;

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".x_tiles_packed.memh"}, x_tiles_mem);
      $readmemh({base, ".y_tiles_expected_packed.memh"}, y_tiles_mem);
      $readmemh({base, ".gamma_beats_packed.memh"}, gamma_beats_mem);
      $readmemh({base, ".input_scale_packed.memh"}, input_scale_mem);
      $readmemh({base, ".output_scale_packed.memh"}, output_scale_mem);
    end
  endtask

  task automatic drive_hidden_stream(input int row_count);
    int wait_cycles;
    begin
      hidden_scale_bus = '0;
      hidden_scale_bus.data = input_scale_mem[0];
      hidden_scale_bus.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
      hidden_scale_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
      hidden_scale_bus.tag.gemm_mode = GEMM_NONE;
      hidden_scale_bus.tag.tile_id = '0;
      hidden_scale_bus.tag.token_base = '0;
      hidden_scale_bus.tag.seq_count = COUNT_W'(row_count);
      hidden_scale_bus.tag.q_head_id = '0;
      hidden_scale_bus.tag.kv_head_id = '0;
      hidden_scale_bus.tag.elem_count = ELEM_COUNT_W'(D_MODEL);
      hidden_scale_bus.tag.is_last = 1'b1;
      hidden_scale_bus.tag.is_partial = 1'b0;

      hidden_scale_valid = 1'b1;
      wait_cycles = 0;
      while (!hidden_scale_ready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 2000) begin
          $error("runtime_final_rmsnorm_tail timeout waiting for hidden_scale_ready");
          $display("debug: hidden_scale_ready=%b scale_seen=%b gamma_req=%b gamma_busy=%b rms_state=%0d",
                   hidden_scale_ready,
                   dut.scale_seen_q,
                   dut.gamma_req_valid_q,
                   dut.gamma_reader_busy,
                   dut.u_final_rmsnorm_wrapper.state_q);
          $finish;
        end
      end
      @(negedge clk);
      hidden_scale_valid = 1'b0;

      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        hidden_act_bus = '0;
        hidden_act_bus.data = x_tiles_mem[tile_idx];
        hidden_act_bus.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
        hidden_act_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
        hidden_act_bus.tag.gemm_mode = GEMM_NONE;
        hidden_act_bus.tag.tile_id = TILE_ID_W'(tile_idx);
        hidden_act_bus.tag.token_base = '0;
        hidden_act_bus.tag.seq_count = COUNT_W'(row_count);
        hidden_act_bus.tag.q_head_id = '0;
        hidden_act_bus.tag.kv_head_id = '0;
        hidden_act_bus.tag.elem_count = ELEM_COUNT_W'(row_count * N_TILE);
        hidden_act_bus.tag.is_last = (tile_idx == (FEATURE_TILE_COUNT - 1));
        hidden_act_bus.tag.is_partial = (row_count != M_TILE);

        hidden_act_valid = 1'b1;
        wait_cycles = 0;
        while (!hidden_act_ready) begin
          @(negedge clk);
          wait_cycles++;
          if (wait_cycles > 4000) begin
            $error("runtime_final_rmsnorm_tail timeout waiting for hidden_act_ready tile=%0d", tile_idx);
            $finish;
          end
        end
        @(negedge clk);
        hidden_act_valid = 1'b0;
      end
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    int row_count;
    int timeout_cycles;
    bit mismatch_reported;
    int expected_lane;
    int actual_lane;
    begin
      load_case(base);
      row_count = meta_mem[0];
      repeat (4) @(negedge clk);
      launch = 1'b1;
      @(negedge clk);
      launch = 1'b0;

      fork
        drive_hidden_stream(row_count);
      join

      timeout_cycles = 0;
      while (!norm_done_pulse) begin
        @(negedge clk);
        timeout_cycles++;
        if (timeout_cycles > 12000) begin
          $error("runtime_final_rmsnorm_tail timeout for %s", case_name);
          $display("debug: busy=%b gamma_state=%0d gamma_busy=%b rms_state=%0d core_state=%0d scale_seen=%b captured_tiles=%0d",
                   busy,
                   dut.u_gamma_reader.state_q,
                   dut.gamma_reader_busy,
                   dut.u_final_rmsnorm_wrapper.state_q,
                   dut.u_final_rmsnorm_wrapper.u_rmsnorm_core_hls_ip.state_q,
                   dut.scale_seen_q,
                   captured_tile_count);
          $display("debug2: gamma_req_valid=%b gamma_req_done=%b rd_desc_valid=%b rd_burst_active=%b rd_data_valid=%b",
                   dut.gamma_req_valid_q,
                   dut.gamma_req_done_q,
                   rd_desc_valid,
                   rd_burst_active_q,
                   rd_data_valid);
          $display("debug3: act_done=%b gamma_done=%b act_tiles=%0d gamma_beats=%0d row_count=%0d act_valid=%b act_ready=%b gamma_valid=%b gamma_ready=%b",
                   dut.u_final_rmsnorm_wrapper.act_capture_done_q,
                   dut.u_final_rmsnorm_wrapper.gamma_capture_done_q,
                   dut.u_final_rmsnorm_wrapper.act_tile_count_q,
                   dut.u_final_rmsnorm_wrapper.gamma_beat_count_q,
                   dut.u_final_rmsnorm_wrapper.row_count_q,
                   hidden_act_valid,
                   hidden_act_ready,
                   dut.gamma_valid,
                   dut.gamma_ready);
          $finish;
        end
      end

      repeat (2) @(posedge clk);

      if (!saw_scale) begin
        $error("runtime_final_rmsnorm_tail missing scale output for %s", case_name);
        $finish;
      end
      if (captured_scale.data !== output_scale_mem[0]) begin
        $error("runtime_final_rmsnorm_tail scale mismatch for %s", case_name);
        $finish;
      end
      if (captured_scale.tag.layer_id != LAYER_ID_W'(N_LAYERS - 1) ||
          captured_scale.tag.block_id != BLOCK_FINAL_RMSNORM ||
          captured_scale.tag.gemm_mode != GEMM_NONE ||
          captured_scale.tag.tile_id != TILE_ID_W'(0) ||
          captured_scale.tag.token_base != TOKEN_W'(0) ||
          captured_scale.tag.seq_count != COUNT_W'(row_count) ||
          captured_scale.tag.q_head_id != Q_HEAD_ID_W'(0) ||
          captured_scale.tag.kv_head_id != KV_HEAD_ID_W'(0) ||
          captured_scale.tag.elem_count != ELEM_COUNT_W'(SCALE_VECTOR_ELEMS) ||
          !captured_scale.tag.is_last ||
          captured_scale.tag.is_partial) begin
        $error("runtime_final_rmsnorm_tail scale tag mismatch for %s", case_name);
        $display("scale tag layer=%0d block=%0d gemm=%0d tile=%0d seq=%0d elem=%0d last=%0b partial=%0b",
                 captured_scale.tag.layer_id,
                 captured_scale.tag.block_id,
                 captured_scale.tag.gemm_mode,
                 captured_scale.tag.tile_id,
                 captured_scale.tag.seq_count,
                 captured_scale.tag.elem_count,
                 captured_scale.tag.is_last,
                 captured_scale.tag.is_partial);
        $finish;
      end
      if (captured_tile_count != FEATURE_TILE_COUNT) begin
        $error("runtime_final_rmsnorm_tail expected %0d tiles, got %0d for %s",
               FEATURE_TILE_COUNT,
               captured_tile_count,
               case_name);
        $finish;
      end
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        if (captured_layer_id[tile_idx] != LAYER_ID_W'(N_LAYERS - 1) ||
            captured_block_id[tile_idx] != BLOCK_FINAL_RMSNORM ||
            captured_gemm_mode[tile_idx] != GEMM_NONE ||
            captured_tile_id[tile_idx] != TILE_ID_W'(tile_idx) ||
            captured_token_base[tile_idx] != TOKEN_W'(0) ||
            captured_seq_count[tile_idx] != COUNT_W'(row_count) ||
            captured_q_head_id[tile_idx] != Q_HEAD_ID_W'(0) ||
            captured_kv_head_id[tile_idx] != KV_HEAD_ID_W'(0) ||
            captured_elem_count[tile_idx] != ELEM_COUNT_W'(row_count * N_TILE) ||
            (captured_is_last[tile_idx] != (tile_idx == (FEATURE_TILE_COUNT - 1))) ||
            (captured_is_partial[tile_idx] != (row_count != M_TILE))) begin
          $error("runtime_final_rmsnorm_tail act tag mismatch at tile %0d for %s",
                 tile_idx,
                 case_name);
          $display("act tag layer=%0d block=%0d gemm=%0d tile=%0d seq=%0d elem=%0d last=%0b partial=%0b",
                   captured_layer_id[tile_idx],
                   captured_block_id[tile_idx],
                   captured_gemm_mode[tile_idx],
                   captured_tile_id[tile_idx],
                   captured_seq_count[tile_idx],
                   captured_elem_count[tile_idx],
                   captured_is_last[tile_idx],
                   captured_is_partial[tile_idx]);
          $finish;
        end
        mismatch_reported = 1'b0;
        for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
          expected_lane = $signed(y_tiles_mem[tile_idx][(lane * ACT_W) +: ACT_W]);
          actual_lane = $signed(captured_tiles[tile_idx][(lane * ACT_W) +: ACT_W]);
          if ((actual_lane < (expected_lane - 16)) || (actual_lane > (expected_lane + 16))) begin
            if (!mismatch_reported) begin
              $error("runtime_final_rmsnorm_tail output mismatch at tile %0d lane %0d for %s expected=%0d actual=%0d",
                     tile_idx,
                     lane,
                     case_name,
                     expected_lane,
                     actual_lane);
              mismatch_reported = 1'b1;
            end
          end
        end
        if (mismatch_reported) begin
          $display("expected=%h", y_tiles_mem[tile_idx]);
          $display("actual  =%h", captured_tiles[tile_idx]);
          $finish;
        end
      end

      if (busy || norm_scale_valid || norm_act_valid) begin
        $error("runtime_final_rmsnorm_tail should return idle after %s", case_name);
        $finish;
      end

      saw_scale = 1'b0;
      captured_scale = '0;
      captured_tile_count = 0;
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        captured_layer_id[tile_idx] = '0;
        captured_block_id[tile_idx] = BLOCK_NONE;
        captured_gemm_mode[tile_idx] = GEMM_NONE;
        captured_tile_id[tile_idx] = '0;
        captured_token_base[tile_idx] = '0;
        captured_seq_count[tile_idx] = '0;
        captured_q_head_id[tile_idx] = '0;
        captured_kv_head_id[tile_idx] = '0;
        captured_elem_count[tile_idx] = '0;
        captured_is_last[tile_idx] = 1'b0;
        captured_is_partial[tile_idx] = 1'b0;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    launch = 1'b0;
    abort_req = 1'b0;
    rd_data_valid = 1'b0;
    rd_data = '0;
    hidden_scale_valid = 1'b0;
    hidden_scale_bus = '0;
    hidden_act_valid = 1'b0;
    hidden_act_bus = '0;

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    run_case("phase5_prefill_layer0_rmsnorm1", PREFILL_BASE);
    run_case("phase5_decode_layer0_rmsnorm1", DECODE_BASE);

    $display("PASS: tb_runtime_final_rmsnorm_tail");
    $finish;
  end

endmodule
