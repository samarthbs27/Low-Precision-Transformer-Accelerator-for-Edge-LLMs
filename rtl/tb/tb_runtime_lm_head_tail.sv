`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_runtime_lm_head_tail;

  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned LMHEAD_TILE_COUNT = (VOCAB_SIZE + VOCAB_TILE - 1) / VOCAB_TILE;
  localparam int unsigned LMHEAD_GROUPS = VOCAB_TILE / N_TILE;
  localparam int unsigned LMHEAD_GROUP_BYTES = FEATURE_TILE_COUNT * N_TILE;
  localparam int unsigned LMHEAD_TILE_BYTES = LMHEAD_GROUPS * LMHEAD_GROUP_BYTES;
  localparam logic [HBM_ADDR_W-1:0] LMHEAD_BASE_ADDR = 64'h0000_0000_2000_0000;
  localparam logic [TOKEN_W-1:0] TARGET_TOKEN_ID = TOKEN_W'(32'd7);
  localparam string QUANT_BASE_DEFAULT =
    "sim/golden_traces/phase6/rtl/phase6_decode_embedding_quantizer_batch0";
  localparam string QUANT_BASE_LOCAL_FALLBACK =
    "fixtures/phase6_decode_embedding_quantizer_batch0";

  logic clk;
  logic rst_n;
  logic launch;
  logic abort_req;
  logic start;
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
  logic hidden_done_pulse;
  logic lm_head_done_pulse;
  logic token_valid;
  logic token_ready;
  logic [TOKEN_W-1:0] token_id;
  logic signed [ACC_W-1:0] token_logit;
  logic context_valid;
  logic lm_head_busy;
  logic argmax_busy;
  logic busy;

  logic [31:0] quant_meta_mem [0:1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] act_tiles_mem [0:FEATURE_TILE_COUNT-1];

  logic                  rd_burst_active_q;
  logic [15:0]           rd_beat_idx_q;
  logic [15:0]           rd_beats_total_q;
  logic [HBM_ADDR_W-1:0] rd_addr_q;
  int unsigned           rd_desc_count;
  int unsigned           rd_group_desc_count [0:LMHEAD_GROUPS-1];
  int unsigned           sched_start_count;
  string                 quant_base_path;

  runtime_lm_head_tail dut (
    .ap_clk              (clk),
    .ap_rst_n            (rst_n),
    .launch_i            (launch),
    .abort_req_i         (abort_req),
    .start_i             (start),
    .lmhead_base_addr_i  (LMHEAD_BASE_ADDR),
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
    .hidden_done_pulse_i (hidden_done_pulse),
    .lm_head_done_pulse_o(lm_head_done_pulse),
    .token_valid_o       (token_valid),
    .token_ready_i       (token_ready),
    .token_id_o          (token_id),
    .token_logit_o       (token_logit),
    .context_valid_o     (context_valid),
    .lm_head_busy_o      (lm_head_busy),
    .argmax_busy_o       (argmax_busy),
    .busy_o              (busy)
  );

  wire signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] first_logits_flat_w = dut.lmh_logits_bus_q.data;

  always #5 clk = ~clk;

  function automatic bit file_exists(
    input string path
  );
    int fd;
    begin
      fd = $fopen(path, "r");
      if (fd != 0) begin
        $fclose(fd);
        file_exists = 1'b1;
      end else begin
        file_exists = 1'b0;
      end
    end
  endfunction

  function automatic logic signed [WEIGHT_W-1:0] generated_weight_value(
    input int unsigned row_count,
    input logic [TOKEN_W-1:0] candidate_token,
    input int unsigned tile_idx,
    input int unsigned lane_idx
  );
    int unsigned row_idx;
    logic signed [ACT_W-1:0] hidden_lane_val;
    logic signed [WEIGHT_W-1:0] sign_weight;
    begin
      row_idx = (row_count == 0) ? 0 : (row_count - 1);
      hidden_lane_val =
        act_tiles_mem[tile_idx][(((row_idx * N_TILE) + lane_idx) * ACT_W) +: ACT_W];
      if (hidden_lane_val > 0) begin
        sign_weight = WEIGHT_W'(8'sd1);
      end else if (hidden_lane_val < 0) begin
        sign_weight = WEIGHT_W'(-8'sd1);
      end else begin
        sign_weight = '0;
      end

      if (candidate_token == TARGET_TOKEN_ID) begin
        generated_weight_value = sign_weight;
      end else begin
        generated_weight_value = -sign_weight;
      end
    end
  endfunction

  function automatic logic [DMA_BEAT_W-1:0] pack_lmhead_weight_beat(
    input int unsigned row_count,
    input logic [HBM_ADDR_W-1:0] addr,
    input int unsigned beat_idx
  );
    logic [DMA_BEAT_W-1:0] beat_value;
    logic [HBM_ADDR_W-1:0] offset;
    int unsigned tile_idx;
    int unsigned group_idx;
    int unsigned feature_idx;
    logic [TOKEN_W-1:0] candidate_token;
    begin
      beat_value = '0;
      offset = addr - LMHEAD_BASE_ADDR;
      tile_idx = offset / LMHEAD_TILE_BYTES;
      group_idx = (offset % LMHEAD_TILE_BYTES) / LMHEAD_GROUP_BYTES;
      feature_idx = beat_idx;

      for (int lane = 0; lane < N_TILE; lane++) begin
        candidate_token = TOKEN_W'((tile_idx * VOCAB_TILE) + (group_idx * N_TILE) + lane);
        beat_value[(lane * WEIGHT_W) +: WEIGHT_W] =
          generated_weight_value(row_count, candidate_token, feature_idx, lane);
      end
      pack_lmhead_weight_beat = beat_value;
    end
  endfunction

  always @(posedge clk) begin
    if (!rst_n) begin
      rd_burst_active_q <= 1'b0;
      rd_beat_idx_q <= '0;
      rd_beats_total_q <= '0;
      rd_addr_q <= '0;
      rd_data_valid <= 1'b0;
      rd_data <= '0;
      rd_desc_count <= 0;
      for (int group_idx = 0; group_idx < LMHEAD_GROUPS; group_idx++) begin
        rd_group_desc_count[group_idx] <= 0;
      end
      sched_start_count <= 0;
    end else begin
      if (dut.lmh_sched_start) begin
        sched_start_count <= sched_start_count + 1;
      end

      if (rd_desc_valid && rd_desc_ready) begin
        if (rd_desc.tensor_id != TENSOR_LM_HEAD ||
            rd_desc.region != REGION_LM_HEAD ||
            rd_desc.byte_count != LMHEAD_GROUP_BYTES) begin
          $error("runtime_lm_head_tail unexpected LM-head descriptor tensor=%0d region=%0d bytes=%0d",
                 rd_desc.tensor_id, rd_desc.region, rd_desc.byte_count);
          $finish;
        end
        rd_burst_active_q <= 1'b1;
        rd_beat_idx_q <= '0;
        rd_beats_total_q <= (rd_desc.byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
        rd_addr_q <= rd_desc.addr;
        rd_data_valid <= 1'b0;
        rd_desc_count <= rd_desc_count + 1;
        rd_group_desc_count[((rd_desc.addr - LMHEAD_BASE_ADDR) % LMHEAD_TILE_BYTES) / LMHEAD_GROUP_BYTES] <=
          rd_group_desc_count[((rd_desc.addr - LMHEAD_BASE_ADDR) % LMHEAD_TILE_BYTES) / LMHEAD_GROUP_BYTES] + 1;
      end

      if (rd_burst_active_q && !rd_data_valid) begin
        rd_data_valid <= 1'b1;
        rd_data <= pack_lmhead_weight_beat(quant_meta_mem[0], rd_addr_q, rd_beat_idx_q);
      end else if (rd_data_valid && rd_data_ready) begin
        if (rd_beat_idx_q + 1 >= rd_beats_total_q) begin
          rd_burst_active_q <= 1'b0;
          rd_data_valid <= 1'b0;
          rd_beat_idx_q <= '0;
        end else begin
          rd_beat_idx_q <= rd_beat_idx_q + 1'b1;
          rd_data <= pack_lmhead_weight_beat(quant_meta_mem[0], rd_addr_q, rd_beat_idx_q + 1'b1);
        end
      end
    end
  end

  assign rd_desc_ready = 1'b1;

  task automatic drive_scale;
    begin
      hidden_scale_bus = '0;
      hidden_scale_bus.data = scale_mem[0];
      hidden_scale_bus.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
      hidden_scale_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
      hidden_scale_bus.tag.gemm_mode = GEMM_NONE;
      hidden_scale_bus.tag.tile_id = '0;
      hidden_scale_bus.tag.token_base = POS_W'(quant_meta_mem[1]);
      hidden_scale_bus.tag.seq_count = COUNT_W'(quant_meta_mem[0]);
      hidden_scale_bus.tag.q_head_id = '0;
      hidden_scale_bus.tag.kv_head_id = '0;
      hidden_scale_bus.tag.elem_count = ELEM_COUNT_W'(D_MODEL);
      hidden_scale_bus.tag.is_last = 1'b1;
      hidden_scale_bus.tag.is_partial = 1'b0;

      hidden_scale_valid = 1'b1;
      while (!hidden_scale_ready) begin
        @(negedge clk);
      end
      @(negedge clk);
      hidden_scale_valid = 1'b0;
    end
  endtask

  task automatic drive_act_tile(
    input int unsigned tile_idx
  );
    begin
      hidden_act_bus = '0;
      hidden_act_bus.data = act_tiles_mem[tile_idx];
      hidden_act_bus.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
      hidden_act_bus.tag.block_id = BLOCK_FINAL_RMSNORM;
      hidden_act_bus.tag.gemm_mode = GEMM_NONE;
      hidden_act_bus.tag.tile_id = TILE_ID_W'(tile_idx);
      hidden_act_bus.tag.token_base = POS_W'(quant_meta_mem[1]);
      hidden_act_bus.tag.seq_count = COUNT_W'(quant_meta_mem[0]);
      hidden_act_bus.tag.q_head_id = '0;
      hidden_act_bus.tag.kv_head_id = '0;
      hidden_act_bus.tag.elem_count = ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE);
      hidden_act_bus.tag.is_last = (tile_idx == FEATURE_TILE_COUNT - 1);
      hidden_act_bus.tag.is_partial = (quant_meta_mem[0] != M_TILE);

      hidden_act_valid = 1'b1;
      while (!hidden_act_ready) begin
        @(negedge clk);
      end
      @(negedge clk);
      hidden_act_valid = 1'b0;
    end
  endtask

  initial begin
    int timeout_cycles;
    int signed lane_value;
    int plusarg_seen;
    logic [TOKEN_W-1:0] first_tile_best_token;
    logic signed [ACC_W-1:0] first_tile_best_logit;
    string quant_meta_path;
    string quant_scale_path;
    string quant_act_path;

    quant_base_path = QUANT_BASE_DEFAULT;
    plusarg_seen = $value$plusargs("QUANT_BASE=%s", quant_base_path);
    if (!plusarg_seen) begin
      quant_meta_path = {quant_base_path, ".meta.memh"};
      if (!file_exists(quant_meta_path)) begin
        quant_base_path = QUANT_BASE_LOCAL_FALLBACK;
      end
    end
    quant_meta_path = {quant_base_path, ".meta.memh"};
    quant_scale_path = {quant_base_path, ".scale_packed.memh"};
    quant_act_path = {quant_base_path, ".act_tiles_expected_packed.memh"};

    if (!file_exists(quant_meta_path) ||
        !file_exists(quant_scale_path) ||
        !file_exists(quant_act_path)) begin
      $fatal(1,
             "tb_runtime_lm_head_tail missing fixture files under QUANT_BASE='%0s'. Run from the repo root, stage fixtures under fixtures/, or pass +QUANT_BASE=<fixture_base>.",
             quant_base_path);
    end

    for (int idx = 0; idx < 2; idx++) begin
      quant_meta_mem[idx] = 'x;
    end
    scale_mem[0] = 'x;
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      act_tiles_mem[tile_idx] = 'x;
    end

    $display("tb_runtime_lm_head_tail using QUANT_BASE=%0s", quant_base_path);
    $readmemh(quant_meta_path, quant_meta_mem);
    $readmemh(quant_scale_path, scale_mem);
    $readmemh(quant_act_path, act_tiles_mem);

    if ($isunknown(quant_meta_mem[0]) || $isunknown(quant_meta_mem[1]) ||
        $isunknown(scale_mem[0]) || $isunknown(act_tiles_mem[0])) begin
      $fatal(1,
             "tb_runtime_lm_head_tail fixture load failed for QUANT_BASE='%0s'. Check the working directory or pass +QUANT_BASE=<fixture_base>.",
             quant_base_path);
    end
    if ((quant_meta_mem[0] == 0) || (quant_meta_mem[0] > M_TILE)) begin
      $fatal(1,
             "tb_runtime_lm_head_tail invalid row_count=%0d from fixture base '%0s'.",
             quant_meta_mem[0],
             quant_base_path);
    end

    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      for (int lane = 0; lane < N_TILE; lane++) begin
        lane_value = ((tile_idx * 3) + lane) % 7 - 3;
        if (lane_value == 0) begin
          lane_value = (lane[0]) ? 2 : -2;
        end
        act_tiles_mem[tile_idx][((((quant_meta_mem[0] - 1) * N_TILE) + lane) * ACT_W) +: ACT_W] =
          ACT_W'(lane_value);
      end
    end

    clk = 1'b0;
    rst_n = 1'b0;
    launch = 1'b0;
    abort_req = 1'b0;
    start = 1'b0;
    hidden_scale_valid = 1'b0;
    hidden_scale_bus = '0;
    hidden_act_valid = 1'b0;
    hidden_act_bus = '0;
    hidden_done_pulse = 1'b0;
    token_ready = 1'b1;

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    launch = 1'b1;
    @(negedge clk);
    launch = 1'b0;

    fork
      begin
        drive_scale();
        for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
          drive_act_tile(tile_idx);
        end
        hidden_done_pulse = 1'b1;
        @(negedge clk);
        hidden_done_pulse = 1'b0;
      end
      begin
        repeat (10) @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
      end
    join

    timeout_cycles = 0;
    while (!dut.lmh_logits_valid_q) begin
      @(negedge clk);
      timeout_cycles++;
      if ((timeout_cycles % 256) == 0) begin
        $display("tb_runtime_lm_head_tail progress cycles=%0d context=%b start_pending=%b sched_start=%b sched_done=%b logits_valid=%b logits_ready=%b req_valid=%b reader_busy=%b group_active=%b feature=%0d tile=%0d group=%0d",
                 timeout_cycles,
                 context_valid,
                 dut.start_pending_q,
                 dut.lmh_sched_start,
                 dut.lmh_sched_done_q,
                 dut.lmh_logits_valid_q,
                 dut.lmh_logits_ready,
                 dut.lmhead_req_valid_q,
                 dut.lmhead_reader_busy,
                 dut.lmhead_group_active_q,
                 dut.feature_idx_q,
                 dut.active_vocab_tile_q,
                 dut.active_group_q);
      end
      if (timeout_cycles > 2048) begin
        $error("runtime_lm_head_tail timed out waiting for first logits tile");
        $display("debug: context=%b start_pending=%b sched_start=%b sched_done=%b logits_valid=%b logits_ready=%b lmh_busy=%b argmax_busy=%b",
                 context_valid,
                 dut.start_pending_q,
                 dut.lmh_sched_start,
                 dut.lmh_sched_done_q,
                 dut.lmh_logits_valid_q,
                 dut.lmh_logits_ready,
                 dut.lm_head_busy_o,
                 dut.argmax_busy_o);
        $display("debug2: req_valid=%b reader_busy=%b group_active=%b tile_complete=%b logits_pending=%b tile=%0d group=%0d feature=%0d desc=%0d sched=%0d",
                 dut.lmhead_req_valid_q,
                 dut.lmhead_reader_busy,
                 dut.lmhead_group_active_q,
                 dut.tile_complete_q,
                 dut.logits_pending_q,
                 dut.active_vocab_tile_q,
                 dut.active_group_q,
                 dut.feature_idx_q,
                 rd_desc_count,
                 sched_start_count);
        $display("debug3: gemm_busy=%b gemm_acc_valid=%b token_valid=%b token_id=%0d token_logit=%0d",
                 dut.gemm_busy,
                 dut.gemm_acc_valid,
                 token_valid,
                 token_id,
                 token_logit);
        $finish;
      end
    end

    repeat (2) @(posedge clk);

    if (!context_valid) begin
      $error("runtime_lm_head_tail should expose captured context before reset");
      $finish;
    end
    if (sched_start_count != 1) begin
      $error("runtime_lm_head_tail expected exactly one LM-head tile launch before first logits, got %0d",
             sched_start_count);
      $finish;
    end
    if (rd_desc_count != LMHEAD_GROUPS) begin
      $error("runtime_lm_head_tail expected %0d LM-head group descriptors for the first tile, got %0d",
             LMHEAD_GROUPS,
             rd_desc_count);
      $display("group counts: g0=%0d g1=%0d g2=%0d g3=%0d token=%0d sched=%0d active_tile=%0d active_group=%0d",
               rd_group_desc_count[0],
               rd_group_desc_count[1],
               rd_group_desc_count[2],
               rd_group_desc_count[3],
               token_id,
               sched_start_count,
               dut.active_vocab_tile_q,
               dut.active_group_q);
      $finish;
    end
    if (rd_group_desc_count[0] != 1 || rd_group_desc_count[1] != 1 ||
        rd_group_desc_count[2] != 1 || rd_group_desc_count[3] != 1) begin
      $error("runtime_lm_head_tail expected one descriptor per LM-head subgroup on the first tile");
      $finish;
    end
    if (dut.lmh_logits_bus_q.tag.tile_id != TILE_ID_W'(0)) begin
      $error("runtime_lm_head_tail expected first logits tile_id 0, got %0d",
             dut.lmh_logits_bus_q.tag.tile_id);
      $finish;
    end

    first_tile_best_token = '0;
    first_tile_best_logit = first_logits_flat_w[0 +: ACC_W];
    for (int lane = 1; lane < VOCAB_TILE; lane++) begin
      if (($signed(first_logits_flat_w[(lane * ACC_W) +: ACC_W]) > first_tile_best_logit) ||
          (($signed(first_logits_flat_w[(lane * ACC_W) +: ACC_W]) == first_tile_best_logit) &&
           (TOKEN_W'(lane) < first_tile_best_token))) begin
        first_tile_best_logit = $signed(first_logits_flat_w[(lane * ACC_W) +: ACC_W]);
        first_tile_best_token = TOKEN_W'(lane);
      end
    end

    if (first_tile_best_token != TARGET_TOKEN_ID) begin
      $error("runtime_lm_head_tail first-tile winner mismatch: expected %0d got %0d",
             TARGET_TOKEN_ID,
             first_tile_best_token);
      $finish;
    end
    if (first_tile_best_logit <= 0) begin
      $error("runtime_lm_head_tail expected positive first-tile winning logit, got %0d",
             first_tile_best_logit);
      $finish;
    end
    if ($signed(first_logits_flat_w[(TARGET_TOKEN_ID * ACC_W) +: ACC_W]) != first_tile_best_logit) begin
      $error("runtime_lm_head_tail expected target-token logit to match the first-tile best logit");
      $finish;
    end

    $display("PASS: tb_runtime_lm_head_tail sched=%0d desc=%0d first_tile_winner=%0d",
             sched_start_count,
             rd_desc_count,
             first_tile_best_token);
    $finish;
  end

endmodule
