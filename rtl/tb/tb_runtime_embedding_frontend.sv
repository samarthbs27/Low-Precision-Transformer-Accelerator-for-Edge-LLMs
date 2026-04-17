`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_runtime_embedding_frontend;

  localparam int unsigned EMBED_ROW_BYTES = D_MODEL * 2;
  localparam int unsigned EMBED_ROW_BEATS = EMBED_ROW_BYTES / DMA_BEAT_BYTES;
  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam string LOOKUP_BASE =
    "sim/golden_traces/phase6/rtl/phase6_decode_embedding_lookup_row0";
  localparam string QUANT_BASE =
    "sim/golden_traces/phase6/rtl/phase6_decode_embedding_quantizer_batch0";
  localparam logic [HBM_ADDR_W-1:0] EMBEDDING_BASE_ADDR = 64'h0000_0000_1000_0000;
  localparam logic [HBM_ADDR_W-1:0] SCALE_META_BASE_ADDR = 64'h0000_0000_0400_0000;

  typedef enum logic [1:0] {
    RD_NONE  = 2'd0,
    RD_SCALE = 2'd1,
    RD_EMBED = 2'd2
  } rd_kind_e;

  logic                               clk;
  logic                               rst_n;
  logic                               launch;
  logic                               token_valid;
  logic                               token_ready;
  token_bus_t                         token_bus;
  logic                               rd_desc_valid;
  logic                               rd_desc_ready;
  dma_desc_t                          rd_desc;
  logic                               rd_data_valid;
  logic                               rd_data_ready;
  logic [DMA_BEAT_W-1:0]              rd_data;
  logic                               scale_valid;
  logic                               scale_ready;
  scale_bus_t                         scale_bus;
  logic                               act_valid;
  logic                               act_ready;
  act_bus_t                           act_bus;
  logic                               busy;
  logic                               done_pulse;

  logic [31:0]                        lookup_meta_mem [0:3];
  logic [DMA_BEAT_W-1:0]              row_beats_mem [0:EMBED_ROW_BEATS-1];
  logic [31:0]                        quant_meta_mem [0:1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] act_tiles_mem [0:FEATURE_TILE_COUNT-1];

  rd_kind_e                           rd_kind_q;
  logic                               rd_burst_active_q;
  logic [15:0]                        rd_beat_idx_q;
  logic [15:0]                        rd_beats_total_q;
  int unsigned                        captured_desc_count;
  tensor_id_e                         captured_desc_tensor_id [0:1];
  logic [HBM_ADDR_W-1:0]              captured_desc_addr [0:1];
  logic [31:0]                        captured_desc_byte_count [0:1];
  logic [TILE_ID_W-1:0]               captured_desc_tile_id [0:1];

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

  runtime_embedding_frontend dut (
    .ap_clk               (clk),
    .ap_rst_n             (rst_n),
    .launch_i             (launch),
    .embedding_base_addr_i(EMBEDDING_BASE_ADDR),
    .scale_meta_base_addr_i(SCALE_META_BASE_ADDR),
    .token_valid_i        (token_valid),
    .token_ready_o        (token_ready),
    .token_i              (token_bus),
    .rd_desc_valid_o      (rd_desc_valid),
    .rd_desc_ready_i      (rd_desc_ready),
    .rd_desc_o            (rd_desc),
    .rd_data_valid_i      (rd_data_valid),
    .rd_data_ready_o      (rd_data_ready),
    .rd_data_i            (rd_data),
    .scale_valid_o        (scale_valid),
    .scale_ready_i        (scale_ready),
    .scale_o              (scale_bus),
    .act_valid_o          (act_valid),
    .act_ready_i          (act_ready),
    .act_o                (act_bus),
    .busy_o               (busy),
    .done_pulse_o         (done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_burst_active_q <= 1'b0;
      rd_kind_q <= RD_NONE;
      rd_beat_idx_q <= '0;
      rd_beats_total_q <= '0;
      rd_data_valid <= 1'b0;
      rd_data <= '0;
      captured_desc_count <= 0;
      captured_desc_tensor_id[0] <= TENSOR_NONE;
      captured_desc_tensor_id[1] <= TENSOR_NONE;
      captured_desc_addr[0] <= '0;
      captured_desc_addr[1] <= '0;
      captured_desc_byte_count[0] <= '0;
      captured_desc_byte_count[1] <= '0;
      captured_desc_tile_id[0] <= '0;
      captured_desc_tile_id[1] <= '0;
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
      if (scale_valid && scale_ready) begin
        saw_scale <= 1'b1;
        captured_scale_bus <= scale_bus;
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

      if (rd_desc_valid && rd_desc_ready) begin
        if (captured_desc_count < 2) begin
          captured_desc_tensor_id[captured_desc_count] <= rd_desc.tensor_id;
          captured_desc_addr[captured_desc_count] <= rd_desc.addr;
          captured_desc_byte_count[captured_desc_count] <= rd_desc.byte_count;
          captured_desc_tile_id[captured_desc_count] <= rd_desc.tile_id;
        end
        captured_desc_count <= captured_desc_count + 1;
        rd_burst_active_q <= 1'b1;
        rd_beat_idx_q <= '0;
        rd_beats_total_q <= (rd_desc.byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
        if (rd_desc.tensor_id == TENSOR_SCALE_META) begin
          rd_kind_q <= RD_SCALE;
        end else if (rd_desc.tensor_id == TENSOR_EMBED) begin
          rd_kind_q <= RD_EMBED;
        end else begin
          rd_kind_q <= RD_NONE;
        end
        rd_data_valid <= 1'b0;
      end else if (rd_burst_active_q && !rd_data_valid) begin
        rd_data_valid <= 1'b1;
        unique case (rd_kind_q)
          RD_SCALE: rd_data <= (rd_beat_idx_q == 0) ?
                               scale_mem[0][0 +: DMA_BEAT_W] :
                               scale_mem[0][DMA_BEAT_W +: DMA_BEAT_W];
          RD_EMBED: rd_data <= row_beats_mem[rd_beat_idx_q];
          default:  rd_data <= '0;
        endcase
      end else if (rd_data_valid && rd_data_ready) begin
        if (rd_beat_idx_q + 1 >= rd_beats_total_q) begin
          rd_data_valid <= 1'b0;
          rd_burst_active_q <= 1'b0;
          rd_kind_q <= RD_NONE;
          rd_beat_idx_q <= '0;
        end else begin
          rd_beat_idx_q <= rd_beat_idx_q + 1'b1;
          unique case (rd_kind_q)
            RD_SCALE: rd_data <= scale_mem[0][DMA_BEAT_W +: DMA_BEAT_W];
            RD_EMBED: rd_data <= row_beats_mem[rd_beat_idx_q + 1'b1];
            default:  rd_data <= '0;
          endcase
        end
      end
    end
  end

  assign rd_desc_ready = 1'b1;
  assign scale_ready = 1'b1;
  assign act_ready = 1'b1;

  initial begin
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] actual_tile;
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] expected_tile;
    int mismatch_count;
    int timeout_cycles;

    $readmemh({LOOKUP_BASE, ".meta.memh"}, lookup_meta_mem);
    $readmemh({LOOKUP_BASE, ".row_beats_packed.memh"}, row_beats_mem);
    $readmemh({QUANT_BASE, ".meta.memh"}, quant_meta_mem);
    $readmemh({QUANT_BASE, ".scale_packed.memh"}, scale_mem);
    $readmemh({QUANT_BASE, ".act_tiles_expected_packed.memh"}, act_tiles_mem);

    clk = 1'b0;
    rst_n = 1'b0;
    launch = 1'b0;
    token_valid = 1'b0;
    token_bus = '0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    token_bus.token_id = lookup_meta_mem[0];
    token_bus.token_count = lookup_meta_mem[1];
    token_bus.tag.layer_id = '0;
    token_bus.tag.block_id = BLOCK_EMBED;
    token_bus.tag.gemm_mode = GEMM_NONE;
    token_bus.tag.tile_id = TILE_ID_W'(lookup_meta_mem[2]);
    token_bus.tag.token_base = POS_W'(lookup_meta_mem[2]);
    token_bus.tag.seq_count = COUNT_W'(lookup_meta_mem[1]);
    token_bus.tag.q_head_id = '0;
    token_bus.tag.kv_head_id = '0;
    token_bus.tag.elem_count = 16'd1;
    token_bus.tag.is_last = lookup_meta_mem[3][0];
    token_bus.tag.is_partial = 1'b1;

    @(negedge clk);
    launch <= 1'b1;
    token_valid <= 1'b1;
    @(negedge clk);
    launch <= 1'b0;

    while (!token_ready) begin
      @(negedge clk);
    end
    token_valid <= 1'b0;

    timeout_cycles = 0;
    while (!done_pulse) begin
      @(posedge clk);
      timeout_cycles++;
      if (timeout_cycles > 5000) begin
        $error("runtime_embedding_frontend timed out waiting for done_pulse");
        $finish;
      end
    end

    @(posedge clk);
    if (busy || act_valid || scale_valid || rd_data_valid) begin
      $error("runtime_embedding_frontend did not return idle after completion");
      $finish;
    end

    if (captured_desc_count != 2) begin
      $error("runtime_embedding_frontend expected 2 DMA descriptors, got %0d", captured_desc_count);
      $finish;
    end
    if ((captured_desc_tensor_id[0] != TENSOR_SCALE_META) ||
        (captured_desc_addr[0] != SCALE_META_BASE_ADDR) ||
        (captured_desc_byte_count[0] != (SCALE_VECTOR_ELEMS * (SCALE_W / 8)))) begin
      $error("runtime_embedding_frontend scale descriptor mismatch");
      $finish;
    end
    if ((captured_desc_tensor_id[1] != TENSOR_EMBED) ||
        (captured_desc_addr[1] != (EMBEDDING_BASE_ADDR + (HBM_ADDR_W'(lookup_meta_mem[0]) * EMBED_ROW_BYTES))) ||
        (captured_desc_byte_count[1] != EMBED_ROW_BYTES) ||
        (captured_desc_tile_id[1] != TILE_ID_W'(lookup_meta_mem[2]))) begin
      $error("runtime_embedding_frontend embedding descriptor mismatch");
      $finish;
    end

    if (!saw_scale ||
        (captured_scale_bus.tag.block_id != BLOCK_EMBED) ||
        (captured_scale_bus.tag.gemm_mode != GEMM_NONE) ||
        (captured_scale_bus.tag.token_base != POS_W'(quant_meta_mem[1])) ||
        (captured_scale_bus.tag.seq_count != COUNT_W'(quant_meta_mem[0])) ||
        (captured_scale_bus.data != scale_mem[0])) begin
      $error("runtime_embedding_frontend scale output mismatch");
      $finish;
    end

    if (captured_tile_count != FEATURE_TILE_COUNT) begin
      $error("runtime_embedding_frontend expected %0d output tiles, got %0d",
             FEATURE_TILE_COUNT,
             captured_tile_count);
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
                "runtime_embedding_frontend mismatch tile %0d lane %0d actual=%0d expected=%0d",
                tile_idx,
                lane_idx,
                actual_tile[lane_idx],
                expected_tile[lane_idx]
              );
            end
            mismatch_count++;
          end
        end
        $error("runtime_embedding_frontend tile data mismatch at tile %0d", tile_idx);
        $finish;
      end

      if ((captured_act_block_id[tile_idx] != BLOCK_EMBED) ||
          (captured_act_gemm_mode[tile_idx] != GEMM_NONE) ||
          (captured_act_tile_id[tile_idx] != TILE_ID_W'(tile_idx)) ||
          (captured_act_token_base[tile_idx] != POS_W'(quant_meta_mem[1])) ||
          (captured_act_seq_count[tile_idx] != COUNT_W'(quant_meta_mem[0])) ||
          (captured_act_elem_count[tile_idx] != ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE)) ||
          (captured_act_is_partial[tile_idx] != (quant_meta_mem[0] != M_TILE)) ||
          (captured_act_is_last[tile_idx] != (tile_idx == FEATURE_TILE_COUNT - 1))) begin
        $error("runtime_embedding_frontend tile tag mismatch at tile %0d", tile_idx);
        $finish;
      end
    end

    $display("PASS: tb_runtime_embedding_frontend");
    $finish;
  end

endmodule
