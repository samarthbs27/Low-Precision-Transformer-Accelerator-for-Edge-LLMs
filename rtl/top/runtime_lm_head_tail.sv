import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module runtime_lm_head_tail (
  input  logic                    ap_clk,
  input  logic                    ap_rst_n,
  input  logic                    launch_i,
  input  logic                    abort_req_i,
  input  logic                    start_i,
  input  logic [HBM_ADDR_W-1:0]   lmhead_base_addr_i,
  output logic                    rd_desc_valid_o,
  input  logic                    rd_desc_ready_i,
  output dma_desc_t               rd_desc_o,
  input  logic                    rd_data_valid_i,
  output logic                    rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0]   rd_data_i,
  input  logic                    hidden_scale_valid_i,
  output logic                    hidden_scale_ready_o,
  input  scale_bus_t              hidden_scale_i,
  input  logic                    hidden_act_valid_i,
  output logic                    hidden_act_ready_o,
  input  act_bus_t                hidden_act_i,
  input  logic                    hidden_done_pulse_i,
  output logic                    lm_head_done_pulse_o,
  output logic                    token_valid_o,
  input  logic                    token_ready_i,
  output logic [TOKEN_W-1:0]      token_id_o,
  output logic signed [ACC_W-1:0] token_logit_o,
  output logic                    context_valid_o,
  output logic                    lm_head_busy_o,
  output logic                    argmax_busy_o,
  output logic                    busy_o
);

  localparam int unsigned FEATURE_TILE_COUNT   = D_MODEL / N_TILE;
  localparam int unsigned LMHEAD_TILE_COUNT    = (VOCAB_SIZE + VOCAB_TILE - 1) / VOCAB_TILE;
  localparam int unsigned LMHEAD_GROUPS        = VOCAB_TILE / N_TILE;
  localparam int unsigned LMHEAD_GROUP_BYTES   = FEATURE_TILE_COUNT * N_TILE;
  localparam int unsigned LMHEAD_TILE_BYTES    = LMHEAD_GROUPS * LMHEAD_GROUP_BYTES;
  localparam int unsigned LAST_ROW_COUNT_W     = (M_TILE > 1) ? $clog2(M_TILE) : 1;
  localparam logic [$clog2(LMHEAD_GROUPS)-1:0] LMHEAD_LAST_GROUP =
    $clog2(LMHEAD_GROUPS)'(LMHEAD_GROUPS - 1);

  logic                               helper_rst_n;
  logic                               scale_seen_q;
  logic [FEATURE_TILE_COUNT-1:0]      tile_seen_q;
  logic                               context_ready_q;
  logic                               start_pending_q;
  logic [LAST_ROW_COUNT_W-1:0]        last_row_idx_q;
  logic [COUNT_W-1:0]                 row_count_q;
  scale_bus_t                         hidden_scale_q;
  act_bus_t                           hidden_seed_q;
  logic signed [(N_TILE * ACT_W)-1:0] hidden_last_row_tile_q [0:FEATURE_TILE_COUNT-1];

  logic                               lmh_hidden_ready;
  logic                               lmh_hidden_scale_ready;
  act_bus_t                           lmh_hidden_bus;
  scale_bus_t                         lmh_hidden_scale_bus;
  logic                               lmh_sched_start;
  logic [TILE_ID_W-1:0]               lmh_vocab_tile_idx;
  logic                               lmh_sched_done_q;
  logic                               lmh_logits_valid_q;
  logic                               lmh_logits_ready;
  acc_bus_t                           lmh_logits_bus_d;
  acc_bus_t                           lmh_logits_bus_q;
  logic                               lmh_busy;
  logic                               lmh_done_pulse;
  logic                               lmh_argmax_valid;
  logic                               lmh_argmax_ready;
  acc_bus_t                           lmh_argmax_bus;
  logic                               argmax_start_w;
  logic                               argmax_busy;
  logic                               argmax_token_valid;
  logic                               argmax_logits_ready;
  logic [TOKEN_W-1:0]                 argmax_token_id;
  logic signed [ACC_W-1:0]            argmax_token_logit;
  logic                               core_start_w;

  logic                               lmhead_req_valid_q;
  logic [HBM_ADDR_W-1:0]              lmhead_req_addr_q;
  logic                               lmhead_reader_busy;
  logic                               lmhead_reader_done_pulse;
  logic                               lmhead_wt_valid;
  logic                               lmhead_wt_ready;
  wt_bus_t                            lmhead_wt_bus;
  logic                               lmhead_group_active_q;
  logic [TILE_ID_W-1:0]               active_vocab_tile_q;
  logic [$clog2(LMHEAD_GROUPS)-1:0]   active_group_q;
  logic [COUNT_W-1:0]                 feature_idx_q;
  logic                               tile_complete_q;
  logic                               logits_pending_q;
  logic signed [ACC_W-1:0]            tile_logits_q [0:VOCAB_TILE-1];

  act_bus_t                           gemm_hidden_bus_d;
  wire signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] hidden_act_flat_w;
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] gemm_hidden_flat_d;
  logic                               gemm_operands_ready;
  logic                               gemm_acc_valid;
  acc_bus_t                           gemm_acc_bus;
  wire signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] gemm_acc_flat_w;
  logic signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] lmh_logits_flat_d;
  logic                               gemm_busy;
  logic                               group_result_pending_q;
  logic                               gemm_last_operand_w;

  function automatic logic [HBM_ADDR_W-1:0] lmhead_group_offset(
    input logic [TILE_ID_W-1:0] tile_idx,
    input logic [$clog2(LMHEAD_GROUPS)-1:0] group_idx
  );
    logic [HBM_ADDR_W-1:0] tile_offset;
    logic [HBM_ADDR_W-1:0] group_offset;
    begin
      tile_offset = HBM_ADDR_W'(tile_idx) * HBM_ADDR_W'(LMHEAD_TILE_BYTES);
      group_offset = HBM_ADDR_W'(group_idx) * HBM_ADDR_W'(LMHEAD_GROUP_BYTES);
      lmhead_group_offset = tile_offset + group_offset;
    end
  endfunction

  assign helper_rst_n = ap_rst_n && !abort_req_i;
  assign hidden_act_flat_w = hidden_act_i.data;
  assign gemm_acc_flat_w = gemm_acc_bus.data;
  assign hidden_scale_ready_o = !scale_seen_q;
  assign hidden_act_ready_o = !context_ready_q;
  assign context_valid_o = context_ready_q;
  assign gemm_last_operand_w = lmhead_group_active_q && lmhead_wt_valid &&
                               gemm_operands_ready &&
                               (feature_idx_q == COUNT_W'(FEATURE_TILE_COUNT - 1));
  assign core_start_w = start_pending_q && context_ready_q &&
                        lmh_hidden_ready && lmh_hidden_scale_ready &&
                        !lmhead_req_valid_q && !lmhead_group_active_q &&
                        !tile_complete_q && !logits_pending_q &&
                        !lmh_logits_valid_q;
  assign argmax_start_w = core_start_w;
  assign lmh_argmax_ready = argmax_logits_ready;
  assign lm_head_done_pulse_o = lmh_done_pulse;
  assign token_valid_o = argmax_token_valid;
  assign token_id_o = argmax_token_id;
  assign token_logit_o = argmax_token_logit;
  assign lm_head_busy_o = start_pending_q || lmh_busy || lmhead_req_valid_q ||
                          lmhead_reader_busy || lmhead_group_active_q ||
                          tile_complete_q || logits_pending_q ||
                          lmh_logits_valid_q || gemm_busy;
  assign argmax_busy_o = argmax_busy;
  assign busy_o = !context_ready_q || lm_head_busy_o || argmax_busy;

  always_comb begin
    gemm_hidden_bus_d = '0;
    gemm_hidden_bus_d.tag = hidden_scale_q.tag;
    gemm_hidden_bus_d.tag.block_id = BLOCK_LM_HEAD;
    gemm_hidden_bus_d.tag.gemm_mode = GEMM_LM_HEAD;
    gemm_hidden_bus_d.tag.tile_id = active_vocab_tile_q;
    gemm_hidden_bus_d.tag.token_base = hidden_scale_q.tag.token_base +
                                       TOKEN_W'(row_count_q ? (row_count_q - 1'b1) : '0);
    gemm_hidden_bus_d.tag.seq_count = COUNT_W'(1);
    gemm_hidden_bus_d.tag.q_head_id = '0;
    gemm_hidden_bus_d.tag.kv_head_id = '0;
    gemm_hidden_bus_d.tag.elem_count = ELEM_COUNT_W'(N_TILE);
    gemm_hidden_bus_d.tag.is_partial = 1'b1;
    gemm_hidden_bus_d.tag.is_last = (feature_idx_q == COUNT_W'(FEATURE_TILE_COUNT - 1));

    gemm_hidden_flat_d = '0;
    for (int lane = 0; lane < N_TILE; lane++) begin
      if (feature_idx_q < FEATURE_TILE_COUNT) begin
        gemm_hidden_flat_d[(lane * ACT_W) +: ACT_W] =
          hidden_last_row_tile_q[feature_idx_q][(lane * ACT_W) +: ACT_W];
      end
    end
    gemm_hidden_bus_d.data = gemm_hidden_flat_d;

    lmh_logits_bus_d = '0;
    lmh_logits_bus_d.tag = hidden_scale_q.tag;
    lmh_logits_bus_d.tag.block_id = BLOCK_LM_HEAD;
    lmh_logits_bus_d.tag.gemm_mode = GEMM_LM_HEAD;
    lmh_logits_bus_d.tag.tile_id = active_vocab_tile_q;
    lmh_logits_bus_d.tag.token_base = hidden_scale_q.tag.token_base +
                                      TOKEN_W'(row_count_q ? (row_count_q - 1'b1) : '0);
    lmh_logits_bus_d.tag.seq_count = COUNT_W'(1);
    lmh_logits_bus_d.tag.q_head_id = '0;
    lmh_logits_bus_d.tag.kv_head_id = '0;
    lmh_logits_bus_d.tag.elem_count = ELEM_COUNT_W'(VOCAB_TILE);
    lmh_logits_bus_d.tag.is_partial = 1'b0;
    lmh_logits_bus_d.tag.is_last = (active_vocab_tile_q == TILE_ID_W'(LMHEAD_TILE_COUNT - 1));

    lmh_logits_flat_d = '0;
    for (int lane = 0; lane < VOCAB_TILE; lane++) begin
      lmh_logits_flat_d[(lane * ACC_W) +: ACC_W] = tile_logits_q[lane];
    end
    lmh_logits_bus_d.data = lmh_logits_flat_d;
  end

  embedding_lmhead_dma_reader u_lmhead_reader (
    .ap_clk            (ap_clk),
    .ap_rst_n          (helper_rst_n),
    .req_valid_i       (lmhead_req_valid_q),
    .req_ready_o       (),
    .base_addr_i       (lmhead_req_addr_q),
    .byte_count_i      (32'(LMHEAD_GROUP_BYTES)),
    .layer_id_i        (LAYER_ID_W'(N_LAYERS - 1)),
    .tensor_id_i       (tensor_id_e'(TENSOR_LM_HEAD)),
    .tile_id_i         (active_vocab_tile_q),
    .busy_o            (lmhead_reader_busy),
    .done_pulse_o      (lmhead_reader_done_pulse),
    .rd_desc_valid_o   (rd_desc_valid_o),
    .rd_desc_ready_i   (rd_desc_ready_i),
    .rd_desc_o         (rd_desc_o),
    .rd_data_valid_i   (rd_data_valid_i),
    .rd_data_i         (rd_data_i),
    .rd_data_ready_o   (rd_data_ready_o),
    .embed_row_valid_o (),
    .embed_row_ready_i (1'b0),
    .embed_row_o       (),
    .embed_row_last_o  (),
    .gamma_valid_o     (),
    .gamma_ready_i     (1'b0),
    .gamma_o           (),
    .gamma_last_o      (),
    .lmhead_wt_valid_o (lmhead_wt_valid),
    .lmhead_wt_ready_i (lmhead_wt_ready),
    .lmhead_wt_o       (lmhead_wt_bus),
    .scale_valid_o     (),
    .scale_ready_i     (1'b0),
    .scale_tensor_id_o (),
    .scale_o           ()
  );

  shared_gemm_engine u_lmhead_gemm (
    .ap_clk          (ap_clk),
    .ap_rst_n        (helper_rst_n),
    .gemm_mode_i     (gemm_mode_e'(GEMM_LM_HEAD)),
    .clear_acc_i     (lmhead_group_active_q && (feature_idx_q == '0)),
    .mac_valid_i     (1'b1),
    .emit_acc_i      (gemm_last_operand_w),
    .operands_valid_i(lmhead_group_active_q && lmhead_wt_valid),
    .operands_ready_o(gemm_operands_ready),
    .act_i           (gemm_hidden_bus_d),
    .wt_i            (lmhead_wt_bus),
    .acc_valid_o     (gemm_acc_valid),
    .acc_ready_i     (1'b1),
    .acc_o           (gemm_acc_bus),
    .busy_o          (gemm_busy)
  );

  assign lmhead_wt_ready = gemm_operands_ready;

  lm_head_controller u_lm_head_controller (
    .ap_clk              (ap_clk),
    .ap_rst_n            (ap_rst_n),
    .start_i             (core_start_w),
    .hidden_valid_i      (core_start_w),
    .hidden_ready_o      (lmh_hidden_ready),
    .hidden_i            (hidden_seed_q),
    .hidden_scale_valid_i(core_start_w),
    .hidden_scale_ready_o(lmh_hidden_scale_ready),
    .hidden_scale_i      (hidden_scale_q),
    .context_valid_o     (),
    .hidden_o            (lmh_hidden_bus),
    .hidden_scale_o      (lmh_hidden_scale_bus),
    .sched_start_o       (lmh_sched_start),
    .vocab_tile_idx_o    (lmh_vocab_tile_idx),
    .sched_done_i        (lmh_sched_done_q),
    .logits_valid_i      (lmh_logits_valid_q),
    .logits_ready_o      (lmh_logits_ready),
    .logits_i            (lmh_logits_bus_q),
    .argmax_valid_o      (lmh_argmax_valid),
    .argmax_ready_i      (lmh_argmax_ready),
    .argmax_o            (lmh_argmax_bus),
    .busy_o              (lmh_busy),
    .done_pulse_o        (lmh_done_pulse)
  );

  argmax_reduction u_argmax_reduction (
    .ap_clk        (ap_clk),
    .ap_rst_n      (ap_rst_n),
    .start_i       (argmax_start_w),
    .logits_valid_i(lmh_argmax_valid),
    .logits_ready_o(argmax_logits_ready),
    .logits_i      (lmh_argmax_bus),
    .token_valid_o (argmax_token_valid),
    .token_ready_i (token_ready_i),
    .token_id_o    (argmax_token_id),
    .token_logit_o (argmax_token_logit),
    .busy_o        (argmax_busy),
    .done_pulse_o  ()
  );

  always_ff @(posedge ap_clk) begin
    lmh_sched_done_q <= 1'b0;

    if (!helper_rst_n) begin
      scale_seen_q <= 1'b0;
      tile_seen_q <= '0;
      context_ready_q <= 1'b0;
      start_pending_q <= 1'b0;
      row_count_q <= '0;
      last_row_idx_q <= '0;
      hidden_scale_q <= '0;
      hidden_seed_q <= '0;
      lmhead_req_valid_q <= 1'b0;
      lmhead_req_addr_q <= '0;
      lmhead_group_active_q <= 1'b0;
      active_vocab_tile_q <= '0;
      active_group_q <= '0;
      feature_idx_q <= '0;
      tile_complete_q <= 1'b0;
      logits_pending_q <= 1'b0;
      lmh_logits_valid_q <= 1'b0;
      group_result_pending_q <= 1'b0;
      lmh_logits_bus_q <= '0;
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        hidden_last_row_tile_q[tile_idx] <= '0;
      end
      for (int lane = 0; lane < VOCAB_TILE; lane++) begin
        tile_logits_q[lane] <= '0;
      end
    end else begin
      if (launch_i) begin
        scale_seen_q <= 1'b0;
        tile_seen_q <= '0;
        context_ready_q <= 1'b0;
        start_pending_q <= 1'b0;
        row_count_q <= '0;
        last_row_idx_q <= '0;
        hidden_scale_q <= '0;
        hidden_seed_q <= '0;
        lmhead_req_valid_q <= 1'b0;
        lmhead_req_addr_q <= '0;
        lmhead_group_active_q <= 1'b0;
        active_vocab_tile_q <= '0;
        active_group_q <= '0;
        feature_idx_q <= '0;
        tile_complete_q <= 1'b0;
        logits_pending_q <= 1'b0;
        lmh_logits_valid_q <= 1'b0;
        group_result_pending_q <= 1'b0;
        lmh_logits_bus_q <= '0;
      end else begin
        if (start_i) begin
          start_pending_q <= 1'b1;
        end

        if (hidden_scale_valid_i && hidden_scale_ready_o) begin
          hidden_scale_q <= hidden_scale_i;
          scale_seen_q <= 1'b1;
          row_count_q <= hidden_scale_i.tag.seq_count;
          if (hidden_scale_i.tag.seq_count > 0) begin
            last_row_idx_q <= hidden_scale_i.tag.seq_count[LAST_ROW_COUNT_W-1:0] - LAST_ROW_COUNT_W'(1);
          end else begin
            last_row_idx_q <= '0;
          end
        end

        if (hidden_act_valid_i && hidden_act_ready_o &&
            (hidden_act_i.tag.tile_id < FEATURE_TILE_COUNT)) begin
          tile_seen_q[hidden_act_i.tag.tile_id] <= 1'b1;
          if (hidden_act_i.tag.tile_id == TILE_ID_W'(0)) begin
            hidden_seed_q <= hidden_act_i;
          end
          begin
            logic signed [(N_TILE * ACT_W)-1:0] last_row_tile_d;
            last_row_tile_d = '0;
            for (int lane = 0; lane < N_TILE; lane++) begin
              last_row_tile_d[(lane * ACT_W) +: ACT_W] =
                hidden_act_flat_w[((((last_row_idx_q * N_TILE) + lane) * ACT_W)) +: ACT_W];
            end
            hidden_last_row_tile_q[hidden_act_i.tag.tile_id] <= last_row_tile_d;
          end
        end

        if (hidden_done_pulse_i && scale_seen_q && (&tile_seen_q)) begin
          context_ready_q <= 1'b1;
        end

        if (core_start_w) begin
          start_pending_q <= 1'b0;
        end

        if (lmh_sched_start) begin
          active_vocab_tile_q <= lmh_vocab_tile_idx;
          active_group_q <= '0;
          feature_idx_q <= '0;
          lmhead_req_addr_q <= lmhead_base_addr_i + lmhead_group_offset(lmh_vocab_tile_idx, '0);
          lmhead_req_valid_q <= 1'b1;
          lmhead_group_active_q <= 1'b1;
          tile_complete_q <= 1'b0;
          logits_pending_q <= 1'b0;
          lmh_logits_valid_q <= 1'b0;
          group_result_pending_q <= 1'b0;
          for (int lane = 0; lane < VOCAB_TILE; lane++) begin
            tile_logits_q[lane] <= '0;
          end
        end else if (rd_desc_valid_o && rd_desc_ready_i) begin
          lmhead_req_valid_q <= 1'b0;
        end

        if (lmhead_reader_done_pulse) begin
          group_result_pending_q <= 1'b1;
        end

        if (lmhead_group_active_q && lmhead_wt_valid && lmhead_wt_ready) begin
          if (feature_idx_q == COUNT_W'(FEATURE_TILE_COUNT - 1)) begin
            feature_idx_q <= '0;
          end else begin
            feature_idx_q <= feature_idx_q + COUNT_W'(1);
          end
        end

        if ((group_result_pending_q || lmhead_reader_done_pulse) && gemm_acc_valid) begin
          group_result_pending_q <= 1'b0;
          for (int lane = 0; lane < N_TILE; lane++) begin
            tile_logits_q[(active_group_q * N_TILE) + lane] <=
              gemm_acc_flat_w[((lane * ACC_W)) +: ACC_W];
          end

          if (active_group_q == LMHEAD_LAST_GROUP) begin
            lmhead_group_active_q <= 1'b0;
            tile_complete_q <= 1'b1;
          end else begin
            active_group_q <= active_group_q + 1'b1;
            feature_idx_q <= '0;
            lmhead_req_addr_q <= lmhead_base_addr_i +
                                 lmhead_group_offset(active_vocab_tile_q, active_group_q + 1'b1);
            lmhead_req_valid_q <= 1'b1;
          end
        end

        if (tile_complete_q) begin
          lmh_sched_done_q <= 1'b1;
          lmh_logits_bus_q <= lmh_logits_bus_d;
          logits_pending_q <= 1'b1;
          tile_complete_q <= 1'b0;
        end else if (logits_pending_q) begin
          lmh_logits_valid_q <= 1'b1;
          logits_pending_q <= 1'b0;
        end else if (lmh_logits_valid_q && lmh_logits_ready) begin
          lmh_logits_valid_q <= 1'b0;
        end
      end
    end
  end

endmodule
