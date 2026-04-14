`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_decoder_layer_smoke;

  localparam int unsigned MAX_BLOCKS = 160;

  localparam string PREFILL_META_FILE       = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.meta.memh";
  localparam string PREFILL_BLOCK_IDS_FILE  = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.block_ids.memh";
  localparam string PREFILL_Q_HEADS_FILE    = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.q_heads.memh";
  localparam string PREFILL_KV_HEADS_FILE   = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.kv_heads.memh";
  localparam string PREFILL_GEMM_MODES_FILE = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.gemm_modes.memh";
  localparam string PREFILL_GEMM_STEPS_FILE = "sim/golden_traces/phase7/rtl/phase7_prefill_layer0_schedule.gemm_steps.memh";

  localparam string DECODE_META_FILE       = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.meta.memh";
  localparam string DECODE_BLOCK_IDS_FILE  = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.block_ids.memh";
  localparam string DECODE_Q_HEADS_FILE    = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.q_heads.memh";
  localparam string DECODE_KV_HEADS_FILE   = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.kv_heads.memh";
  localparam string DECODE_GEMM_MODES_FILE = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.gemm_modes.memh";
  localparam string DECODE_GEMM_STEPS_FILE = "sim/golden_traces/phase7/rtl/phase7_decode_layer0_schedule.gemm_steps.memh";

  logic clk;
  logic rst_n;
  logic start;
  logic abort_req;
  logic block_done;
  runtime_mode_e runtime_mode;

  logic                  layer_busy;
  logic                  layer_done;
  logic                  layer_start;
  logic                  layer_ctx_valid;
  logic                  block_valid;
  logic                  block_start;
  runtime_mode_e         runtime_mode_out;
  logic [LAYER_ID_W-1:0] layer_id;
  logic [LAYER_ID_W-1:0] weight_layer_sel;
  logic [LAYER_ID_W-1:0] kv_layer_sel;
  block_id_e             block_id;
  logic [Q_HEAD_ID_W-1:0]  q_head_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;

  logic                  sched_busy;
  logic                  sched_done;
  logic                  step_valid;
  gemm_mode_e            gemm_mode;
  block_id_e             sched_block_id;
  logic                  clear_acc;
  logic                  emit_acc;
  logic [TILE_ID_W-1:0]  m_tile_idx;
  logic [TILE_ID_W-1:0]  n_tile_idx;
  logic [TILE_ID_W-1:0]  k_tile_idx;
  logic [TILE_ID_W-1:0]  m_tile_count;
  logic [TILE_ID_W-1:0]  n_tile_count;
  logic [TILE_ID_W-1:0]  k_tile_count;
  logic [Q_HEAD_ID_W-1:0]  sched_q_head_id;
  logic [KV_HEAD_ID_W-1:0] sched_kv_head_id;

  logic                  act_valid;
  logic                  act_ready;
  logic                  weight_valid;
  logic                  weight_ready;
  logic                  score_valid_in;
  logic                  score_ready_in;
  logic                  kv_valid;
  logic                  kv_ready;
  logic                  operands_valid;
  act_bus_t              act_bus;
  wt_bus_t               weight_bus;
  act_bus_t              score_bus_in;
  act_bus_t              kv_bus;
  act_bus_t              routed_act;
  wt_bus_t               routed_wt;

  logic                  acc_valid;
  logic                  acc_ready;
  acc_bus_t              acc_bus;
  logic                  scale_valid;
  scale_bus_t            scale_bus;
  logic                  quant_valid;
  logic                  quant_ready;
  act_bus_t              quant_bus;
  logic                  score_valid;
  logic                  score_ready;
  acc_bus_t              score_bus_out;
  logic                  lmhead_valid;
  logic                  lmhead_ready;
  acc_bus_t              lmhead_bus;

  logic [COUNT_W-1:0]    seq_count_cfg;
  logic [COUNT_W-1:0]    kv_token_count_cfg;
  integer                meta_mem [0:2];
  integer                block_id_mem [0:MAX_BLOCKS-1];
  integer                q_head_mem [0:MAX_BLOCKS-1];
  integer                kv_head_mem [0:MAX_BLOCKS-1];
  integer                gemm_mode_mem [0:MAX_BLOCKS-1];
  integer                gemm_steps_mem [0:MAX_BLOCKS-1];

  layer_controller dut_layer (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(start),
    .abort_req_i(abort_req),
    .runtime_mode_i(runtime_mode),
    .block_done_i(block_done),
    .busy_o(layer_busy),
    .run_done_o(layer_done),
    .layer_start_o(layer_start),
    .layer_ctx_valid_o(layer_ctx_valid),
    .block_valid_o(block_valid),
    .block_start_o(block_start),
    .runtime_mode_o(runtime_mode_out),
    .layer_id_o(layer_id),
    .weight_layer_sel_o(weight_layer_sel),
    .kv_layer_sel_o(kv_layer_sel),
    .block_id_o(block_id),
    .q_head_id_o(q_head_id),
    .kv_head_id_o(kv_head_id)
  );

  gemm_op_scheduler dut_sched (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .start_i(1'b0),
    .abort_req_i(abort_req),
    .lm_head_only_i(1'b0),
    .block_start_i(block_start),
    .block_id_i(block_id),
    .block_q_head_id_i(q_head_id),
    .block_kv_head_id_i(kv_head_id),
    .dma_ready_i(1'b1),
    .buffer_ready_i(1'b1),
    .step_ready_i(1'b1),
    .seq_count_i(seq_count_cfg),
    .kv_token_count_i(kv_token_count_cfg),
    .busy_o(sched_busy),
    .done_pulse_o(sched_done),
    .step_valid_o(step_valid),
    .gemm_mode_o(gemm_mode),
    .block_id_o(sched_block_id),
    .clear_acc_o(clear_acc),
    .emit_acc_o(emit_acc),
    .m_tile_idx_o(m_tile_idx),
    .n_tile_idx_o(n_tile_idx),
    .k_tile_idx_o(k_tile_idx),
    .m_tile_count_o(m_tile_count),
    .n_tile_count_o(n_tile_count),
    .k_tile_count_o(k_tile_count),
    .q_head_id_o(sched_q_head_id),
    .kv_head_id_o(sched_kv_head_id)
  );

  gemm_operand_router dut_operand_router (
    .gemm_mode_i(gemm_mode),
    .act_valid_i(act_valid),
    .act_ready_o(act_ready),
    .act_i(act_bus),
    .weight_valid_i(weight_valid),
    .weight_ready_o(weight_ready),
    .weight_i(weight_bus),
    .score_valid_i(score_valid_in),
    .score_ready_o(score_ready_in),
    .score_i(score_bus_in),
    .kv_valid_i(kv_valid),
    .kv_ready_o(kv_ready),
    .kv_i(kv_bus),
    .operands_valid_o(operands_valid),
    .operands_ready_i(1'b1),
    .act_o(routed_act),
    .wt_o(routed_wt)
  );

  gemm_result_router dut_result_router (
    .gemm_mode_i(gemm_mode),
    .acc_valid_i(acc_valid),
    .acc_ready_o(acc_ready),
    .acc_i(acc_bus),
    .scale_valid_i(scale_valid),
    .scale_i(scale_bus),
    .quant_valid_o(quant_valid),
    .quant_ready_i(1'b1),
    .quant_o(quant_bus),
    .score_valid_o(score_valid),
    .score_ready_i(1'b1),
    .score_o(score_bus_out),
    .lmhead_valid_o(lmhead_valid),
    .lmhead_ready_i(1'b1),
    .lmhead_o(lmhead_bus)
  );

  always #5 clk = ~clk;

  // Keep the synthetic operand/result buses aligned to the active layer context.
  always_comb begin
    act_valid      = 1'b1;
    weight_valid   = 1'b1;
    score_valid_in = 1'b1;
    kv_valid       = 1'b1;
    acc_valid      = step_valid;
    scale_valid    = 1'b1;
    act_bus        = '0;
    weight_bus     = '0;
    score_bus_in   = '0;
    kv_bus         = '0;
    acc_bus        = '0;
    scale_bus      = '0;

    act_bus.data[0]      = 8'sd7;
    weight_bus.data[0]   = -8'sd5;
    score_bus_in.data[0] = 8'sd22;
    kv_bus.data[0]       = 8'sd9;
    acc_bus.data[0]      = 32'sd12;
    scale_bus.data[0]    = 32'h0001_0000;

    act_bus.tag.layer_id       = layer_id;
    act_bus.tag.block_id       = BLOCK_RMSNORM1;
    act_bus.tag.gemm_mode      = GEMM_NONE;
    act_bus.tag.tile_id        = m_tile_idx;
    act_bus.tag.token_base     = '0;
    act_bus.tag.seq_count      = seq_count_cfg;
    act_bus.tag.q_head_id      = q_head_id;
    act_bus.tag.kv_head_id     = kv_head_id;
    act_bus.tag.elem_count     = ELEM_COUNT_W'(ACT_VECTOR_ELEMS);
    act_bus.tag.is_last        = 1'b1;

    weight_bus.tag.layer_id    = layer_id;
    weight_bus.tag.block_id    = block_id;
    weight_bus.tag.gemm_mode   = gemm_mode;
    weight_bus.tag.tile_id     = n_tile_idx;
    weight_bus.tag.token_base  = '0;
    weight_bus.tag.seq_count   = seq_count_cfg;
    weight_bus.tag.elem_count  = ELEM_COUNT_W'(WEIGHT_VECTOR_ELEMS);
    weight_bus.tag.is_last     = 1'b1;

    score_bus_in.tag           = act_bus.tag;
    score_bus_in.tag.block_id  = BLOCK_SOFTMAX;
    score_bus_in.tag.gemm_mode = GEMM_NONE;
    kv_bus.tag                 = act_bus.tag;
    kv_bus.tag.block_id        = BLOCK_WEIGHTED_SUM;
    kv_bus.tag.gemm_mode       = GEMM_NONE;
    kv_bus.tag.q_head_id       = q_head_id;
    kv_bus.tag.kv_head_id      = kv_head_id;

    acc_bus.tag.layer_id       = layer_id;
    acc_bus.tag.block_id       = sched_block_id;
    acc_bus.tag.gemm_mode      = gemm_mode;
    acc_bus.tag.tile_id        = m_tile_idx;
    acc_bus.tag.token_base     = '0;
    acc_bus.tag.seq_count      = seq_count_cfg;
    acc_bus.tag.q_head_id      = sched_q_head_id;
    acc_bus.tag.kv_head_id     = sched_kv_head_id;
    acc_bus.tag.elem_count     = ELEM_COUNT_W'(ACC_VECTOR_ELEMS);
    acc_bus.tag.is_last        = emit_acc;
    scale_bus.tag              = acc_bus.tag;
  end

  task automatic load_case(
    input string meta_file,
    input string block_ids_file,
    input string q_heads_file,
    input string kv_heads_file,
    input string gemm_modes_file,
    input string gemm_steps_file
  );
    begin
      $readmemh(meta_file, meta_mem);
      $readmemh(block_ids_file, block_id_mem);
      $readmemh(q_heads_file, q_head_mem);
      $readmemh(kv_heads_file, kv_head_mem);
      $readmemh(gemm_modes_file, gemm_mode_mem);
      $readmemh(gemm_steps_file, gemm_steps_mem);
    end
  endtask

  function automatic logic block_is_gemm(
    input block_id_e block_id_value
  );
    begin
      unique case (block_id_value)
        BLOCK_Q,
        BLOCK_K,
        BLOCK_V,
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM,
        BLOCK_O,
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_DOWN: block_is_gemm = 1'b1;
        default:    block_is_gemm = 1'b0;
      endcase
    end
  endfunction

  task automatic check_router_paths(
    input integer expected_gemm_mode,
    input integer expected_block_id,
    input integer expected_q_head,
    input integer expected_kv_head
  );
    begin
      if (sched_block_id != expected_block_id[BLOCK_ID_W-1:0]) begin
        $error("decoder layer smoke expected scheduler block_id %0d, got %0d", expected_block_id, sched_block_id);
        $finish;
      end
      if (gemm_mode != expected_gemm_mode[GEMM_MODE_W-1:0]) begin
        $error("decoder layer smoke expected gemm_mode %0d, got %0d", expected_gemm_mode, gemm_mode);
        $finish;
      end
      if ((sched_q_head_id != expected_q_head[Q_HEAD_ID_W-1:0]) ||
          (sched_kv_head_id != expected_kv_head[KV_HEAD_ID_W-1:0])) begin
        $error("decoder layer smoke scheduler head mismatch q=%0d/%0d kv=%0d/%0d",
          sched_q_head_id, expected_q_head, sched_kv_head_id, expected_kv_head);
        $finish;
      end

      unique case (gemm_mode)
        GEMM_Q,
        GEMM_K,
        GEMM_V,
        GEMM_O,
        GEMM_GATE,
        GEMM_UP,
        GEMM_DOWN: begin
          if (!operands_valid || !act_ready || !weight_ready ||
              (routed_act.data[0] != act_bus.data[0]) || (routed_wt.data[0] != weight_bus.data[0]) ||
              !quant_valid || score_valid || lmhead_valid) begin
            $error("decoder layer smoke generic GEMM route mismatch for mode %0d", gemm_mode);
            $finish;
          end
        end

        GEMM_SCORE: begin
          if (!operands_valid || !act_ready || !kv_ready || weight_ready ||
              (routed_act.data[0] != act_bus.data[0]) || (routed_wt.data[0] != kv_bus.data[0]) ||
              !score_valid || quant_valid || lmhead_valid) begin
            $error("decoder layer smoke SCORE route mismatch");
            $finish;
          end
        end

        GEMM_WEIGHTED_SUM: begin
          if (!operands_valid || !score_ready_in || !kv_ready || act_ready ||
              (routed_act.data[0] != score_bus_in.data[0]) || (routed_wt.data[0] != kv_bus.data[0]) ||
              !quant_valid || score_valid || lmhead_valid) begin
            $error("decoder layer smoke WEIGHTED_SUM route mismatch");
            $finish;
          end
        end

        default: begin
          $error("decoder layer smoke unexpected GEMM mode %0d", gemm_mode);
          $finish;
        end
      endcase
    end
  endtask

  task automatic run_case(
    input string case_name,
    input runtime_mode_e case_mode,
    input string meta_file,
    input string block_ids_file,
    input string q_heads_file,
    input string kv_heads_file,
    input string gemm_modes_file,
    input string gemm_steps_file
  );
    int unsigned block_count;
    int unsigned block_idx;
    int unsigned wait_cycles;
    int unsigned observed_steps;
    begin
      load_case(meta_file, block_ids_file, q_heads_file, kv_heads_file, gemm_modes_file, gemm_steps_file);
      seq_count_cfg      = COUNT_W'(meta_mem[0]);
      kv_token_count_cfg = COUNT_W'(meta_mem[1]);
      block_count        = meta_mem[2];

      runtime_mode = case_mode;

      @(negedge clk);
      start <= 1'b1;
      @(negedge clk);
      start <= 1'b0;

      if (!layer_busy || !block_valid || !layer_ctx_valid || (runtime_mode_out != case_mode)) begin
        $error("decoder layer smoke failed to launch %s", case_name);
        $finish;
      end

      for (block_idx = 0; block_idx < block_count; block_idx++) begin
        wait_cycles = 0;
        while (!block_start) begin
          @(negedge clk);
          wait_cycles++;
          if (wait_cycles > 512) begin
            $error("decoder layer smoke timeout waiting for block %0d in %s", block_idx, case_name);
            $finish;
          end
        end

        if ((block_idx == 0) && !layer_start) begin
          $error("decoder layer smoke expected layer_start on first block in %s", case_name);
          $finish;
        end
        if ((layer_id != '0) || (weight_layer_sel != '0) || (kv_layer_sel != '0)) begin
          $error("decoder layer smoke expected first-layer selectors in %s", case_name);
          $finish;
        end
        if ((block_id != block_id_mem[block_idx][BLOCK_ID_W-1:0]) ||
            (q_head_id != q_head_mem[block_idx][Q_HEAD_ID_W-1:0]) ||
            (kv_head_id != kv_head_mem[block_idx][KV_HEAD_ID_W-1:0])) begin
          $error("decoder layer smoke block mismatch in %s at idx %0d", case_name, block_idx);
          $finish;
        end

        if (block_is_gemm(block_id)) begin
          observed_steps = 0;
          wait_cycles = 0;
          while (!sched_done) begin
            @(negedge clk);
            wait_cycles++;
            if (step_valid) begin
              observed_steps++;
              check_router_paths(
                gemm_mode_mem[block_idx],
                block_id_mem[block_idx],
                q_head_mem[block_idx],
                kv_head_mem[block_idx]
              );
            end
            if (wait_cycles > (gemm_steps_mem[block_idx] + 256)) begin
              $error("decoder layer smoke timeout waiting for GEMM block %0d in %s", block_idx, case_name);
              $finish;
            end
          end
          if (observed_steps != gemm_steps_mem[block_idx]) begin
            $error("decoder layer smoke step-count mismatch in %s block %0d exp=%0d got=%0d",
              case_name, block_idx, gemm_steps_mem[block_idx], observed_steps);
            $finish;
          end
        end else begin
          #1;
          if (sched_busy || step_valid) begin
            $error("decoder layer smoke expected non-GEMM block %0d in %s to bypass scheduler", block_idx, case_name);
            $finish;
          end
        end

        block_done <= 1'b1;
        @(negedge clk);
        block_done <= 1'b0;
      end

      wait_cycles = 0;
      while (!(layer_start && (layer_id == LAYER_ID_W'(1)))) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 512) begin
          $error("decoder layer smoke expected transition into layer 1 after %s", case_name);
          $finish;
        end
      end

      abort_req <= 1'b1;
      @(negedge clk);
      abort_req <= 1'b0;

      if (!layer_done || layer_busy) begin
        $error("decoder layer smoke expected abort completion after %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk               = 1'b0;
    rst_n             = 1'b0;
    start             = 1'b0;
    abort_req         = 1'b0;
    block_done        = 1'b0;
    runtime_mode      = MODE_PREFILL;
    seq_count_cfg     = '0;
    kv_token_count_cfg = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    run_case(
      "phase7_prefill_layer0_schedule",
      MODE_PREFILL,
      PREFILL_META_FILE,
      PREFILL_BLOCK_IDS_FILE,
      PREFILL_Q_HEADS_FILE,
      PREFILL_KV_HEADS_FILE,
      PREFILL_GEMM_MODES_FILE,
      PREFILL_GEMM_STEPS_FILE
    );

    repeat (4) @(negedge clk);

    run_case(
      "phase7_decode_layer0_schedule",
      MODE_DECODE,
      DECODE_META_FILE,
      DECODE_BLOCK_IDS_FILE,
      DECODE_Q_HEADS_FILE,
      DECODE_KV_HEADS_FILE,
      DECODE_GEMM_MODES_FILE,
      DECODE_GEMM_STEPS_FILE
    );

    $display("PASS: tb_decoder_layer_smoke");
    $finish;
  end

endmodule
