import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module runtime_decoder_datapath (
  input  logic                    ap_clk,
  input  logic                    ap_rst_n,
  input  logic                    launch_i,
  input  logic                    abort_req_i,
  input  logic                    embed_scale_valid_i,
  output logic                    embed_scale_ready_o,
  input  scale_bus_t              embed_scale_i,
  input  logic                    embed_act_valid_i,
  output logic                    embed_act_ready_o,
  input  act_bus_t                embed_act_i,
  input  logic                    block_valid_i,
  input  logic                    block_start_i,
  input  runtime_mode_e           runtime_mode_i,
  input  logic [LAYER_ID_W-1:0]   layer_id_i,
  input  block_id_e               block_id_i,
  input  logic [Q_HEAD_ID_W-1:0]  q_head_id_i,
  input  logic [KV_HEAD_ID_W-1:0] kv_head_id_i,
  output logic                    context_valid_o,
  output logic                    block_done_o,
  output logic                    final_scale_valid_o,
  input  logic                    final_scale_ready_i,
  output scale_bus_t              final_scale_o,
  output logic                    final_act_valid_o,
  input  logic                    final_act_ready_i,
  output act_bus_t                final_act_o,
  output logic                    final_hidden_done_pulse_o,
  output logic                    busy_o
);

  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned BLOCKS_PER_LAYER = 6 + (4 * N_Q_HEADS) + 12;

  typedef enum logic [4:0] {
    DDP_IDLE            = 5'd0,
    DDP_CAPTURE         = 5'd1,
    DDP_READY           = 5'd2,
    DDP_BLOCK           = 5'd3,
    DDP_APPLY_SEND      = 5'd4,
    DDP_APPLY_WAIT      = 5'd5,
    DDP_GEMM_SEND       = 5'd6,
    DDP_GEMM_WAIT       = 5'd7,
    DDP_SILU_SEND       = 5'd8,
    DDP_SILU_SCALE      = 5'd9,
    DDP_SILU_ACT        = 5'd10,
    DDP_MUL_SEND        = 5'd11,
    DDP_MUL_WAIT        = 5'd12,
    DDP_OUT_SCALE       = 5'd13,
    DDP_OUT_ACT         = 5'd14,
    DDP_MASK_APPLY      = 5'd15,
    DDP_SOFTMAX_ARM     = 5'd16,
    DDP_SOFTMAX_SCORE   = 5'd17,
    DDP_SOFTMAX_SCALE   = 5'd18,
    DDP_SOFTMAX_ACT     = 5'd19
  } ddp_state_e;

  ddp_state_e                           state_q;
  logic                                 scale_seen_q;
  logic [FEATURE_TILE_COUNT-1:0]        tile_seen_q;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] hidden_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] hidden_scale_q;
  logic [POS_W-1:0]                     context_token_base_q;
  logic [COUNT_W-1:0]                   context_seq_count_q;
  logic [ELEM_COUNT_W-1:0]              context_elem_count_q;
  logic                                 context_is_partial_q;
  logic [31:0]                          context_signature_q;
  logic [3:0]                           block_countdown_q;
  logic [7:0]                           layer_block_count_q;
  logic [TILE_ID_W-1:0]                 final_tile_idx_q;
  logic [TILE_ID_W-1:0]                 tile_cursor_q;
  logic [TILE_ID_W-1:0]                 apply_tile_idx_q;
  logic [TILE_ID_W-1:0]                 apply_stride_q;
  logic [TILE_ID_W-1:0]                 ffn_tile_anchor_q;
  logic [TILE_ID_W-1:0]                 ffn_stride_q;
  logic [TILE_ID_W-1:0]                 attn_o_tile_anchor_q;
  logic [TILE_ID_W-1:0]                 attn_o_stride_q;
  logic                                 down_apply_from_gemm_q;
  logic [LAYER_ID_W-1:0]                active_layer_q;
  block_id_e                            active_block_q;
  logic [Q_HEAD_ID_W-1:0]               active_q_head_q;
  logic [KV_HEAD_ID_W-1:0]              active_kv_head_q;
  logic [31:0]                          apply_signature_q;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] ffn_gate_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] ffn_up_tiles_q   [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] ffn_silu_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] ffn_mul_tiles_q  [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] attn_weighted_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] attn_o_tiles_q        [0:FEATURE_TILE_COUNT-1];
  logic [FEATURE_TILE_COUNT-1:0]         attn_weighted_valid_q;
  logic [FEATURE_TILE_COUNT-1:0]         attn_o_valid_q;
  acc_bus_t                               attn_score_acc_q;
  acc_bus_t                               attn_masked_acc_q;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] attn_prob_tile_q;
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] attn_score_scale_q;
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] attn_prob_scale_q;
  logic                                   attn_score_valid_q;
  logic                                   attn_masked_valid_q;
  logic                                   attn_prob_valid_q;
  scale_bus_t                           final_scale_d;
  act_bus_t                             final_act_d;
  acc_bus_t                             apply_residual_d;
  acc_bus_t                             apply_update_d;
  acc_bus_t                             apply_sum_w;
  scale_bus_t                           apply_scale_d;
  act_bus_t                             apply_requant_w;
  act_bus_t                             gemm_act_d;
  wt_bus_t                              gemm_wt_d;
  acc_bus_t                             gemm_acc_w;
  scale_bus_t                           gemm_scale_d;
  act_bus_t                             gemm_requant_w;
  act_bus_t                             silu_gate_d;
  scale_bus_t                           silu_scale_w;
  act_bus_t                             silu_act_w;
  act_bus_t                             mul_silu_d;
  act_bus_t                             mul_up_d;
  acc_bus_t                             mul_prod_w;
  scale_bus_t                           mul_scale_d;
  act_bus_t                             mul_requant_w;
  acc_bus_t                             down_gemm_acc_q;
  logic signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] down_gemm_acc_flat_d;
  logic signed [(WEIGHT_VECTOR_ELEMS * WEIGHT_W)-1:0] gemm_wt_flat_d;
  logic signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] apply_residual_flat_d;
  logic signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] apply_update_flat_d;
  logic                                 apply_residual_ready_w;
  logic                                 apply_update_ready_w;
  logic                                 apply_sum_valid_w;
  logic                                 gemm_operands_ready_w;
  logic                                 gemm_acc_valid_w;
  logic                                 gemm_busy_w;
  logic [SCALE_W-1:0]                   silu_input_scale_d;
  logic [SCALE_W-1:0]                   silu_output_scale_d;
  logic                                 silu_gate_ready_w;
  logic                                 silu_scale_valid_w;
  logic                                 silu_act_valid_w;
  logic                                 silu_busy_w;
  logic                                 silu_done_w;
  logic                                 mul_silu_ready_w;
  logic                                 mul_up_ready_w;
  logic                                 mul_prod_valid_w;
  logic                                 mul_busy_w;
  logic                                 mul_done_w;
  acc_bus_t                             masked_score_comb_w;
  logic                                 softmax_score_ready_w;
  logic                                 softmax_prob_valid_w;
  logic                                 softmax_prob_scale_valid_w;
  logic                                 softmax_busy_w;
  logic                                 softmax_done_w;
  scale_bus_t                           softmax_prob_scale_bus_w;
  act_bus_t                             softmax_prob_bus_w;
  logic [SCALE_W-1:0]                   softmax_score_scale_w;


  wire                                  softmax_score_valid_w;
  wire                                  softmax_prob_scale_ready_w;
  wire                                  softmax_prob_ready_w;
  wire                                  gemm_operands_valid_w;

  function automatic logic signed [31:0] sext_act8(
    input logic signed [ACT_W-1:0] value
  );
    begin
      sext_act8 = {{(32-ACT_W){value[ACT_W-1]}}, value};
    end
  endfunction

  function automatic logic [31:0] block_signature(
    input logic [LAYER_ID_W-1:0]   layer_id,
    input block_id_e               block_id,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id,
    input runtime_mode_e           runtime_mode
  );
    logic [31:0] sig;
    begin
      sig = 32'h0;
      sig[0 +: LAYER_ID_W] = layer_id;
      sig[8 +: BLOCK_ID_W] = block_id;
      sig[16 +: Q_HEAD_ID_W] = q_head_id;
      sig[24 +: KV_HEAD_ID_W] = kv_head_id;
      sig[31] = runtime_mode;
      block_signature = sig;
    end
  endfunction

  function automatic logic [3:0] block_latency(
    input block_id_e              block_id,
    input logic [Q_HEAD_ID_W-1:0] q_head_id,
    input logic [31:0]            context_signature
  );
    logic [3:0] latency_d;
    begin
      unique case (block_id)
        BLOCK_RMSNORM1,
        BLOCK_RMSNORM2:       latency_d = 4'd3;
        BLOCK_Q,
        BLOCK_K,
        BLOCK_V,
        BLOCK_O,
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_DOWN:           latency_d = 4'd4;
        BLOCK_ROPE,
        BLOCK_KV_CACHE_WRITE,
        BLOCK_RESIDUAL1,
        BLOCK_RESIDUAL2,
        BLOCK_REQUANTIZE,
        BLOCK_GLU_MUL:        latency_d = 4'd2;
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM,
        BLOCK_CAUSAL_MASK,
        BLOCK_SOFTMAX:        latency_d = 4'd1;
        BLOCK_SILU:           latency_d = 4'd3;
        default:              latency_d = 4'd1;
      endcase
      block_latency = latency_d + {3'd0, context_signature[0]};
    end
  endfunction

  function automatic logic is_ffn_chain_block(
    input block_id_e block_id
  );
    begin
      unique case (block_id)
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_SILU,
        BLOCK_GLU_MUL,
        BLOCK_DOWN: is_ffn_chain_block = 1'b1;
        default:    is_ffn_chain_block = 1'b0;
      endcase
    end
  endfunction

  function automatic logic is_ffn_anchor_block(
    input block_id_e block_id
  );
    begin
      is_ffn_anchor_block = (block_id == BLOCK_GATE);
    end
  endfunction

  function automatic logic is_ffn_projection_block(
    input block_id_e block_id
  );
    begin
      unique case (block_id)
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_DOWN,
        BLOCK_O,
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM: is_ffn_projection_block = 1'b1;
        default:            is_ffn_projection_block = 1'b0;
      endcase
    end
  endfunction

  function automatic logic is_attn_output_chain_block(
    input block_id_e block_id
  );
    begin
      unique case (block_id)
        BLOCK_O,
        BLOCK_RESIDUAL1: is_attn_output_chain_block = 1'b1;
        default:         is_attn_output_chain_block = 1'b0;
      endcase
    end
  endfunction

  function automatic logic is_attn_output_anchor_block(
    input block_id_e block_id
  );
    begin
      is_attn_output_anchor_block = (block_id == BLOCK_O);
    end
  endfunction

  function automatic gemm_mode_e projection_gemm_mode(
    input block_id_e block_id
  );
    begin
      unique case (block_id)
        BLOCK_GATE:         projection_gemm_mode = GEMM_GATE;
        BLOCK_UP:           projection_gemm_mode = GEMM_UP;
        BLOCK_DOWN:         projection_gemm_mode = GEMM_DOWN;
        BLOCK_O:            projection_gemm_mode = GEMM_O;
        BLOCK_SCORE:        projection_gemm_mode = GEMM_SCORE;
        BLOCK_WEIGHTED_SUM: projection_gemm_mode = GEMM_WEIGHTED_SUM;
        default:            projection_gemm_mode = GEMM_NONE;
      endcase
    end
  endfunction

  function automatic logic [15:0] score_query_rows_from_elem(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    int unsigned eff;
    begin
      eff = (elem_count == '0) ? int'(SCORE_CHUNK_ELEMS) : int'(elem_count);
      score_query_rows_from_elem =
        16'((eff + int'(SCORE_K_TILE) - 1) / int'(SCORE_K_TILE));
    end
  endfunction

  function automatic logic advance_tile_cursor_on_block(
    input block_id_e block_id
  );
    begin
      advance_tile_cursor_on_block =
        (!is_ffn_chain_block(block_id) && !is_attn_output_chain_block(block_id)) ||
        (block_id == BLOCK_DOWN) || (block_id == BLOCK_RESIDUAL1);
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] block_tile_stride(
    input block_id_e               block_id,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id,
    input logic [31:0]             signature
  );
    int unsigned stride_d;
    begin
      stride_d = 1 + (((int'(block_id) * 3) ^
                       (int'(q_head_id) * 5) ^
                       (int'(kv_head_id) * 7) ^
                       signature[5:0]) % FEATURE_TILE_COUNT);
      if ((stride_d & 1) == 0) begin
        if (stride_d == (FEATURE_TILE_COUNT - 1)) begin
          stride_d = FEATURE_TILE_COUNT - 1;
        end else begin
          stride_d = stride_d + 1;
        end
      end
      if (stride_d == 0) begin
        stride_d = 1;
      end
      block_tile_stride = TILE_ID_W'(stride_d);
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] next_tile_cursor(
    input logic [TILE_ID_W-1:0] current_tile,
    input logic [TILE_ID_W-1:0] stride
  );
    int unsigned next_d;
    begin
      next_d = (current_tile + stride) % FEATURE_TILE_COUNT;
      next_tile_cursor = TILE_ID_W'(next_d);
    end
  endfunction

  function automatic int unsigned block_neighbor_lane_idx(
    input int unsigned             lane_idx,
    input block_id_e               block_id,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id,
    input logic [31:0]             signature
  );
    int unsigned row_idx;
    int unsigned lane_local;
    int unsigned lane_offset;
    begin
      row_idx = lane_idx / N_TILE;
      lane_local = lane_idx % N_TILE;
      lane_offset = 1 + ((int'(block_id) +
                          int'(q_head_id) +
                          (int'(kv_head_id) << 1) +
                          signature[3:0]) % N_TILE);
      block_neighbor_lane_idx = (row_idx * N_TILE) + ((lane_local + lane_offset) % N_TILE);
    end
  endfunction

  function automatic logic signed [ACC_W-1:0] block_lane_bias(
    input logic [LAYER_ID_W-1:0]   layer_id,
    input block_id_e               block_id,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id,
    input logic [31:0]             signature,
    input logic [TILE_ID_W-1:0]    tile_idx,
    input int unsigned             lane_idx,
    input logic [COUNT_W-1:0]      seq_count
  );
    int signed lane_term;
    int signed row_term;
    int signed seq_last_row;
    int signed tile_term;
    int signed head_term;
    int signed layer_term;
    int signed sig_term;
    begin
      lane_term = (lane_idx % N_TILE) - (N_TILE / 2);
      seq_last_row = (seq_count == '0) ? 0 : (seq_count - 1'b1);
      row_term = (lane_idx / N_TILE) - seq_last_row;
      tile_term = tile_idx[5:0];
      head_term = int'(q_head_id) - (int'(kv_head_id) <<< 1);
      layer_term = int'(layer_id) - (N_LAYERS / 2);
      sig_term = signature[(lane_idx + int'(block_id)) & 5'h1f] ? 3 : -3;

      unique case (block_id)
        BLOCK_RMSNORM1,
        BLOCK_RMSNORM2:      block_lane_bias = sig_term - row_term + layer_term;
        BLOCK_Q,
        BLOCK_K,
        BLOCK_V:             block_lane_bias = lane_term + head_term + sig_term;
        BLOCK_ROPE,
        BLOCK_KV_CACHE_WRITE: block_lane_bias = sig_term - lane_term + row_term;
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM:  block_lane_bias = head_term + (row_term <<< 1) + sig_term;
        BLOCK_CAUSAL_MASK,
        BLOCK_SOFTMAX:       block_lane_bias = (sig_term <<< 1) - row_term - tile_term[2:0];
        BLOCK_O,
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_DOWN:          block_lane_bias = lane_term - head_term + layer_term + tile_term[2:0];
        BLOCK_RESIDUAL1,
        BLOCK_RESIDUAL2,
        BLOCK_REQUANTIZE:    block_lane_bias = sig_term + tile_term[3:0] - row_term;
        BLOCK_SILU,
        BLOCK_GLU_MUL:       block_lane_bias = lane_term + row_term + sig_term - layer_term;
        default:             block_lane_bias = sig_term + head_term;
      endcase
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] clamp_act8(
    input logic signed [ACC_W-1:0] value
  );
    begin
      if (value > 127) begin
        clamp_act8 = 8'sd127;
      end else if (value < -127) begin
        clamp_act8 = -8'sd127;
      end else begin
        clamp_act8 = ACT_W'(value);
      end
    end
  endfunction

  function automatic logic signed [WEIGHT_W-1:0] clamp_weight8(
    input logic signed [ACC_W-1:0] value
  );
    begin
      if (value > 31) begin
        clamp_weight8 = WEIGHT_W'(8'sd31);
      end else if (value < -31) begin
        clamp_weight8 = WEIGHT_W'(-8'sd31);
      end else begin
        clamp_weight8 = WEIGHT_W'(value);
      end
    end
  endfunction

  function automatic logic signed [WEIGHT_W-1:0] projection_weight_scalar(
    input logic [LAYER_ID_W-1:0]   layer_id,
    input block_id_e               block_id,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id,
    input logic [31:0]             signature,
    input logic [TILE_ID_W-1:0]    tile_idx,
    input int unsigned             lane_idx,
    input logic [COUNT_W-1:0]      seq_count
  );
    int signed lane_term;
    int signed row_term;
    int signed seq_last_row;
    int signed tile_term;
    int signed head_term;
    int signed layer_term;
    int signed sig_term;
    int signed weight_term;
    begin
      lane_term = (lane_idx % N_TILE) - (N_TILE / 2);
      seq_last_row = (seq_count == '0) ? 0 : (seq_count - 1'b1);
      row_term = (lane_idx / N_TILE) - seq_last_row;
      tile_term = tile_idx[5:0];
      head_term = int'(q_head_id) - (int'(kv_head_id) <<< 1);
      layer_term = int'(layer_id) - (N_LAYERS / 2);
      sig_term = signature[(lane_idx + (int'(block_id) * 3)) & 5'h1f] ? 5 : -5;

      unique case (block_id)
        BLOCK_GATE: begin
          weight_term = (lane_term >>> 1) - row_term + sig_term + head_term -
                        tile_term[2:0];
        end
        BLOCK_UP: begin
          weight_term = row_term + sig_term - head_term + layer_term +
                        tile_term[2:0];
        end
        BLOCK_DOWN: begin
          weight_term = lane_term - row_term + sig_term + head_term -
                        layer_term;
        end
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM: begin
          weight_term = head_term + row_term + sig_term - tile_term[2:0];
        end
        default: begin
          weight_term = sig_term;
        end
      endcase

      projection_weight_scalar = clamp_weight8(ACC_W'(weight_term));
    end
  endfunction

  function automatic logic [SCALE_W-1:0] effective_scale(
    input logic [SCALE_W-1:0] scale_val
  );
    begin
      if (scale_val == '0) begin
        effective_scale = SCALE_W'(32'd65536);
      end else begin
        effective_scale = scale_val;
      end
    end
  endfunction

  assign context_valid_o = (state_q == DDP_READY) || (state_q == DDP_BLOCK) ||
                           (state_q == DDP_APPLY_SEND) || (state_q == DDP_APPLY_WAIT) ||
                           (state_q == DDP_GEMM_SEND) || (state_q == DDP_GEMM_WAIT) ||
                           (state_q == DDP_SILU_SEND) || (state_q == DDP_SILU_SCALE) ||
                           (state_q == DDP_SILU_ACT) || (state_q == DDP_MUL_SEND) ||
                           (state_q == DDP_MUL_WAIT) ||
                           (state_q == DDP_MASK_APPLY) ||
                           (state_q == DDP_SOFTMAX_ARM) || (state_q == DDP_SOFTMAX_SCORE) ||
                           (state_q == DDP_SOFTMAX_SCALE) || (state_q == DDP_SOFTMAX_ACT) ||
                           (state_q == DDP_OUT_SCALE) || (state_q == DDP_OUT_ACT);
  assign embed_scale_ready_o = (state_q == DDP_CAPTURE) && !scale_seen_q;
  assign embed_act_ready_o = (state_q == DDP_CAPTURE) && scale_seen_q;
  assign final_scale_valid_o = (state_q == DDP_OUT_SCALE);
  assign final_act_valid_o = (state_q == DDP_OUT_ACT);
  assign final_scale_o = final_scale_d;
  assign final_act_o = final_act_d;
  assign busy_o = (state_q != DDP_IDLE);

  always_comb begin
    final_scale_d = '0;
    final_scale_d.data = hidden_scale_q;
    final_scale_d.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
    final_scale_d.tag.block_id = BLOCK_FINAL_RMSNORM;
    final_scale_d.tag.gemm_mode = GEMM_NONE;
    final_scale_d.tag.tile_id = '0;
    final_scale_d.tag.token_base = context_token_base_q;
    final_scale_d.tag.seq_count = context_seq_count_q;
    final_scale_d.tag.q_head_id = '0;
    final_scale_d.tag.kv_head_id = '0;
    final_scale_d.tag.elem_count = ELEM_COUNT_W'(D_MODEL);
    final_scale_d.tag.is_last = 1'b1;
    final_scale_d.tag.is_partial = 1'b0;

    final_act_d = '0;
    final_act_d.tag.layer_id = LAYER_ID_W'(N_LAYERS - 1);
    final_act_d.tag.block_id = BLOCK_FINAL_RMSNORM;
    final_act_d.tag.gemm_mode = GEMM_NONE;
    final_act_d.tag.tile_id = final_tile_idx_q;
    final_act_d.tag.token_base = context_token_base_q;
    final_act_d.tag.seq_count = context_seq_count_q;
    final_act_d.tag.q_head_id = '0;
    final_act_d.tag.kv_head_id = '0;
    final_act_d.tag.elem_count = context_elem_count_q;
    final_act_d.tag.is_partial = context_is_partial_q;
    final_act_d.tag.is_last = (final_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1));

    if (final_tile_idx_q < FEATURE_TILE_COUNT) begin
      final_act_d.data = hidden_tiles_q[final_tile_idx_q];
    end
  end

  always_comb begin
    gemm_act_d = '0;
    gemm_act_d.tag.layer_id = active_layer_q;
    gemm_act_d.tag.block_id = active_block_q;
    gemm_act_d.tag.gemm_mode = projection_gemm_mode(active_block_q);
    gemm_act_d.tag.tile_id = apply_tile_idx_q;
    gemm_act_d.tag.token_base = context_token_base_q;
    gemm_act_d.tag.seq_count = context_seq_count_q;
    gemm_act_d.tag.q_head_id = active_q_head_q;
    gemm_act_d.tag.kv_head_id = active_kv_head_q;
    gemm_act_d.tag.elem_count = context_elem_count_q;
    gemm_act_d.tag.is_last = (apply_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1));
    gemm_act_d.tag.is_partial = context_is_partial_q;

    gemm_wt_d = '0;
    gemm_wt_d.tag = gemm_act_d.tag;
    gemm_wt_d.tag.block_id = active_block_q;
    gemm_wt_d.tag.gemm_mode = gemm_act_d.tag.gemm_mode;
    gemm_wt_flat_d = '0;

    gemm_scale_d = '0;
    gemm_scale_d.data = hidden_scale_q;
    gemm_scale_d.tag = gemm_act_d.tag;
    gemm_scale_d.tag.block_id = active_block_q;

    if (((state_q == DDP_GEMM_SEND) || (state_q == DDP_GEMM_WAIT)) &&
        (apply_tile_idx_q < FEATURE_TILE_COUNT)) begin
      if (active_block_q == BLOCK_DOWN) begin
        gemm_act_d.data = ffn_mul_tiles_q[apply_tile_idx_q];
      end else if (active_block_q == BLOCK_O) begin
        gemm_act_d.data = attn_weighted_tiles_q[apply_tile_idx_q];
      end else if (active_block_q == BLOCK_WEIGHTED_SUM) begin
        gemm_act_d.data = attn_prob_tile_q;
        gemm_scale_d.data = attn_prob_scale_q;
      end else begin
        gemm_act_d.data = hidden_tiles_q[apply_tile_idx_q];
      end

      for (int lane = 0; lane < WEIGHT_VECTOR_ELEMS; lane++) begin
        if (lane < context_elem_count_q) begin
          gemm_wt_flat_d[(lane * WEIGHT_W) +: WEIGHT_W] =
            projection_weight_scalar(active_layer_q,
                                     active_block_q,
                                     active_q_head_q,
                                     active_kv_head_q,
                                     apply_signature_q,
                                     apply_tile_idx_q,
                                     lane,
                                     context_seq_count_q);
        end else begin
          gemm_wt_flat_d[(lane * WEIGHT_W) +: WEIGHT_W] = '0;
        end
      end
    end

    gemm_wt_d.data = gemm_wt_flat_d;
  end

  always_comb begin
    apply_scale_d = '0;
    apply_scale_d.data = hidden_scale_q;
    apply_scale_d.tag.layer_id = active_layer_q;
    apply_scale_d.tag.block_id = active_block_q;
    apply_scale_d.tag.gemm_mode = GEMM_NONE;
    apply_scale_d.tag.tile_id = apply_tile_idx_q;
    apply_scale_d.tag.token_base = context_token_base_q;
    apply_scale_d.tag.seq_count = context_seq_count_q;
    apply_scale_d.tag.q_head_id = active_q_head_q;
    apply_scale_d.tag.kv_head_id = active_kv_head_q;
    apply_scale_d.tag.elem_count = context_elem_count_q;
    apply_scale_d.tag.is_last = (apply_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1));
    apply_scale_d.tag.is_partial = context_is_partial_q;

    apply_residual_d = '0;
    apply_residual_d.tag = apply_scale_d.tag;
    apply_residual_d.tag.block_id = active_block_q;

    apply_update_d = '0;
    apply_update_d.tag = apply_residual_d.tag;
    down_gemm_acc_flat_d = down_gemm_acc_q.data;
    apply_residual_flat_d = '0;
    apply_update_flat_d = '0;

    if (((state_q == DDP_APPLY_SEND) || (state_q == DDP_APPLY_WAIT)) &&
        (apply_tile_idx_q < FEATURE_TILE_COUNT)) begin
      for (int lane = 0; lane < ACC_VECTOR_ELEMS; lane++) begin
        logic signed [ACT_W-1:0] current_lane_val;
        logic signed [ACT_W-1:0] neighbor_lane_val;
        logic signed [ACT_W-1:0] mul_lane_val;
        logic signed [ACT_W-1:0] attn_o_lane_val;
        logic signed [ACC_W-1:0] current_acc_val;
        logic signed [ACC_W-1:0] neighbor_acc_val;
        logic signed [ACC_W-1:0] mul_acc_val;
        logic signed [ACC_W-1:0] attn_o_acc_val;
        logic signed [ACC_W-1:0] bias_acc_val;
        logic signed [ACC_W-1:0] update_acc_val;

        current_lane_val = hidden_tiles_q[apply_tile_idx_q][(lane * ACT_W) +: ACT_W];
        neighbor_lane_val = hidden_tiles_q[apply_tile_idx_q]
                                         [(block_neighbor_lane_idx(lane,
                                                                   active_block_q,
                                                                   active_q_head_q,
                                                                   active_kv_head_q,
                                                                   apply_signature_q) * ACT_W) +: ACT_W];
        mul_lane_val = ffn_mul_tiles_q[apply_tile_idx_q][(lane * ACT_W) +: ACT_W];
        attn_o_lane_val = attn_o_tiles_q[apply_tile_idx_q][(lane * ACT_W) +: ACT_W];
        current_acc_val = sext_act8(current_lane_val);
        neighbor_acc_val = sext_act8(neighbor_lane_val);
        mul_acc_val = sext_act8(mul_lane_val);
        attn_o_acc_val = sext_act8(attn_o_lane_val);
        bias_acc_val = block_lane_bias(active_layer_q,
                                       active_block_q,
                                       active_q_head_q,
                                       active_kv_head_q,
                                       apply_signature_q,
                                       apply_tile_idx_q,
                                       lane,
                                       context_seq_count_q);

        if (lane < context_elem_count_q) begin
          unique case (active_block_q)
            BLOCK_RMSNORM1,
            BLOCK_RMSNORM2: begin
              update_acc_val = bias_acc_val - (current_acc_val >>> 2);
            end
            BLOCK_Q,
            BLOCK_K,
            BLOCK_V,
            BLOCK_O,
            BLOCK_GATE,
            BLOCK_UP: begin
              update_acc_val = (neighbor_acc_val >>> 1) + bias_acc_val;
            end
            BLOCK_DOWN: begin
              if (down_apply_from_gemm_q) begin
                update_acc_val = down_gemm_acc_flat_d[(lane * ACC_W) +: ACC_W];
              end else begin
                update_acc_val = mul_acc_val + (bias_acc_val >>> 1);
              end
            end
            BLOCK_ROPE,
            BLOCK_KV_CACHE_WRITE: begin
              update_acc_val = ((neighbor_acc_val - current_acc_val) >>> 1) + bias_acc_val;
            end
            BLOCK_SCORE,
            BLOCK_WEIGHTED_SUM: begin
              update_acc_val = (neighbor_acc_val >>> 1) - (current_acc_val >>> 2) + bias_acc_val;
            end
            BLOCK_CAUSAL_MASK,
            BLOCK_SOFTMAX: begin
              update_acc_val = (bias_acc_val <<< 1) - (neighbor_acc_val >>> 2);
            end
            BLOCK_RESIDUAL1,
            BLOCK_RESIDUAL2,
            BLOCK_REQUANTIZE: begin
              if (active_block_q == BLOCK_RESIDUAL1) begin
                update_acc_val = attn_o_acc_val;
              end else begin
                update_acc_val = (current_acc_val >>> 1) + bias_acc_val;
              end
            end
            BLOCK_SILU,
            BLOCK_GLU_MUL: begin
              update_acc_val = ((current_acc_val * ((lane & 3) + 1)) >>> 3) -
                               (neighbor_acc_val >>> 3) + bias_acc_val;
            end
            default: begin
              update_acc_val = bias_acc_val;
            end
          endcase

          apply_residual_flat_d[(lane * ACC_W) +: ACC_W] = current_acc_val;
          apply_update_flat_d[(lane * ACC_W) +: ACC_W] = update_acc_val;
        end else begin
          apply_residual_flat_d[(lane * ACC_W) +: ACC_W] = '0;
          apply_update_flat_d[(lane * ACC_W) +: ACC_W] = '0;
        end
      end
    end

    apply_residual_d.data = apply_residual_flat_d;
    apply_update_d.data = apply_update_flat_d;
  end

  always_comb begin
    silu_input_scale_d = effective_scale(hidden_scale_q[0]);
    silu_output_scale_d = effective_scale(hidden_scale_q[0]);

    silu_gate_d = '0;
    silu_gate_d.tag.layer_id = active_layer_q;
    silu_gate_d.tag.block_id = BLOCK_GATE;
    silu_gate_d.tag.gemm_mode = GEMM_GATE;
    silu_gate_d.tag.tile_id = apply_tile_idx_q;
    silu_gate_d.tag.token_base = context_token_base_q;
    silu_gate_d.tag.seq_count = context_seq_count_q;
    silu_gate_d.tag.q_head_id = active_q_head_q;
    silu_gate_d.tag.kv_head_id = active_kv_head_q;
    silu_gate_d.tag.elem_count = context_elem_count_q;
    silu_gate_d.tag.is_last = (apply_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1));
    silu_gate_d.tag.is_partial = context_is_partial_q;

    if (apply_tile_idx_q < FEATURE_TILE_COUNT) begin
      silu_gate_d.data = ffn_gate_tiles_q[apply_tile_idx_q];
    end
  end

  always_comb begin
    mul_scale_d = '0;
    mul_scale_d.data = hidden_scale_q;
    mul_scale_d.tag.layer_id = active_layer_q;
    mul_scale_d.tag.block_id = BLOCK_GLU_MUL;
    mul_scale_d.tag.gemm_mode = GEMM_NONE;
    mul_scale_d.tag.tile_id = apply_tile_idx_q;
    mul_scale_d.tag.token_base = context_token_base_q;
    mul_scale_d.tag.seq_count = context_seq_count_q;
    mul_scale_d.tag.q_head_id = active_q_head_q;
    mul_scale_d.tag.kv_head_id = active_kv_head_q;
    mul_scale_d.tag.elem_count = context_elem_count_q;
    mul_scale_d.tag.is_last = (apply_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1));
    mul_scale_d.tag.is_partial = context_is_partial_q;

    mul_silu_d = '0;
    mul_silu_d.tag = mul_scale_d.tag;
    mul_silu_d.tag.block_id = BLOCK_SILU;
    mul_silu_d.data = ffn_silu_tiles_q[apply_tile_idx_q];

    mul_up_d = '0;
    mul_up_d.tag = mul_scale_d.tag;
    mul_up_d.tag.block_id = BLOCK_UP;
    mul_up_d.data = ffn_up_tiles_q[apply_tile_idx_q];
  end

  residual_add u_block_residual_add (
    .ap_clk           (ap_clk),
    .ap_rst_n         (ap_rst_n && !abort_req_i),
    .block_id_i       (active_block_q),
    .residual_valid_i (state_q == DDP_APPLY_SEND),
    .residual_ready_o (apply_residual_ready_w),
    .residual_i       (apply_residual_d),
    .update_valid_i   (state_q == DDP_APPLY_SEND),
    .update_ready_o   (apply_update_ready_w),
    .update_i         (apply_update_d),
    .sum_valid_o      (apply_sum_valid_w),
    .sum_ready_i      (1'b1),
    .sum_o            (apply_sum_w),
    .busy_o           (),
    .done_pulse_o     ()
  );

  shared_gemm_engine u_block_projection_gemm (
    .ap_clk          (ap_clk),
    .ap_rst_n        (ap_rst_n && !abort_req_i),
    .gemm_mode_i     (projection_gemm_mode(active_block_q)),
    .clear_acc_i     (state_q == DDP_GEMM_SEND),
    .mac_valid_i     (state_q == DDP_GEMM_SEND),
    .emit_acc_i      (state_q == DDP_GEMM_SEND),
    .operands_valid_i(gemm_operands_valid_w),
    .operands_ready_o(gemm_operands_ready_w),
    .act_i           (gemm_act_d),
    .wt_i            (gemm_wt_d),
    .acc_valid_o     (gemm_acc_valid_w),
    .acc_ready_i     (state_q == DDP_GEMM_WAIT),
    .acc_o           (gemm_acc_w),
    .busy_o          (gemm_busy_w)
  );

  requantize_unit u_block_projection_requantize (
    .acc_i              (gemm_acc_w),
    .scale_i            (gemm_scale_d),
    .nonnegative_only_i (1'b0),
    .act_o              (gemm_requant_w)
  );

  silu_wrapper u_block_silu (
    .ap_clk        (ap_clk),
    .ap_rst_n      (ap_rst_n && !abort_req_i),
    .gate_valid_i  (state_q == DDP_SILU_SEND),
    .gate_ready_o  (silu_gate_ready_w),
    .gate_i        (silu_gate_d),
    .input_scale_i (silu_input_scale_d),
    .output_scale_i(silu_output_scale_d),
    .scale_valid_o (silu_scale_valid_w),
    .scale_ready_i (state_q == DDP_SILU_SCALE),
    .scale_o       (silu_scale_w),
    .silu_valid_o  (silu_act_valid_w),
    .silu_ready_i  (state_q == DDP_SILU_ACT),
    .silu_o        (silu_act_w),
    .busy_o        (silu_busy_w),
    .done_pulse_o  (silu_done_w)
  );

  elementwise_mul u_block_ffn_mul (
    .ap_clk       (ap_clk),
    .ap_rst_n     (ap_rst_n && !abort_req_i),
    .silu_valid_i (state_q == DDP_MUL_SEND),
    .silu_ready_o (mul_silu_ready_w),
    .silu_i       (mul_silu_d),
    .up_valid_i   (state_q == DDP_MUL_SEND),
    .up_ready_o   (mul_up_ready_w),
    .up_i         (mul_up_d),
    .prod_valid_o (mul_prod_valid_w),
    .prod_ready_i (state_q == DDP_MUL_WAIT),
    .prod_o       (mul_prod_w),
    .busy_o       (mul_busy_w),
    .done_pulse_o (mul_done_w)
  );

  requantize_unit u_block_requantize (
    .acc_i              (apply_sum_w),
    .scale_i            (apply_scale_d),
    .nonnegative_only_i (1'b0),
    .act_o              (apply_requant_w)
  );

  requantize_unit u_block_mul_requantize (
    .acc_i              (mul_prod_w),
    .scale_i            (mul_scale_d),
    .nonnegative_only_i (1'b0),
    .act_o              (mul_requant_w)
  );

  causal_mask_unit u_causal_mask (
    .runtime_mode_i   (runtime_mode_i),
    .query_pos_base_i  (context_token_base_q),
    .key_pos_base_i     (context_token_base_q),
    .query_row_count_i (COUNT_W'(score_query_rows_from_elem(attn_score_acc_q.tag.elem_count))),
    .key_col_count_i    (COUNT_W'(SCORE_K_TILE)),
    .score_i            (attn_score_acc_q),
    .masked_o           (masked_score_comb_w)
  );

  assign softmax_score_valid_w     = (state_q == DDP_SOFTMAX_SCORE) && attn_masked_valid_q;
  assign softmax_score_scale_w     = effective_scale(attn_score_scale_q[0 +: SCALE_W]);
  assign softmax_prob_scale_ready_w = (state_q == DDP_SOFTMAX_SCALE);
  assign softmax_prob_ready_w      = (state_q == DDP_SOFTMAX_ACT);
  assign gemm_operands_valid_w     = (state_q == DDP_GEMM_SEND) &&
                                     ((active_block_q != BLOCK_WEIGHTED_SUM) || attn_prob_valid_q);

  softmax_wrapper u_block_softmax (
    .ap_clk             (ap_clk),
    .ap_rst_n           (ap_rst_n && !abort_req_i),
    .score_valid_i      (softmax_score_valid_w),
    .score_ready_o      (softmax_score_ready_w),
    .score_i            (attn_masked_acc_q),
    .score_scale_i      (softmax_score_scale_w),
    .prob_scale_valid_o (softmax_prob_scale_valid_w),
    .prob_scale_ready_i (softmax_prob_scale_ready_w),
    .prob_scale_o       (softmax_prob_scale_bus_w),
    .prob_valid_o       (softmax_prob_valid_w),
    .prob_ready_i       (softmax_prob_ready_w),
    .prob_o             (softmax_prob_bus_w),
    .busy_o             (softmax_busy_w),
    .done_pulse_o       (softmax_done_w)
  );

  always_ff @(posedge ap_clk) begin
    block_done_o <= 1'b0;
    final_hidden_done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q <= DDP_IDLE;
      scale_seen_q <= 1'b0;
      tile_seen_q <= '0;
      hidden_scale_q <= '0;
      context_token_base_q <= '0;
      context_seq_count_q <= '0;
      context_elem_count_q <= '0;
      context_is_partial_q <= 1'b0;
      context_signature_q <= '0;
      block_countdown_q <= '0;
      layer_block_count_q <= '0;
      final_tile_idx_q <= '0;
      tile_cursor_q <= '0;
      apply_tile_idx_q <= '0;
      apply_stride_q <= '0;
      ffn_tile_anchor_q <= '0;
      ffn_stride_q <= '0;
      attn_o_tile_anchor_q <= '0;
      attn_o_stride_q <= '0;
      down_apply_from_gemm_q <= 1'b0;
      active_layer_q <= '0;
      active_block_q <= BLOCK_NONE;
      active_q_head_q <= '0;
      active_kv_head_q <= '0;
      apply_signature_q <= '0;
      down_gemm_acc_q <= '0;
      attn_score_acc_q   <= '0;
      attn_masked_acc_q  <= '0;
      attn_prob_tile_q   <= '0;
      attn_score_scale_q <= '0;
      attn_prob_scale_q  <= '0;
      attn_weighted_valid_q <= '0;
      attn_o_valid_q <= '0;
      attn_score_valid_q <= 1'b0;
      attn_masked_valid_q <= 1'b0;
      attn_prob_valid_q <= 1'b0;
    end else begin
      if (launch_i) begin
        state_q <= DDP_CAPTURE;
        scale_seen_q <= 1'b0;
        tile_seen_q <= '0;
        hidden_scale_q <= '0;
        context_token_base_q <= '0;
        context_seq_count_q <= '0;
        context_elem_count_q <= '0;
        context_is_partial_q <= 1'b0;
        context_signature_q <= '0;
        block_countdown_q <= '0;
        layer_block_count_q <= '0;
        final_tile_idx_q <= '0;
        tile_cursor_q <= '0;
        apply_tile_idx_q <= '0;
        apply_stride_q <= '0;
        ffn_tile_anchor_q <= '0;
        ffn_stride_q <= '0;
        attn_o_tile_anchor_q <= '0;
        attn_o_stride_q <= '0;
        down_apply_from_gemm_q <= 1'b0;
        active_layer_q <= '0;
        active_block_q <= BLOCK_NONE;
        active_q_head_q <= '0;
        active_kv_head_q <= '0;
        apply_signature_q <= '0;
        down_gemm_acc_q <= '0;
        attn_score_acc_q   <= '0;
        attn_masked_acc_q  <= '0;
        attn_prob_tile_q   <= '0;
        attn_score_scale_q <= '0;
        attn_prob_scale_q  <= '0;
        attn_weighted_valid_q <= '0;
        attn_o_valid_q <= '0;
        attn_score_valid_q <= 1'b0;
        attn_masked_valid_q <= 1'b0;
        attn_prob_valid_q <= 1'b0;
      end else if (abort_req_i) begin
        state_q <= DDP_IDLE;
        scale_seen_q <= 1'b0;
        tile_seen_q <= '0;
        block_countdown_q <= '0;
        layer_block_count_q <= '0;
        final_tile_idx_q <= '0;
        tile_cursor_q <= '0;
        apply_tile_idx_q <= '0;
        apply_stride_q <= '0;
        ffn_tile_anchor_q <= '0;
        ffn_stride_q <= '0;
        attn_o_tile_anchor_q <= '0;
        attn_o_stride_q <= '0;
        down_apply_from_gemm_q <= 1'b0;
        active_layer_q <= '0;
        active_block_q <= BLOCK_NONE;
        active_q_head_q <= '0;
        active_kv_head_q <= '0;
        apply_signature_q <= '0;
        down_gemm_acc_q <= '0;
        attn_score_acc_q   <= '0;
        attn_masked_acc_q  <= '0;
        attn_prob_tile_q   <= '0;
        attn_score_scale_q <= '0;
        attn_prob_scale_q  <= '0;
        attn_weighted_valid_q <= '0;
        attn_o_valid_q <= '0;
        attn_score_valid_q <= 1'b0;
        attn_masked_valid_q <= 1'b0;
        attn_prob_valid_q <= 1'b0;
      end else begin
        if (embed_scale_valid_i && embed_scale_ready_o) begin
          hidden_scale_q <= embed_scale_i.data;
          context_token_base_q <= embed_scale_i.tag.token_base;
          context_seq_count_q <= embed_scale_i.tag.seq_count;
          context_signature_q <= context_signature_q ^
                                 embed_scale_i.data[0] ^
                                 embed_scale_i.data[SCALE_VECTOR_ELEMS-1];
          scale_seen_q <= 1'b1;
        end

        if (embed_act_valid_i && embed_act_ready_o &&
            (embed_act_i.tag.tile_id < FEATURE_TILE_COUNT)) begin
          hidden_tiles_q[embed_act_i.tag.tile_id] <= embed_act_i.data;
          tile_seen_q[embed_act_i.tag.tile_id] <= 1'b1;
          context_token_base_q <= embed_act_i.tag.token_base;
          context_seq_count_q <= embed_act_i.tag.seq_count;
          context_elem_count_q <= embed_act_i.tag.elem_count;
          context_is_partial_q <= embed_act_i.tag.is_partial;
          context_signature_q <= context_signature_q ^
                                 sext_act8(embed_act_i.data[0]) ^
                                 sext_act8(embed_act_i.data[ACT_VECTOR_ELEMS/2]) ^
                                 sext_act8(embed_act_i.data[ACT_VECTOR_ELEMS-1]) ^
                                 {16'd0, embed_act_i.tag.tile_id};
        end

        unique case (state_q)
          DDP_IDLE: begin
            state_q <= DDP_IDLE;
          end

          DDP_CAPTURE: begin
            if (scale_seen_q && (&tile_seen_q)) begin
              state_q <= DDP_READY;
            end
          end

          DDP_READY: begin
            if (block_valid_i && block_start_i) begin
              if (block_id_i == BLOCK_SCORE) begin
                attn_score_valid_q <= 1'b0;
                attn_masked_valid_q <= 1'b0;
                attn_prob_valid_q <= 1'b0;
              end
              if (layer_id_i != active_layer_q) begin
                layer_block_count_q <= '0;
              end
              active_layer_q <= layer_id_i;
              active_block_q <= block_id_i;
              active_q_head_q <= q_head_id_i;
              active_kv_head_q <= kv_head_id_i;
              block_countdown_q <= block_latency(block_id_i, q_head_id_i, context_signature_q);
              state_q <= DDP_BLOCK;
            end
          end

          DDP_BLOCK: begin
            if (block_countdown_q > 4'd1) begin
              block_countdown_q <= block_countdown_q - 1'b1;
            end else begin
              logic [TILE_ID_W-1:0] block_tile_d;
              logic [31:0] block_signature_d;
              logic [TILE_ID_W-1:0] block_stride_d;

              block_countdown_q <= '0;
              if (is_ffn_anchor_block(active_block_q)) begin
                block_tile_d = tile_cursor_q;
              end else if (is_ffn_chain_block(active_block_q)) begin
                block_tile_d = ffn_tile_anchor_q;
              end else if (is_attn_output_anchor_block(active_block_q)) begin
                block_tile_d = tile_cursor_q;
              end else if (is_attn_output_chain_block(active_block_q)) begin
                block_tile_d = attn_o_tile_anchor_q;
              end else begin
                block_tile_d = tile_cursor_q;
              end

              block_signature_d = context_signature_q ^
                                  block_signature(active_layer_q,
                                                  active_block_q,
                                                  active_q_head_q,
                                                  active_kv_head_q,
                                                  runtime_mode_i);
              if (is_ffn_anchor_block(active_block_q)) begin
                block_stride_d = block_tile_stride(active_block_q,
                                                  active_q_head_q,
                                                  active_kv_head_q,
                                                  block_signature_d);
              end else if (is_ffn_chain_block(active_block_q)) begin
                block_stride_d = ffn_stride_q;
              end else if (is_attn_output_anchor_block(active_block_q)) begin
                block_stride_d = block_tile_stride(active_block_q,
                                                  active_q_head_q,
                                                  active_kv_head_q,
                                                  block_signature_d);
              end else if (is_attn_output_chain_block(active_block_q)) begin
                block_stride_d = attn_o_stride_q;
              end else begin
                block_stride_d = block_tile_stride(active_block_q,
                                                  active_q_head_q,
                                                  active_kv_head_q,
                                                  block_signature_d);
              end

              apply_tile_idx_q <= block_tile_d;
              apply_signature_q <= block_signature_d;
              apply_stride_q <= block_stride_d;
              if (is_ffn_anchor_block(active_block_q)) begin
                ffn_tile_anchor_q <= block_tile_d;
                ffn_stride_q <= block_stride_d;
              end
              if (is_attn_output_anchor_block(active_block_q)) begin
                attn_o_tile_anchor_q <= block_tile_d;
                attn_o_stride_q <= block_stride_d;
              end

              if (is_ffn_projection_block(active_block_q)) begin
                state_q <= DDP_GEMM_SEND;
              end else if (active_block_q == BLOCK_CAUSAL_MASK) begin
                state_q <= DDP_MASK_APPLY;
              end else if (active_block_q == BLOCK_SOFTMAX) begin
                state_q <= DDP_SOFTMAX_ARM;
              end else if (active_block_q == BLOCK_SILU) begin
                state_q <= DDP_SILU_SEND;
              end else if (active_block_q == BLOCK_GLU_MUL) begin
                state_q <= DDP_MUL_SEND;
              end else begin
                state_q <= DDP_APPLY_SEND;
              end
            end
          end

          DDP_APPLY_SEND: begin
            if (apply_residual_ready_w && apply_update_ready_w) begin
              state_q <= DDP_APPLY_WAIT;
            end
          end

          DDP_APPLY_WAIT: begin
            if (apply_sum_valid_w) begin
              unique case (active_block_q)
                BLOCK_GATE: begin
                  ffn_gate_tiles_q[apply_tile_idx_q] <= apply_requant_w.data;
                end
                BLOCK_UP: begin
                  ffn_up_tiles_q[apply_tile_idx_q] <= apply_requant_w.data;
                end

                BLOCK_RESIDUAL1: begin
                  hidden_tiles_q[apply_tile_idx_q] <= apply_requant_w.data;
                  attn_o_valid_q[apply_tile_idx_q] <= 1'b0;
                end

                default: begin
                  hidden_tiles_q[apply_tile_idx_q] <= apply_requant_w.data;
                end
              endcase
              if (advance_tile_cursor_on_block(active_block_q)) begin
                tile_cursor_q <= next_tile_cursor(tile_cursor_q, apply_stride_q);
              end
              if (active_block_q == BLOCK_DOWN) begin
                down_apply_from_gemm_q <= 1'b0;
              end
              context_signature_q <= apply_signature_q;
              layer_block_count_q <= layer_block_count_q + 1'b1;
              block_done_o <= 1'b1;

              if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                  (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                final_tile_idx_q <= '0;
                state_q <= DDP_OUT_SCALE;
              end else begin
                state_q <= DDP_READY;
              end
            end
          end

          DDP_GEMM_SEND: begin
            if (gemm_operands_valid_w && gemm_operands_ready_w) begin
              state_q <= DDP_GEMM_WAIT;
            end
          end

          DDP_MASK_APPLY: begin
            if (attn_score_valid_q) begin
              attn_masked_acc_q <= masked_score_comb_w;
              attn_masked_valid_q <= 1'b1;
              attn_score_valid_q <= 1'b0;
              if (advance_tile_cursor_on_block(active_block_q)) begin
                tile_cursor_q <= next_tile_cursor(tile_cursor_q, apply_stride_q);
              end
              context_signature_q <= apply_signature_q;
              layer_block_count_q <= layer_block_count_q + 1'b1;
              block_done_o <= 1'b1;
              if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                  (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                final_tile_idx_q <= '0;
                state_q <= DDP_OUT_SCALE;
              end else begin
                state_q <= DDP_READY;
              end
            end
          end

          DDP_SOFTMAX_ARM: begin
            state_q <= DDP_SOFTMAX_SCORE;
          end

          DDP_SOFTMAX_SCORE: begin
            if (softmax_score_valid_w && softmax_score_ready_w) begin
              attn_masked_valid_q <= 1'b0;
              state_q <= DDP_SOFTMAX_SCALE;
            end
          end

          DDP_SOFTMAX_SCALE: begin
            if (softmax_prob_scale_valid_w && softmax_prob_scale_ready_w) begin
              attn_prob_scale_q <= softmax_prob_scale_bus_w.data;
              state_q <= DDP_SOFTMAX_ACT;
            end
          end

          DDP_SOFTMAX_ACT: begin
            if (softmax_prob_valid_w && softmax_prob_ready_w) begin
              attn_prob_tile_q <= softmax_prob_bus_w.data;
              attn_prob_valid_q <= 1'b1;
              if (advance_tile_cursor_on_block(active_block_q)) begin
                tile_cursor_q <= next_tile_cursor(tile_cursor_q, apply_stride_q);
              end
              context_signature_q <= apply_signature_q;
              layer_block_count_q <= layer_block_count_q + 1'b1;
              block_done_o <= 1'b1;
              if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                  (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                final_tile_idx_q <= '0;
                state_q <= DDP_OUT_SCALE;
              end else begin
                state_q <= DDP_READY;
              end
            end
          end

          DDP_GEMM_WAIT: begin
            if (gemm_acc_valid_w) begin
              unique case (active_block_q)
                BLOCK_SCORE: begin
                  attn_score_acc_q <= gemm_acc_w;
                  attn_score_scale_q <= gemm_scale_d.data;
                  attn_score_valid_q <= 1'b1;
                  if (advance_tile_cursor_on_block(active_block_q)) begin
                    tile_cursor_q <= next_tile_cursor(tile_cursor_q, apply_stride_q);
                  end
                  context_signature_q <= apply_signature_q;
                  layer_block_count_q <= layer_block_count_q + 1'b1;
                  block_done_o <= 1'b1;
                  state_q <= ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                              (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) ?
                             DDP_OUT_SCALE : DDP_READY;
                  if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                      (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                    final_tile_idx_q <= '0;
                  end
                end

                BLOCK_WEIGHTED_SUM: begin
                  attn_weighted_tiles_q[apply_tile_idx_q] <= gemm_requant_w.data;
                  attn_weighted_valid_q[apply_tile_idx_q] <= 1'b1;
                  attn_prob_valid_q <= 1'b0;
                  if (advance_tile_cursor_on_block(active_block_q)) begin
                    tile_cursor_q <= next_tile_cursor(tile_cursor_q, apply_stride_q);
                  end
                  context_signature_q <= apply_signature_q;
                  layer_block_count_q <= layer_block_count_q + 1'b1;
                  block_done_o <= 1'b1;
                  state_q <= ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                              (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) ?
                             DDP_OUT_SCALE : DDP_READY;
                  if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                      (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                    final_tile_idx_q <= '0;
                  end
                end

                BLOCK_GATE: begin
                  ffn_gate_tiles_q[apply_tile_idx_q] <= gemm_requant_w.data;
                  context_signature_q <= apply_signature_q;
                  layer_block_count_q <= layer_block_count_q + 1'b1;
                  block_done_o <= 1'b1;
                  state_q <= ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                              (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) ?
                             DDP_OUT_SCALE : DDP_READY;
                  if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                      (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                    final_tile_idx_q <= '0;
                  end
                end

                BLOCK_UP: begin
                  ffn_up_tiles_q[apply_tile_idx_q] <= gemm_requant_w.data;
                  context_signature_q <= apply_signature_q;
                  layer_block_count_q <= layer_block_count_q + 1'b1;
                  block_done_o <= 1'b1;
                  state_q <= ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                              (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) ?
                             DDP_OUT_SCALE : DDP_READY;
                  if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                      (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                    final_tile_idx_q <= '0;
                  end
                end

                BLOCK_DOWN: begin
                  down_gemm_acc_q <= gemm_acc_w;
                  down_apply_from_gemm_q <= 1'b1;
                  state_q <= DDP_APPLY_SEND;
                end

                BLOCK_O: begin
                  attn_o_tiles_q[apply_tile_idx_q] <= gemm_requant_w.data;
                  attn_o_valid_q[apply_tile_idx_q] <= 1'b1;
                  attn_weighted_valid_q[apply_tile_idx_q] <= 1'b0;
                  context_signature_q <= apply_signature_q;
                  layer_block_count_q <= layer_block_count_q + 1'b1;
                  block_done_o <= 1'b1;
                  state_q <= ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                              (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) ?
                             DDP_OUT_SCALE : DDP_READY;
                  if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                      (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                    final_tile_idx_q <= '0;
                  end
                end

                default: begin
                  state_q <= DDP_READY;
                end
              endcase
            end
          end

          DDP_SILU_SEND: begin
            if (silu_gate_ready_w) begin
              state_q <= DDP_SILU_SCALE;
            end
          end

          DDP_SILU_SCALE: begin
            if (silu_scale_valid_w) begin
              state_q <= DDP_SILU_ACT;
            end
          end

          DDP_SILU_ACT: begin
            if (silu_act_valid_w) begin
              ffn_silu_tiles_q[apply_tile_idx_q] <= silu_act_w.data;
              context_signature_q <= apply_signature_q;
              layer_block_count_q <= layer_block_count_q + 1'b1;
              block_done_o <= 1'b1;
              if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                  (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                final_tile_idx_q <= '0;
                state_q <= DDP_OUT_SCALE;
              end else begin
                state_q <= DDP_READY;
              end
            end
          end

          DDP_MUL_SEND: begin
            if (mul_silu_ready_w && mul_up_ready_w) begin
              state_q <= DDP_MUL_WAIT;
            end
          end

          DDP_MUL_WAIT: begin
            if (mul_prod_valid_w) begin
              ffn_mul_tiles_q[apply_tile_idx_q] <= mul_requant_w.data;
              context_signature_q <= apply_signature_q;
              layer_block_count_q <= layer_block_count_q + 1'b1;
              block_done_o <= 1'b1;
              if ((active_layer_q == LAYER_ID_W'(N_LAYERS - 1)) &&
                  (layer_block_count_q == (BLOCKS_PER_LAYER - 1))) begin
                final_tile_idx_q <= '0;
                state_q <= DDP_OUT_SCALE;
              end else begin
                state_q <= DDP_READY;
              end
            end
          end

          DDP_OUT_SCALE: begin
            if (final_scale_valid_o && final_scale_ready_i) begin
              final_tile_idx_q <= '0;
              state_q <= DDP_OUT_ACT;
            end
          end

          DDP_OUT_ACT: begin
            if (final_act_valid_o && final_act_ready_i) begin
              if (final_tile_idx_q == TILE_ID_W'(FEATURE_TILE_COUNT - 1)) begin
                final_hidden_done_pulse_o <= 1'b1;
                state_q <= DDP_IDLE;
                scale_seen_q <= 1'b0;
                tile_seen_q <= '0;
                final_tile_idx_q <= '0;
                tile_cursor_q <= '0;
                active_block_q <= BLOCK_NONE;
              end else begin
                final_tile_idx_q <= final_tile_idx_q + 1'b1;
              end
            end
          end

          default: begin
            state_q <= DDP_IDLE;
          end
        endcase
      end
    end
  end

endmodule
