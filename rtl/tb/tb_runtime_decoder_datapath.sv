`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_runtime_decoder_datapath;

  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned EXPECTED_BLOCKS_PER_LAYER = 6 + (4 * N_Q_HEADS) + 12;
  localparam int unsigned EXPECTED_TOTAL_BLOCKS_MAX = N_LAYERS * EXPECTED_BLOCKS_PER_LAYER;
  localparam int unsigned EXPECTED_TOTAL_BLOCKS_MIN = N_LAYERS * (EXPECTED_BLOCKS_PER_LAYER - 1);
  localparam int unsigned SCALE_FRAC_W = 16;
  localparam longint unsigned ROUND_HALF = 16'd32768;
  localparam string QUANT_BASE_DEFAULT =
    "sim/golden_traces/phase6/rtl/phase6_decode_embedding_quantizer_batch0";

  logic clk;
  logic rst_n;
  logic launch;
  logic abort_req;
  logic embed_scale_valid;
  logic embed_scale_ready;
  scale_bus_t embed_scale_bus;
  logic embed_act_valid;
  logic embed_act_ready;
  act_bus_t embed_act_bus;

  logic layer_start;
  logic layer_busy;
  logic layer_run_done;
  logic layer_ctx_valid;
  logic block_valid;
  logic block_start;
  runtime_mode_e layer_runtime_mode;
  logic [LAYER_ID_W-1:0] layer_id;
  logic [LAYER_ID_W-1:0] weight_layer_sel;
  logic [LAYER_ID_W-1:0] kv_layer_sel;
  block_id_e layer_block_id;
  logic [Q_HEAD_ID_W-1:0] q_head_id;
  logic [KV_HEAD_ID_W-1:0] kv_head_id;

  logic context_valid;
  logic block_done;
  logic final_scale_valid;
  logic final_scale_ready;
  scale_bus_t final_scale_bus;
  logic final_act_valid;
  logic final_act_ready;
  act_bus_t final_act_bus;
  logic final_hidden_done_pulse;
  logic decoder_busy;

  logic [31:0] quant_meta_mem [0:1];
  logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_mem [0:0];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] act_tiles_mem [0:FEATURE_TILE_COUNT-1];

  int unsigned cycle_count;
  int unsigned block_start_count;
  int unsigned block_done_count;
  int unsigned first_block_start_cycle;
  int unsigned first_block_done_cycle;
  logic        saw_first_block_start;
  logic        saw_first_block_done;
  logic        saw_final_scale;
  logic        saw_layer_run_done;
  scale_bus_t  captured_final_scale;
  int          captured_final_tile_count;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] captured_final_tiles [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_seed_tiles [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_final_tiles [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_gate_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_up_tiles_q   [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_silu_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_mul_tiles_q  [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_weighted_tiles_q [0:FEATURE_TILE_COUNT-1];
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] expected_o_tiles_q        [0:FEATURE_TILE_COUNT-1];
  logic [FEATURE_TILE_COUNT-1:0]         expected_touched_tiles_q;
  logic [31:0]                          expected_signature_seed;
  logic [31:0]                          expected_signature_q;
  logic [TILE_ID_W-1:0]                 expected_tile_cursor_q;
  logic [TILE_ID_W-1:0]                 expected_ffn_tile_anchor_q;
  logic [TILE_ID_W-1:0]                 expected_ffn_stride_q;
  logic [TILE_ID_W-1:0]                 expected_attn_o_tile_anchor_q;
  logic [TILE_ID_W-1:0]                 expected_attn_o_stride_q;
  logic [LAYER_ID_W-1:0]                pending_layer_id_q;
  block_id_e                            pending_block_id_q;
  logic [Q_HEAD_ID_W-1:0]               pending_q_head_id_q;
  logic [KV_HEAD_ID_W-1:0]              pending_kv_head_id_q;
  runtime_mode_e                        pending_runtime_mode_q;
  logic                                 pending_block_valid_q;
  int unsigned                          mul_done_count;
  logic                                 saw_mul_done;
  int unsigned                          silu_done_count;
  logic                                 saw_silu_done;
  int unsigned                          gate_gemm_count;
  int unsigned                          up_gemm_count;
  int unsigned                          down_gemm_count;
  int unsigned                          o_gemm_count;
  string                                quant_base_path;

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

  function automatic logic signed [31:0] sext_act8(
    input logic signed [ACT_W-1:0] value
  );
    begin
      sext_act8 = {{(32-ACT_W){value[ACT_W-1]}}, value};
    end
  endfunction

  function automatic logic [31:0] tb_block_signature(
    input logic [LAYER_ID_W-1:0]   layer_id_f,
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input runtime_mode_e           runtime_mode_f
  );
    logic [31:0] sig;
    begin
      sig = 32'h0;
      sig[0 +: LAYER_ID_W] = layer_id_f;
      sig[8 +: BLOCK_ID_W] = block_id_f;
      sig[16 +: Q_HEAD_ID_W] = q_head_id_f;
      sig[24 +: KV_HEAD_ID_W] = kv_head_id_f;
      sig[31] = runtime_mode_f;
      tb_block_signature = sig;
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] tb_block_tile_stride(
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input logic [31:0]             signature
  );
    int unsigned stride_d;
    begin
      stride_d = 1 + (((int'(block_id_f) * 3) ^
                       (int'(q_head_id_f) * 5) ^
                       (int'(kv_head_id_f) * 7) ^
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
      tb_block_tile_stride = TILE_ID_W'(stride_d);
    end
  endfunction

  function automatic logic [TILE_ID_W-1:0] tb_next_tile_cursor(
    input logic [TILE_ID_W-1:0] current_tile,
    input logic [TILE_ID_W-1:0] stride
  );
    int unsigned next_d;
    begin
      next_d = (current_tile + stride) % FEATURE_TILE_COUNT;
      tb_next_tile_cursor = TILE_ID_W'(next_d);
    end
  endfunction

  function automatic logic tb_is_ffn_chain_block(
    input block_id_e block_id_f
  );
    begin
      unique case (block_id_f)
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_SILU,
        BLOCK_GLU_MUL,
        BLOCK_DOWN: tb_is_ffn_chain_block = 1'b1;
        default:    tb_is_ffn_chain_block = 1'b0;
      endcase
    end
  endfunction

  function automatic logic tb_is_ffn_anchor_block(
    input block_id_e block_id_f
  );
    begin
      tb_is_ffn_anchor_block = (block_id_f == BLOCK_GATE);
    end
  endfunction

  function automatic logic tb_advance_tile_cursor_on_block(
    input block_id_e block_id_f
  );
    begin
      tb_advance_tile_cursor_on_block =
        (!tb_is_ffn_chain_block(block_id_f) && !tb_is_attn_output_chain_block(block_id_f)) ||
        (block_id_f == BLOCK_DOWN) || (block_id_f == BLOCK_RESIDUAL1);
    end
  endfunction

  function automatic logic tb_is_attn_output_chain_block(
    input block_id_e block_id_f
  );
    begin
      unique case (block_id_f)
        BLOCK_O,
        BLOCK_RESIDUAL1: tb_is_attn_output_chain_block = 1'b1;
        default:         tb_is_attn_output_chain_block = 1'b0;
      endcase
    end
  endfunction

  function automatic logic tb_is_attn_output_anchor_block(
    input block_id_e block_id_f
  );
    begin
      tb_is_attn_output_anchor_block = (block_id_f == BLOCK_O);
    end
  endfunction

  function automatic int unsigned tb_block_neighbor_lane_idx(
    input int unsigned             lane_idx,
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input logic [31:0]             signature
  );
    int unsigned row_idx;
    int unsigned lane_local;
    int unsigned lane_offset;
    begin
      row_idx = lane_idx / N_TILE;
      lane_local = lane_idx % N_TILE;
      lane_offset = 1 + ((int'(block_id_f) +
                          int'(q_head_id_f) +
                          (int'(kv_head_id_f) << 1) +
                          signature[3:0]) % N_TILE);
      tb_block_neighbor_lane_idx = (row_idx * N_TILE) + ((lane_local + lane_offset) % N_TILE);
    end
  endfunction

  function automatic logic signed [ACC_W-1:0] tb_block_lane_bias(
    input logic [LAYER_ID_W-1:0]   layer_id_f,
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
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
      head_term = int'(q_head_id_f) - (int'(kv_head_id_f) <<< 1);
      layer_term = int'(layer_id_f) - (N_LAYERS / 2);
      sig_term = signature[(lane_idx + int'(block_id_f)) & 5'h1f] ? 3 : -3;

      unique case (block_id_f)
        BLOCK_RMSNORM1,
        BLOCK_RMSNORM2:       tb_block_lane_bias = sig_term - row_term + layer_term;
        BLOCK_Q,
        BLOCK_K,
        BLOCK_V:              tb_block_lane_bias = lane_term + head_term + sig_term;
        BLOCK_ROPE,
        BLOCK_KV_CACHE_WRITE: tb_block_lane_bias = sig_term - lane_term + row_term;
        BLOCK_SCORE,
        BLOCK_WEIGHTED_SUM:   tb_block_lane_bias = head_term + (row_term <<< 1) + sig_term;
        BLOCK_CAUSAL_MASK,
        BLOCK_SOFTMAX:        tb_block_lane_bias = (sig_term <<< 1) - row_term - tile_term[2:0];
        BLOCK_O,
        BLOCK_GATE,
        BLOCK_UP,
        BLOCK_DOWN:           tb_block_lane_bias = lane_term - head_term + layer_term + tile_term[2:0];
        BLOCK_RESIDUAL1,
        BLOCK_RESIDUAL2,
        BLOCK_REQUANTIZE:     tb_block_lane_bias = sig_term + tile_term[3:0] - row_term;
        BLOCK_SILU,
        BLOCK_GLU_MUL:        tb_block_lane_bias = lane_term + row_term + sig_term - layer_term;
        default:              tb_block_lane_bias = sig_term + head_term;
      endcase
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] tb_clamp_act8(
    input logic signed [ACC_W-1:0] value
  );
    begin
      if (value > 127) begin
        tb_clamp_act8 = 8'sd127;
      end else if (value < -127) begin
        tb_clamp_act8 = -8'sd127;
      end else begin
        tb_clamp_act8 = ACT_W'(value);
      end
    end
  endfunction

  function automatic logic signed [WEIGHT_W-1:0] tb_clamp_weight8(
    input logic signed [ACC_W-1:0] value
  );
    begin
      if (value > 31) begin
        tb_clamp_weight8 = WEIGHT_W'(8'sd31);
      end else if (value < -31) begin
        tb_clamp_weight8 = WEIGHT_W'(-8'sd31);
      end else begin
        tb_clamp_weight8 = WEIGHT_W'(value);
      end
    end
  endfunction

  function automatic logic signed [WEIGHT_W-1:0] tb_projection_weight_scalar(
    input logic [LAYER_ID_W-1:0]   layer_id_f,
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
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
      head_term = int'(q_head_id_f) - (int'(kv_head_id_f) <<< 1);
      layer_term = int'(layer_id_f) - (N_LAYERS / 2);
      sig_term = signature[(lane_idx + (int'(block_id_f) * 3)) & 5'h1f] ? 5 : -5;

      unique case (block_id_f)
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
        default: begin
          weight_term = sig_term;
        end
      endcase

      tb_projection_weight_scalar = tb_clamp_weight8(ACC_W'(weight_term));
    end
  endfunction

  function automatic logic [SCALE_W-1:0] tb_effective_scale(
    input logic [SCALE_W-1:0] scale_val
  );
    begin
      if (scale_val == '0) begin
        tb_effective_scale = SCALE_W'(32'd65536);
      end else begin
        tb_effective_scale = scale_val;
      end
    end
  endfunction

  function automatic real tb_q16_to_real(
    input logic signed [31:0] value_q16
  );
    begin
      tb_q16_to_real = $itor($signed(value_q16)) / 65536.0;
    end
  endfunction

  function automatic logic signed [31:0] tb_q16_from_real(
    input real value_fp
  );
    real scaled_fp;
    real abs_scaled_fp;
    real frac_fp;
    longint signed floor_mag;
    longint signed rounded_val;
    begin
      scaled_fp = value_fp * 65536.0;
      if (scaled_fp >= 0.0) begin
        floor_mag = $rtoi(scaled_fp);
        frac_fp = scaled_fp - $itor(floor_mag);
        rounded_val = floor_mag;
        if (frac_fp > 0.5000000001) begin
          rounded_val = floor_mag + 1;
        end else if ((frac_fp >= 0.4999999999) &&
                     (frac_fp <= 0.5000000001) &&
                     floor_mag[0]) begin
          rounded_val = floor_mag + 1;
        end
      end else begin
        abs_scaled_fp = -scaled_fp;
        floor_mag = $rtoi(abs_scaled_fp);
        frac_fp = abs_scaled_fp - $itor(floor_mag);
        rounded_val = -floor_mag;
        if (frac_fp > 0.5000000001) begin
          rounded_val = -(floor_mag + 1);
        end else if ((frac_fp >= 0.4999999999) &&
                     (frac_fp <= 0.5000000001) &&
                     floor_mag[0]) begin
          rounded_val = -(floor_mag + 1);
        end
      end

      if (rounded_val > 64'sd2147483647) begin
        tb_q16_from_real = 32'sh7fff_ffff;
      end else if (rounded_val < -64'sd2147483648) begin
        tb_q16_from_real = 32'sh8000_0000;
      end else begin
        tb_q16_from_real = rounded_val[31:0];
      end
    end
  endfunction

  function automatic logic signed [31:0] tb_dequantize_act_lane(
    input logic signed [ACT_W-1:0] act_val,
    input logic [SCALE_W-1:0]      scale_val
  );
    longint signed product;
    begin
      product = $signed(act_val) * $signed({1'b0, scale_val});
      if (product > 64'sd2147483647) begin
        tb_dequantize_act_lane = 32'sh7fff_ffff;
      end else if (product < -64'sd2147483648) begin
        tb_dequantize_act_lane = 32'sh8000_0000;
      end else begin
        tb_dequantize_act_lane = product[31:0];
      end
    end
  endfunction

  function automatic real tb_fixed_sigmoid(
    input real value_fp
  );
    begin
      tb_fixed_sigmoid = 1.0 / (1.0 + $exp(-value_fp));
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] tb_quantize_fixed_lane(
    input logic signed [31:0] value_q16,
    input logic [SCALE_W-1:0] scale_q16
  );
    longint signed numerator_abs;
    longint unsigned denominator;
    longint unsigned quotient_mag;
    longint unsigned rounded_mag;
    longint unsigned remainder_mag;
    longint signed rounded_signed;
    begin
      denominator = (scale_q16 == '0) ? 1 : scale_q16;
      if (value_q16 < 0) begin
        numerator_abs = -value_q16;
      end else begin
        numerator_abs = value_q16;
      end
      quotient_mag = numerator_abs / denominator;
      remainder_mag = numerator_abs % denominator;
      rounded_mag = quotient_mag;

      if ((remainder_mag << 1) > denominator) begin
        rounded_mag = quotient_mag + 1;
      end else if (((remainder_mag << 1) == denominator) && quotient_mag[0]) begin
        rounded_mag = quotient_mag + 1;
      end

      rounded_signed = (value_q16 < 0) ? -rounded_mag : rounded_mag;
      if (rounded_signed > 127) begin
        tb_quantize_fixed_lane = 8'sd127;
      end else if (rounded_signed < -127) begin
        tb_quantize_fixed_lane = -8'sd127;
      end else begin
        tb_quantize_fixed_lane = ACT_W'(rounded_signed);
      end
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] tb_silu_scalar(
    input logic signed [ACT_W-1:0] gate_val,
    input logic [SCALE_W-1:0]      input_scale,
    input logic [SCALE_W-1:0]      output_scale
  );
    logic signed [31:0] gate_q16;
    logic signed [31:0] silu_q16;
    real gate_fp;
    begin
      gate_q16 = tb_dequantize_act_lane(gate_val, input_scale);
      gate_fp = tb_q16_to_real(gate_q16);
      silu_q16 = tb_q16_from_real(gate_fp * tb_fixed_sigmoid(gate_fp));
      tb_silu_scalar = tb_quantize_fixed_lane(silu_q16, output_scale);
    end
  endfunction

  function automatic logic [SCALE_W-1:0] tb_scale_bank_value(
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input int unsigned lane_idx
  );
    int unsigned scale_idx;
    begin
      scale_idx = lane_idx / BANK_SLICE_INT8;
      tb_scale_bank_value = scale_vec[(scale_idx * SCALE_W) +: SCALE_W];
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] tb_requantize_scalar(
    input logic signed [ACC_W-1:0] acc_val,
    input logic [SCALE_W-1:0]      scale_val
  );
    longint signed   product;
    longint signed   rounded_signed;
    longint unsigned abs_product;
    longint unsigned quotient_mag;
    longint unsigned rounded_mag;
    logic [SCALE_FRAC_W-1:0] remainder_bits;
    begin
      product = $signed(acc_val) * $signed({1'b0, scale_val});
      if (product < 0) begin
        abs_product = -product;
      end else begin
        abs_product = product;
      end

      quotient_mag = abs_product >> SCALE_FRAC_W;
      remainder_bits = abs_product[SCALE_FRAC_W-1:0];
      rounded_mag = quotient_mag;

      if (remainder_bits > ROUND_HALF[SCALE_FRAC_W-1:0]) begin
        rounded_mag = quotient_mag + 1;
      end else if ((remainder_bits == ROUND_HALF[SCALE_FRAC_W-1:0]) && quotient_mag[0]) begin
        rounded_mag = quotient_mag + 1;
      end

      if (product < 0) begin
        rounded_signed = -rounded_mag;
      end else begin
        rounded_signed = rounded_mag;
      end

      if (rounded_signed > 127) begin
        tb_requantize_scalar = 8'sd127;
      end else if (rounded_signed < -127) begin
        tb_requantize_scalar = -8'sd127;
      end else begin
        tb_requantize_scalar = ACT_W'(rounded_signed);
      end
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_block_update(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tile_data,
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input logic [TILE_ID_W-1:0] tile_idx,
    input logic [LAYER_ID_W-1:0] layer_id_f,
    input block_id_e             block_id_f,
    input logic [Q_HEAD_ID_W-1:0] q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input logic [31:0] signature,
    input logic [COUNT_W-1:0] seq_count,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    logic signed [ACT_W-1:0] current_lane_val;
    logic signed [ACT_W-1:0] neighbor_lane_val;
    logic signed [ACC_W-1:0] current_acc_val;
    logic signed [ACC_W-1:0] neighbor_acc_val;
    logic signed [ACC_W-1:0] update_acc_val;
    logic signed [ACC_W-1:0] sum_acc_val;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          current_lane_val = tile_data[(lane * ACT_W) +: ACT_W];
          neighbor_lane_val = tile_data[(tb_block_neighbor_lane_idx(lane,
                                                                    block_id_f,
                                                                    q_head_id_f,
                                                                    kv_head_id_f,
                                                                    signature) * ACT_W) +: ACT_W];
          current_acc_val = sext_act8(current_lane_val);
          neighbor_acc_val = sext_act8(neighbor_lane_val);

          unique case (block_id_f)
            BLOCK_RMSNORM1,
            BLOCK_RMSNORM2: begin
              update_acc_val =
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count) -
                (current_acc_val >>> 2);
            end
            BLOCK_Q,
            BLOCK_K,
            BLOCK_V,
            BLOCK_O,
            BLOCK_GATE,
            BLOCK_UP,
            BLOCK_DOWN: begin
              update_acc_val =
                (neighbor_acc_val >>> 1) +
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
            BLOCK_ROPE,
            BLOCK_KV_CACHE_WRITE: begin
              update_acc_val =
                ((neighbor_acc_val - current_acc_val) >>> 1) +
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
            BLOCK_SCORE,
            BLOCK_WEIGHTED_SUM: begin
              update_acc_val =
                (neighbor_acc_val >>> 1) - (current_acc_val >>> 2) +
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
            BLOCK_CAUSAL_MASK,
            BLOCK_SOFTMAX: begin
              update_acc_val =
                (tb_block_lane_bias(layer_id_f,
                                    block_id_f,
                                    q_head_id_f,
                                    kv_head_id_f,
                                    signature,
                                    tile_idx,
                                    lane,
                                    seq_count) <<< 1) -
                (neighbor_acc_val >>> 2);
            end
            BLOCK_RESIDUAL1,
            BLOCK_RESIDUAL2,
            BLOCK_REQUANTIZE: begin
              update_acc_val =
                (current_acc_val >>> 1) +
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
            BLOCK_SILU,
            BLOCK_GLU_MUL: begin
              update_acc_val =
                ((current_acc_val * ((lane & 3) + 1)) >>> 3) -
                (neighbor_acc_val >>> 3) +
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
            default: begin
              update_acc_val =
                tb_block_lane_bias(layer_id_f,
                                   block_id_f,
                                   q_head_id_f,
                                   kv_head_id_f,
                                   signature,
                                   tile_idx,
                                   lane,
                                   seq_count);
            end
          endcase

          sum_acc_val = current_acc_val + update_acc_val;
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_requantize_scalar(sum_acc_val, tb_scale_bank_value(scale_vec, lane));
        end else begin
          out_tile[(lane * ACT_W) +: ACT_W] = '0;
        end
      end
      tb_apply_block_update = out_tile;
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_silu_tile(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] gate_tile,
    input logic [SCALE_W-1:0] input_scale,
    input logic [SCALE_W-1:0] output_scale,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_silu_scalar(gate_tile[(lane * ACT_W) +: ACT_W], input_scale, output_scale);
        end
      end
      tb_apply_silu_tile = out_tile;
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_projection_tile(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] src_tile,
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input logic [TILE_ID_W-1:0]    tile_idx,
    input logic [LAYER_ID_W-1:0]   layer_id_f,
    input block_id_e               block_id_f,
    input logic [Q_HEAD_ID_W-1:0]  q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input logic [31:0]             signature,
    input logic [COUNT_W-1:0]      seq_count,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    logic signed [ACC_W-1:0] proj_acc;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          proj_acc =
            ACC_W'($signed(src_tile[(lane * ACT_W) +: ACT_W]) *
                   $signed(tb_projection_weight_scalar(layer_id_f,
                                                      block_id_f,
                                                      q_head_id_f,
                                                      kv_head_id_f,
                                                      signature,
                                                      tile_idx,
                                                      lane,
                                                      seq_count)));
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_requantize_scalar(proj_acc, tb_scale_bank_value(scale_vec, lane));
        end
      end
      tb_apply_projection_tile = out_tile;
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_glu_mul_tile(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] silu_tile,
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] up_tile,
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    logic signed [ACC_W-1:0] prod_acc;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          prod_acc =
            ACC_W'($signed(silu_tile[(lane * ACT_W) +: ACT_W]) *
                   $signed(up_tile[(lane * ACT_W) +: ACT_W]));
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_requantize_scalar(prod_acc, tb_scale_bank_value(scale_vec, lane));
        end
      end
      tb_apply_glu_mul_tile = out_tile;
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_down_update(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tile_data,
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] mul_tile,
    input logic [TILE_ID_W-1:0] tile_idx,
    input logic [LAYER_ID_W-1:0] layer_id_f,
    input block_id_e              block_id_f,
    input logic [Q_HEAD_ID_W-1:0] q_head_id_f,
    input logic [KV_HEAD_ID_W-1:0] kv_head_id_f,
    input logic [31:0]            signature,
    input logic [COUNT_W-1:0]     seq_count,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    logic signed [ACT_W-1:0] current_lane_val;
    logic signed [ACC_W-1:0] current_acc_val;
    logic signed [ACC_W-1:0] down_proj_acc_val;
    logic signed [ACC_W-1:0] update_acc_val;
    logic signed [ACC_W-1:0] sum_acc_val;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          current_lane_val = tile_data[(lane * ACT_W) +: ACT_W];
          current_acc_val = sext_act8(current_lane_val);
          down_proj_acc_val =
            ACC_W'($signed(mul_tile[(lane * ACT_W) +: ACT_W]) *
                   $signed(tb_projection_weight_scalar(layer_id_f,
                                                      block_id_f,
                                                      q_head_id_f,
                                                      kv_head_id_f,
                                                      signature,
                                                      tile_idx,
                                                      lane,
                                                      seq_count)));
          update_acc_val = down_proj_acc_val;
          sum_acc_val = current_acc_val + update_acc_val;
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_requantize_scalar(sum_acc_val, tb_scale_bank_value(scale_vec, lane));
        end
      end
      tb_apply_down_update = out_tile;
    end
  endfunction

  function automatic logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tb_apply_residual_stage_update(
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] tile_data,
    input logic [(SCALE_VECTOR_ELEMS * SCALE_W)-1:0] scale_vec,
    input logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] update_tile,
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0] out_tile;
    logic signed [ACC_W-1:0] current_acc_val;
    logic signed [ACC_W-1:0] update_acc_val;
    logic signed [ACC_W-1:0] sum_acc_val;
    begin
      out_tile = '0;
      for (int lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin
        if (lane < elem_count) begin
          current_acc_val = sext_act8(tile_data[(lane * ACT_W) +: ACT_W]);
          update_acc_val = sext_act8(update_tile[(lane * ACT_W) +: ACT_W]);
          sum_acc_val = current_acc_val + update_acc_val;
          out_tile[(lane * ACT_W) +: ACT_W] =
            tb_requantize_scalar(sum_acc_val, tb_scale_bank_value(scale_vec, lane));
        end
      end
      tb_apply_residual_stage_update = out_tile;
    end
  endfunction

  runtime_decoder_datapath dut (
    .ap_clk                  (clk),
    .ap_rst_n                (rst_n),
    .launch_i                (launch),
    .abort_req_i             (abort_req),
    .embed_scale_valid_i     (embed_scale_valid),
    .embed_scale_ready_o     (embed_scale_ready),
    .embed_scale_i           (embed_scale_bus),
    .embed_act_valid_i       (embed_act_valid),
    .embed_act_ready_o       (embed_act_ready),
    .embed_act_i             (embed_act_bus),
    .block_valid_i           (block_valid),
    .block_start_i           (block_start),
    .runtime_mode_i          (layer_runtime_mode),
    .layer_id_i              (layer_id),
    .block_id_i              (layer_block_id),
    .q_head_id_i             (q_head_id),
    .kv_head_id_i            (kv_head_id),
    .context_valid_o         (context_valid),
    .block_done_o            (block_done),
    .final_scale_valid_o     (final_scale_valid),
    .final_scale_ready_i     (final_scale_ready),
    .final_scale_o           (final_scale_bus),
    .final_act_valid_o       (final_act_valid),
    .final_act_ready_i       (final_act_ready),
    .final_act_o             (final_act_bus),
    .final_hidden_done_pulse_o(final_hidden_done_pulse),
    .busy_o                  (decoder_busy)
  );

  layer_controller u_layer_controller (
    .ap_clk            (clk),
    .ap_rst_n          (rst_n),
    .start_i           (layer_start),
    .abort_req_i       (abort_req),
    .runtime_mode_i    (MODE_PREFILL),
    .block_done_i      (block_done),
    .busy_o            (layer_busy),
    .run_done_o        (layer_run_done),
    .layer_start_o     (),
    .layer_ctx_valid_o (layer_ctx_valid),
    .block_valid_o     (block_valid),
    .block_start_o     (block_start),
    .runtime_mode_o    (layer_runtime_mode),
    .layer_id_o        (layer_id),
    .weight_layer_sel_o(weight_layer_sel),
    .kv_layer_sel_o    (kv_layer_sel),
    .block_id_o        (layer_block_id),
    .q_head_id_o       (q_head_id),
    .kv_head_id_o      (kv_head_id)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cycle_count <= 0;
      block_start_count <= 0;
      block_done_count <= 0;
      first_block_start_cycle <= 0;
      first_block_done_cycle <= 0;
      saw_first_block_start <= 1'b0;
      saw_first_block_done <= 1'b0;
      saw_final_scale <= 1'b0;
      saw_layer_run_done <= 1'b0;
      captured_final_scale <= '0;
      captured_final_tile_count <= 0;
      expected_touched_tiles_q <= '0;
      expected_signature_q <= expected_signature_seed;
      expected_tile_cursor_q <= '0;
      expected_ffn_tile_anchor_q <= '0;
      expected_ffn_stride_q <= '0;
      expected_attn_o_tile_anchor_q <= '0;
      expected_attn_o_stride_q <= '0;
      pending_layer_id_q <= '0;
      pending_block_id_q <= BLOCK_NONE;
      pending_q_head_id_q <= '0;
      pending_kv_head_id_q <= '0;
      pending_runtime_mode_q <= MODE_PREFILL;
      pending_block_valid_q <= 1'b0;
      mul_done_count <= 0;
      saw_mul_done <= 1'b0;
      silu_done_count <= 0;
      saw_silu_done <= 1'b0;
      gate_gemm_count <= 0;
      up_gemm_count <= 0;
      down_gemm_count <= 0;
      o_gemm_count <= 0;
      for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
        captured_final_tiles[tile_idx] <= '0;
        expected_final_tiles[tile_idx] <= expected_seed_tiles[tile_idx];
        expected_gate_tiles_q[tile_idx] <= '0;
        expected_up_tiles_q[tile_idx] <= '0;
        expected_silu_tiles_q[tile_idx] <= '0;
        expected_mul_tiles_q[tile_idx] <= '0;
        expected_weighted_tiles_q[tile_idx] <= '0;
        expected_o_tiles_q[tile_idx] <= '0;
      end
    end else begin
      if (launch) begin
        expected_touched_tiles_q <= '0;
        expected_signature_q <= expected_signature_seed;
        expected_tile_cursor_q <= '0;
        expected_ffn_tile_anchor_q <= '0;
        expected_ffn_stride_q <= '0;
        expected_attn_o_tile_anchor_q <= '0;
        expected_attn_o_stride_q <= '0;
        pending_layer_id_q <= '0;
        pending_block_id_q <= BLOCK_NONE;
        pending_q_head_id_q <= '0;
        pending_kv_head_id_q <= '0;
        pending_runtime_mode_q <= MODE_PREFILL;
        pending_block_valid_q <= 1'b0;
        mul_done_count <= 0;
        saw_mul_done <= 1'b0;
        silu_done_count <= 0;
        saw_silu_done <= 1'b0;
        gate_gemm_count <= 0;
        up_gemm_count <= 0;
        down_gemm_count <= 0;
        o_gemm_count <= 0;
        for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
          expected_final_tiles[tile_idx] <= expected_seed_tiles[tile_idx];
          expected_gate_tiles_q[tile_idx] <= '0;
          expected_up_tiles_q[tile_idx] <= '0;
          expected_silu_tiles_q[tile_idx] <= '0;
          expected_mul_tiles_q[tile_idx] <= '0;
          expected_weighted_tiles_q[tile_idx] <= '0;
          expected_o_tiles_q[tile_idx] <= '0;
        end
      end

      cycle_count <= cycle_count + 1;

      if (dut.mul_done_w) begin
        mul_done_count <= mul_done_count + 1;
        saw_mul_done <= 1'b1;
      end

      if (dut.silu_done_w) begin
        silu_done_count <= silu_done_count + 1;
        saw_silu_done <= 1'b1;
      end

      if (dut.gemm_acc_valid_w) begin
        unique case (dut.active_block_q)
          BLOCK_GATE: gate_gemm_count <= gate_gemm_count + 1;
          BLOCK_UP:   up_gemm_count <= up_gemm_count + 1;
          BLOCK_DOWN: down_gemm_count <= down_gemm_count + 1;
          BLOCK_O:    o_gemm_count <= o_gemm_count + 1;
          default: begin end
        endcase
      end

      if (block_start) begin
        block_start_count <= block_start_count + 1;
        pending_layer_id_q <= layer_id;
        pending_block_id_q <= layer_block_id;
        pending_q_head_id_q <= q_head_id;
        pending_kv_head_id_q <= kv_head_id;
        pending_runtime_mode_q <= layer_runtime_mode;
        pending_block_valid_q <= 1'b1;
        if (!saw_first_block_start) begin
          saw_first_block_start <= 1'b1;
          first_block_start_cycle <= cycle_count;
        end
      end

      if (block_done) begin
        logic [31:0] expected_apply_signature;
        logic [TILE_ID_W-1:0] expected_apply_tile;
        logic [TILE_ID_W-1:0] expected_stride;

        block_done_count <= block_done_count + 1;
        if (!saw_first_block_done) begin
          saw_first_block_done <= 1'b1;
          first_block_done_cycle <= cycle_count;
        end

        if (!pending_block_valid_q) begin
          $error("runtime_decoder_datapath observed block_done without a matching pending block");
          $finish;
        end

        expected_apply_signature =
          expected_signature_q ^
          tb_block_signature(pending_layer_id_q,
                             pending_block_id_q,
                             pending_q_head_id_q,
                             pending_kv_head_id_q,
                             pending_runtime_mode_q);

        if (tb_is_ffn_anchor_block(pending_block_id_q)) begin
          expected_apply_tile = expected_tile_cursor_q;
          expected_stride =
            tb_block_tile_stride(pending_block_id_q,
                                 pending_q_head_id_q,
                                 pending_kv_head_id_q,
                                 expected_apply_signature);
          expected_ffn_tile_anchor_q <= expected_apply_tile;
          expected_ffn_stride_q <= expected_stride;
        end else if (tb_is_ffn_chain_block(pending_block_id_q)) begin
          expected_apply_tile = expected_ffn_tile_anchor_q;
          expected_stride = expected_ffn_stride_q;
        end else if (tb_is_attn_output_anchor_block(pending_block_id_q)) begin
          expected_apply_tile = expected_tile_cursor_q;
          expected_stride =
            tb_block_tile_stride(pending_block_id_q,
                                 pending_q_head_id_q,
                                 pending_kv_head_id_q,
                                 expected_apply_signature);
          expected_attn_o_tile_anchor_q <= expected_apply_tile;
          expected_attn_o_stride_q <= expected_stride;
        end else if (tb_is_attn_output_chain_block(pending_block_id_q)) begin
          expected_apply_tile = expected_attn_o_tile_anchor_q;
          expected_stride = expected_attn_o_stride_q;
        end else begin
          expected_apply_tile = expected_tile_cursor_q;
          expected_stride =
            tb_block_tile_stride(pending_block_id_q,
                                 pending_q_head_id_q,
                                 pending_kv_head_id_q,
                                 expected_apply_signature);
        end

        unique case (pending_block_id_q)
          BLOCK_WEIGHTED_SUM: begin
            expected_weighted_tiles_q[expected_apply_tile] <=
              tb_apply_block_update(expected_final_tiles[expected_apply_tile],
                                    scale_mem[0],
                                    expected_apply_tile,
                                    pending_layer_id_q,
                                    pending_block_id_q,
                                    pending_q_head_id_q,
                                    pending_kv_head_id_q,
                                    expected_apply_signature,
                                    COUNT_W'(quant_meta_mem[0]),
                                    ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_O: begin
            expected_o_tiles_q[expected_apply_tile] <=
              tb_apply_projection_tile(expected_weighted_tiles_q[expected_apply_tile],
                                       scale_mem[0],
                                       expected_apply_tile,
                                       pending_layer_id_q,
                                       pending_block_id_q,
                                       pending_q_head_id_q,
                                       pending_kv_head_id_q,
                                       expected_apply_signature,
                                       COUNT_W'(quant_meta_mem[0]),
                                       ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_RESIDUAL1: begin
            expected_final_tiles[expected_apply_tile] <=
              tb_apply_residual_stage_update(expected_final_tiles[expected_apply_tile],
                                             scale_mem[0],
                                             expected_o_tiles_q[expected_apply_tile],
                                             ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_GATE: begin
            expected_gate_tiles_q[expected_apply_tile] <=
              tb_apply_projection_tile(expected_final_tiles[expected_apply_tile],
                                       scale_mem[0],
                                       expected_apply_tile,
                                       pending_layer_id_q,
                                       pending_block_id_q,
                                       pending_q_head_id_q,
                                       pending_kv_head_id_q,
                                       expected_apply_signature,
                                       COUNT_W'(quant_meta_mem[0]),
                                       ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_UP: begin
            expected_up_tiles_q[expected_apply_tile] <=
              tb_apply_projection_tile(expected_final_tiles[expected_apply_tile],
                                       scale_mem[0],
                                       expected_apply_tile,
                                       pending_layer_id_q,
                                       pending_block_id_q,
                                       pending_q_head_id_q,
                                       pending_kv_head_id_q,
                                       expected_apply_signature,
                                       COUNT_W'(quant_meta_mem[0]),
                                       ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_SILU: begin
            expected_silu_tiles_q[expected_apply_tile] <=
              tb_apply_silu_tile(expected_gate_tiles_q[expected_apply_tile],
                                 tb_effective_scale(scale_mem[0][0 +: SCALE_W]),
                                 tb_effective_scale(scale_mem[0][0 +: SCALE_W]),
                                 ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_GLU_MUL: begin
            expected_mul_tiles_q[expected_apply_tile] <=
              tb_apply_glu_mul_tile(expected_silu_tiles_q[expected_apply_tile],
                                    expected_up_tiles_q[expected_apply_tile],
                                    scale_mem[0],
                                    ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          BLOCK_DOWN: begin
            expected_final_tiles[expected_apply_tile] <=
              tb_apply_down_update(expected_final_tiles[expected_apply_tile],
                                   scale_mem[0],
                                   expected_mul_tiles_q[expected_apply_tile],
                                   expected_apply_tile,
                                   pending_layer_id_q,
                                   pending_block_id_q,
                                   pending_q_head_id_q,
                                   pending_kv_head_id_q,
                                   expected_apply_signature,
                                   COUNT_W'(quant_meta_mem[0]),
                                   ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end

          default: begin
            expected_final_tiles[expected_apply_tile] <=
              tb_apply_block_update(expected_final_tiles[expected_apply_tile],
                                    scale_mem[0],
                                    expected_apply_tile,
                                    pending_layer_id_q,
                                    pending_block_id_q,
                                    pending_q_head_id_q,
                                    pending_kv_head_id_q,
                                    expected_apply_signature,
                                    COUNT_W'(quant_meta_mem[0]),
                                    ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE));
          end
        endcase

        expected_touched_tiles_q[expected_apply_tile] <= 1'b1;
        expected_signature_q <= expected_apply_signature;
        if (tb_advance_tile_cursor_on_block(pending_block_id_q)) begin
          expected_tile_cursor_q <= tb_next_tile_cursor(expected_tile_cursor_q, expected_stride);
        end
        pending_block_valid_q <= 1'b0;
      end

      if (final_scale_valid && final_scale_ready) begin
        saw_final_scale <= 1'b1;
        captured_final_scale <= final_scale_bus;
      end
      if (layer_run_done) begin
        saw_layer_run_done <= 1'b1;
      end

      if (final_act_valid && final_act_ready &&
          (captured_final_tile_count < FEATURE_TILE_COUNT)) begin
        captured_final_tiles[captured_final_tile_count] <= final_act_bus.data;
        captured_final_tile_count <= captured_final_tile_count + 1;
      end
    end
  end

  assign final_scale_ready = 1'b1;
  assign final_act_ready = 1'b1;

  task automatic drive_scale;
    begin
      embed_scale_bus = '0;
      embed_scale_bus.data = scale_mem[0];
      embed_scale_bus.tag.layer_id = '0;
      embed_scale_bus.tag.block_id = BLOCK_EMBED;
      embed_scale_bus.tag.gemm_mode = GEMM_NONE;
      embed_scale_bus.tag.tile_id = '0;
      embed_scale_bus.tag.token_base = POS_W'(quant_meta_mem[1]);
      embed_scale_bus.tag.seq_count = COUNT_W'(quant_meta_mem[0]);
      embed_scale_bus.tag.q_head_id = '0;
      embed_scale_bus.tag.kv_head_id = '0;
      embed_scale_bus.tag.elem_count = ELEM_COUNT_W'(D_MODEL);
      embed_scale_bus.tag.is_last = 1'b1;
      embed_scale_bus.tag.is_partial = 1'b0;

      embed_scale_valid = 1'b1;
      while (!embed_scale_ready) begin
        @(negedge clk);
      end
      @(negedge clk);
      embed_scale_valid = 1'b0;
    end
  endtask

  task automatic drive_act_tile(
    input int unsigned tile_idx
  );
    begin
      embed_act_bus = '0;
      embed_act_bus.data = act_tiles_mem[tile_idx];
      embed_act_bus.tag.layer_id = '0;
      embed_act_bus.tag.block_id = BLOCK_EMBED;
      embed_act_bus.tag.gemm_mode = GEMM_NONE;
      embed_act_bus.tag.tile_id = TILE_ID_W'(tile_idx);
      embed_act_bus.tag.token_base = POS_W'(quant_meta_mem[1]);
      embed_act_bus.tag.seq_count = COUNT_W'(quant_meta_mem[0]);
      embed_act_bus.tag.q_head_id = '0;
      embed_act_bus.tag.kv_head_id = '0;
      embed_act_bus.tag.elem_count = ELEM_COUNT_W'(quant_meta_mem[0] * N_TILE);
      embed_act_bus.tag.is_last = (tile_idx == FEATURE_TILE_COUNT - 1);
      embed_act_bus.tag.is_partial = (quant_meta_mem[0] != M_TILE);

      embed_act_valid = 1'b1;
      while (!embed_act_ready) begin
        @(negedge clk);
      end
      @(negedge clk);
      embed_act_valid = 1'b0;
    end
  endtask

  initial begin
    int timeout_cycles;
    int plusarg_seen;
    int changed_tile_count;
    string quant_meta_path;
    string quant_scale_path;
    string quant_act_path;

    quant_base_path = QUANT_BASE_DEFAULT;
    plusarg_seen = $value$plusargs("QUANT_BASE=%s", quant_base_path);
    quant_meta_path = {quant_base_path, ".meta.memh"};
    quant_scale_path = {quant_base_path, ".scale_packed.memh"};
    quant_act_path = {quant_base_path, ".act_tiles_expected_packed.memh"};

    if (!file_exists(quant_meta_path) ||
        !file_exists(quant_scale_path) ||
        !file_exists(quant_act_path)) begin
      $fatal(1,
             "tb_runtime_decoder_datapath missing fixture files under QUANT_BASE='%0s'. Run from the repo root or pass +QUANT_BASE=<fixture_base>.",
             quant_base_path);
    end

    for (int idx = 0; idx < 2; idx++) begin
      quant_meta_mem[idx] = 'x;
    end
    scale_mem[0] = 'x;
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      act_tiles_mem[tile_idx] = 'x;
      expected_seed_tiles[tile_idx] = '0;
    end

    $display("tb_runtime_decoder_datapath using QUANT_BASE=%0s", quant_base_path);
    $readmemh(quant_meta_path, quant_meta_mem);
    $readmemh(quant_scale_path, scale_mem);
    $readmemh(quant_act_path, act_tiles_mem);

    if ($isunknown(quant_meta_mem[0]) || $isunknown(quant_meta_mem[1]) ||
        $isunknown(scale_mem[0]) || $isunknown(act_tiles_mem[0])) begin
      $fatal(1,
             "tb_runtime_decoder_datapath fixture load failed for QUANT_BASE='%0s'. Check the working directory or pass +QUANT_BASE=<fixture_base>.",
             quant_base_path);
    end
    if ((quant_meta_mem[0] == 0) || (quant_meta_mem[0] > M_TILE)) begin
      $fatal(1,
             "tb_runtime_decoder_datapath invalid row_count=%0d from fixture base '%0s'.",
             quant_meta_mem[0],
             quant_base_path);
    end

    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      expected_seed_tiles[tile_idx] = act_tiles_mem[tile_idx];
    end

    expected_signature_seed =
      scale_mem[0][0 +: SCALE_W] ^
      scale_mem[0][((SCALE_VECTOR_ELEMS - 1) * SCALE_W) +: SCALE_W];
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      expected_signature_seed =
        expected_signature_seed ^
        sext_act8(act_tiles_mem[tile_idx][0 +: ACT_W]) ^
        sext_act8(act_tiles_mem[tile_idx][((ACT_VECTOR_ELEMS / 2) * ACT_W) +: ACT_W]) ^
        sext_act8(act_tiles_mem[tile_idx][((ACT_VECTOR_ELEMS - 1) * ACT_W) +: ACT_W]) ^
        {16'd0, TILE_ID_W'(tile_idx)};
    end

    clk = 1'b0;
    rst_n = 1'b0;
    launch = 1'b0;
    abort_req = 1'b0;
    embed_scale_valid = 1'b0;
    embed_scale_bus = '0;
    embed_act_valid = 1'b0;
    embed_act_bus = '0;
    layer_start = 1'b0;

    repeat (5) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    launch <= 1'b1;
    @(negedge clk);
    launch <= 1'b0;

    drive_scale();
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      drive_act_tile(tile_idx);
    end

    timeout_cycles = 0;
    while (!context_valid) begin
      @(negedge clk);
      timeout_cycles++;
      if (timeout_cycles > 2000) begin
        $error("runtime_decoder_datapath timed out waiting for captured context");
        $finish;
      end
    end

    layer_start <= 1'b1;
    @(negedge clk);
    layer_start <= 1'b0;

    timeout_cycles = 0;
    while (!final_hidden_done_pulse) begin
      @(negedge clk);
      timeout_cycles++;
      if (timeout_cycles > 200000) begin
        $error("runtime_decoder_datapath timed out waiting for final hidden output");
        $finish;
      end
    end

    repeat (2) @(posedge clk);

    if (!saw_first_block_start || !saw_first_block_done) begin
      $error("runtime_decoder_datapath expected both block_start and block_done activity");
      $finish;
    end
    if (first_block_done_cycle <= first_block_start_cycle) begin
      $error("runtime_decoder_datapath first block completion should lag block start");
      $finish;
    end
    if ((block_start_count != block_done_count) &&
        (block_start_count != (block_done_count + 1))) begin
      $error("runtime_decoder_datapath expected block starts/dones to match within the registered seam, got starts=%0d dones=%0d",
             block_start_count,
             block_done_count);
      $finish;
    end
    if ((block_done_count < EXPECTED_TOTAL_BLOCKS_MIN) ||
        (block_done_count > EXPECTED_TOTAL_BLOCKS_MAX)) begin
      $error("runtime_decoder_datapath expected TinyLlama-scale block count in [%0d, %0d], got %0d",
             EXPECTED_TOTAL_BLOCKS_MIN,
             EXPECTED_TOTAL_BLOCKS_MAX,
             block_done_count);
      $finish;
    end
    if (!saw_final_scale) begin
      $error("runtime_decoder_datapath expected final scale output");
      $finish;
    end
    if (captured_final_scale.data != scale_mem[0] ||
        captured_final_scale.tag.block_id != BLOCK_FINAL_RMSNORM ||
        captured_final_scale.tag.gemm_mode != GEMM_NONE ||
        captured_final_scale.tag.layer_id != LAYER_ID_W'(N_LAYERS - 1) ||
        captured_final_scale.tag.token_base != POS_W'(quant_meta_mem[1]) ||
        captured_final_scale.tag.seq_count != COUNT_W'(quant_meta_mem[0])) begin
      $error("runtime_decoder_datapath final scale mismatch");
      $finish;
    end
    if (captured_final_tile_count != FEATURE_TILE_COUNT) begin
      $error("runtime_decoder_datapath expected %0d final hidden tiles, got %0d",
             FEATURE_TILE_COUNT,
             captured_final_tile_count);
      $finish;
    end
    if (expected_touched_tiles_q != {FEATURE_TILE_COUNT{1'b1}}) begin
      $error("runtime_decoder_datapath expected every feature tile to participate in block-driven updates, touched mask=%0h",
             expected_touched_tiles_q);
      $finish;
    end

    changed_tile_count = 0;
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      if (captured_final_tiles[tile_idx] !== expected_final_tiles[tile_idx]) begin
        $error("runtime_decoder_datapath final hidden tile mismatch at tile %0d", tile_idx);
        $display("expected=%h", expected_final_tiles[tile_idx]);
        $display("actual  =%h", captured_final_tiles[tile_idx]);
        $finish;
      end
      if (captured_final_tiles[tile_idx] !== act_tiles_mem[tile_idx]) begin
        changed_tile_count++;
      end
    end

    if (changed_tile_count == 0) begin
      $error("runtime_decoder_datapath final hidden output still matches the raw embedding input");
      $finish;
    end
    if (!saw_mul_done || (mul_done_count != N_LAYERS)) begin
      $error("runtime_decoder_datapath expected one real GLU_MUL leaf completion per layer, saw_mul_done=%0b mul_done_count=%0d",
             saw_mul_done,
             mul_done_count);
      $finish;
    end
    if (!saw_silu_done || (silu_done_count != N_LAYERS)) begin
      $error("runtime_decoder_datapath expected one real SiLU leaf completion per layer, saw_silu_done=%0b silu_done_count=%0d",
             saw_silu_done,
             silu_done_count);
      $finish;
    end
    if ((gate_gemm_count != N_LAYERS) ||
        (up_gemm_count != N_LAYERS) ||
        (down_gemm_count != N_LAYERS) ||
        (o_gemm_count != N_LAYERS)) begin
      $error("runtime_decoder_datapath expected one real projection GEMM per closed projection block per layer, gate=%0d up=%0d down=%0d o=%0d",
             gate_gemm_count,
             up_gemm_count,
             down_gemm_count,
             o_gemm_count);
      $finish;
    end

    if (decoder_busy || context_valid || final_scale_valid || final_act_valid) begin
      $error("runtime_decoder_datapath should be idle after final hidden emission");
      $finish;
    end

    $display("PASS: tb_runtime_decoder_datapath starts=%0d dones=%0d changed_tiles=%0d gate_gemm=%0d up_gemm=%0d down_gemm=%0d o_gemm=%0d silu_done=%0d mul_done=%0d run_done=%0b",
             block_start_count,
             block_done_count,
             changed_tile_count,
             gate_gemm_count,
             up_gemm_count,
             down_gemm_count,
             o_gemm_count,
             silu_done_count,
             mul_done_count,
             saw_layer_run_done);
    $finish;
  end

endmodule
