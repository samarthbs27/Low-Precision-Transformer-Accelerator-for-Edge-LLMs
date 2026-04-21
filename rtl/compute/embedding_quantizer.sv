import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module embedding_quantizer (
  input  logic                               ap_clk,
  input  logic                               ap_rst_n,
  input  logic                               row_valid_i,
  output logic                               row_ready_o,
  input  logic [(D_MODEL * 16)-1:0]          row_fp16_i,
  input  token_bus_t                         row_meta_i,
  input  logic                               scale_valid_i,
  output logic                               scale_ready_o,
  input  scale_bus_t                         scale_i,
  output logic                               scale_out_valid_o,
  input  logic                               scale_out_ready_i,
  output scale_bus_t                         scale_out_o,
  output logic                               act_valid_o,
  input  logic                               act_ready_i,
  output act_bus_t                           act_o,
  output logic                               busy_o,
  output logic                               done_pulse_o
);

  localparam int unsigned EMBED_ELEM_W      = 16;
  localparam int unsigned EMBED_ROW_W       = D_MODEL * EMBED_ELEM_W;
  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned QUANT_TILE_W      = N_TILE * ACT_W;

  typedef enum logic [1:0] {
    EQ_IDLE      = 2'd0,
    EQ_QUANTIZE  = 2'd1,
    EQ_OUT_SCALE = 2'd2,
    EQ_OUT_ACT   = 2'd3
  } eq_state_e;

  eq_state_e                               state_q;
  logic                                    scale_captured_q;
  logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] scale_data_q;
  logic [EMBED_ROW_W-1:0]                  current_row_fp16_q;
  logic [QUANT_TILE_W-1:0]                 quant_tile_storage_q [0:M_TILE-1][0:FEATURE_TILE_COUNT-1];
  logic [COUNT_W-1:0]                      batch_row_count_q;
  logic [POS_W-1:0]                        batch_token_base_q;
  logic                                    batch_last_q;
  logic [TILE_ID_W-1:0]                    feature_tile_idx_q;
  logic [COUNT_W-1:0]                      quant_row_slot_q;
  logic [TILE_ID_W-1:0]                    quant_feature_tile_idx_q;
  logic                                    current_row_last_q;
  act_bus_t                                act_bus_d;
  scale_bus_t                              scale_bus_d;
  logic [(ACT_VECTOR_ELEMS * ACT_W)-1:0]   act_data_flat_d;
  wire signed [N_TILE-1:0][ACT_W-1:0]      quant_tile_data_w;

  function automatic logic signed [31:0] fp16_to_q16_16(
    input logic [15:0] fp16_bits
  );
    logic        sign_bit;
    logic [4:0]  exp_bits;
    logic [9:0]  frac_bits;
    logic signed [63:0] abs_val;
    logic signed [63:0] signed_val;
    int           shift_amt;
    logic [63:0]  mantissa;
    logic [63:0]  round_bias;
    begin
      sign_bit = fp16_bits[15];
      exp_bits = fp16_bits[14:10];
      frac_bits = fp16_bits[9:0];

      if ((exp_bits == 5'd0) && (frac_bits == 10'd0)) begin
        fp16_to_q16_16 = '0;
      end else if (exp_bits == 5'd0) begin
        abs_val = ({{54{1'b0}}, frac_bits} + 64'sd128) >>> 8;
        signed_val = sign_bit ? -abs_val : abs_val;
        if (signed_val > 64'sd2147483647) begin
          fp16_to_q16_16 = 32'sh7fff_ffff;
        end else if (signed_val < -64'sd2147483648) begin
          fp16_to_q16_16 = 32'sh8000_0000;
        end else begin
          fp16_to_q16_16 = signed_val[31:0];
        end
      end else if (exp_bits == 5'h1f) begin
        fp16_to_q16_16 = sign_bit ? 32'sh8000_0000 : 32'sh7fff_ffff;
      end else begin
        // This path uses round-half-up on right shifts; it can differ from the
        // banker's-rounding gamma-unpack path by 1 LSB when bits are discarded.
        mantissa = {1'b1, frac_bits};
        shift_amt = $signed({1'b0, exp_bits}) - 9;
        if (shift_amt >= 0) begin
          abs_val = mantissa <<< shift_amt;
        end else begin
          round_bias = 64'sd1 <<< ((-shift_amt) - 1);
          abs_val = (mantissa + round_bias) >>> (-shift_amt);
        end
        signed_val = sign_bit ? -abs_val : abs_val;
        if (signed_val > 64'sd2147483647) begin
          fp16_to_q16_16 = 32'sh7fff_ffff;
        end else if (signed_val < -64'sd2147483648) begin
          fp16_to_q16_16 = 32'sh8000_0000;
        end else begin
          fp16_to_q16_16 = signed_val[31:0];
        end
      end
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] quantize_fixed_lane(
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
        quantize_fixed_lane = 8'sd127;
      end else if (rounded_signed < -127) begin
        quantize_fixed_lane = -8'sd127;
      end else begin
        quantize_fixed_lane = ACT_W'(rounded_signed);
      end
    end
  endfunction

  assign scale_ready_o = !scale_captured_q && (state_q == EQ_IDLE);
  assign row_ready_o = scale_captured_q &&
                       (state_q == EQ_IDLE) &&
                       (batch_row_count_q < M_TILE);
  assign scale_out_valid_o = (state_q == EQ_OUT_SCALE);
  assign act_valid_o = (state_q == EQ_OUT_ACT);
  assign scale_out_o = scale_bus_d;
  assign act_o = act_bus_d;
  assign busy_o = scale_captured_q || (state_q != EQ_IDLE);

  always_comb begin
    scale_bus_d = '0;
    scale_bus_d.tag.layer_id = '0;
    scale_bus_d.tag.block_id = BLOCK_EMBED;
    scale_bus_d.tag.gemm_mode = GEMM_NONE;
    scale_bus_d.tag.tile_id = '0;
    scale_bus_d.tag.token_base = batch_token_base_q;
    scale_bus_d.tag.seq_count = batch_row_count_q;
    scale_bus_d.tag.q_head_id = '0;
    scale_bus_d.tag.kv_head_id = '0;
    scale_bus_d.tag.elem_count = ELEM_COUNT_W'(SCALE_VECTOR_ELEMS);
    scale_bus_d.tag.is_partial = (batch_row_count_q != M_TILE);
    scale_bus_d.tag.is_last = batch_last_q;
    scale_bus_d.data = scale_data_q;

    act_bus_d = '0;
    act_data_flat_d = '0;

    act_bus_d.tag.layer_id = '0;
    act_bus_d.tag.block_id = BLOCK_EMBED;
    act_bus_d.tag.gemm_mode = GEMM_NONE;
    act_bus_d.tag.tile_id = feature_tile_idx_q;
    act_bus_d.tag.token_base = batch_token_base_q;
    act_bus_d.tag.seq_count = batch_row_count_q;
    act_bus_d.tag.q_head_id = '0;
    act_bus_d.tag.kv_head_id = '0;
    act_bus_d.tag.elem_count = batch_row_count_q * N_TILE;
    act_bus_d.tag.is_partial = (batch_row_count_q != M_TILE);
    act_bus_d.tag.is_last = batch_last_q && (feature_tile_idx_q == FEATURE_TILE_COUNT - 1);
    for (int row_local = 0; row_local < M_TILE; row_local++) begin
      if (row_local < batch_row_count_q) begin
        for (int col_local = 0; col_local < N_TILE; col_local++) begin
          act_data_flat_d[(((row_local * N_TILE) + col_local) * ACT_W) +: ACT_W] =
            quant_tile_storage_q[row_local][feature_tile_idx_q][(col_local * ACT_W) +: ACT_W];
        end
      end
    end
    act_bus_d.data = act_data_flat_d;
  end

  generate
    for (genvar quant_lane = 0; quant_lane < N_TILE; quant_lane++) begin : g_quant_lane
      // Quantize one 32-element feature-tile row per cycle, then buffer the
      // INT8 result. This avoids a 512-way divider fanout at output time.
      assign quant_tile_data_w[quant_lane] =
        quantize_fixed_lane(
          fp16_to_q16_16(
            current_row_fp16_q[
              (((quant_feature_tile_idx_q * N_TILE) + quant_lane) * EMBED_ELEM_W) +: EMBED_ELEM_W
            ]
          ),
          scale_data_q[quant_row_slot_q]
        );
    end

  endgenerate

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q            <= EQ_IDLE;
      scale_captured_q   <= 1'b0;
      scale_data_q       <= '0;
      current_row_fp16_q <= '0;
      batch_row_count_q  <= '0;
      batch_token_base_q <= '0;
      batch_last_q       <= 1'b0;
      feature_tile_idx_q <= '0;
      quant_row_slot_q   <= '0;
      quant_feature_tile_idx_q <= '0;
      current_row_last_q <= 1'b0;
    end else begin
      if (scale_valid_i && scale_ready_o) begin
        scale_data_q     <= scale_i.data;
        scale_captured_q <= 1'b1;
      end

      unique case (state_q)
        EQ_IDLE: begin
          if (row_valid_i && row_ready_o) begin
            current_row_fp16_q <= row_fp16_i;
            quant_row_slot_q <= batch_row_count_q;
            quant_feature_tile_idx_q <= '0;
            current_row_last_q <= row_meta_i.tag.is_last;
            if (batch_row_count_q == '0) begin
              batch_token_base_q <= row_meta_i.tag.token_base;
            end
            state_q <= EQ_QUANTIZE;
          end
        end

        EQ_QUANTIZE: begin
          quant_tile_storage_q[quant_row_slot_q][quant_feature_tile_idx_q] <= quant_tile_data_w;
          if (quant_feature_tile_idx_q == FEATURE_TILE_COUNT - 1) begin
            batch_row_count_q <= batch_row_count_q + 1'b1;
            batch_last_q <= current_row_last_q;
            if ((batch_row_count_q + 1'b1 == M_TILE) || current_row_last_q) begin
              feature_tile_idx_q <= '0;
              state_q <= EQ_OUT_SCALE;
            end else begin
              state_q <= EQ_IDLE;
            end
          end else begin
            quant_feature_tile_idx_q <= quant_feature_tile_idx_q + 1'b1;
          end
        end

        EQ_OUT_SCALE: begin
          if (scale_out_valid_o && scale_out_ready_i) begin
            state_q <= EQ_OUT_ACT;
          end
        end

        EQ_OUT_ACT: begin
          if (act_valid_o && act_ready_i) begin
            if (feature_tile_idx_q == FEATURE_TILE_COUNT - 1) begin
              if (batch_last_q) begin
                done_pulse_o <= 1'b1;
                scale_captured_q <= 1'b0;
                batch_row_count_q <= '0;
                batch_token_base_q <= '0;
                batch_last_q <= 1'b0;
                state_q <= EQ_IDLE;
              end else begin
                batch_row_count_q <= '0;
                batch_token_base_q <= batch_token_base_q + POS_W'(batch_row_count_q);
                batch_last_q <= 1'b0;
                state_q <= EQ_IDLE;
              end
              feature_tile_idx_q <= '0;
            end else begin
              feature_tile_idx_q <= feature_tile_idx_q + 1'b1;
            end
          end
        end

        default: begin
          state_q <= EQ_IDLE;
        end
      endcase
    end
  end

endmodule
