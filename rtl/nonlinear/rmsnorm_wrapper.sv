import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module rmsnorm_wrapper (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  block_id_e             block_id_i,
  input  logic                  act_valid_i,
  output logic                  act_ready_o,
  input  act_bus_t              act_i,
  input  logic [SCALE_W-1:0]    input_scale_i,
  input  logic [SCALE_W-1:0]    output_scale_i,
  input  logic                  gamma_valid_i,
  output logic                  gamma_ready_o,
  input  logic [DMA_BEAT_W-1:0] gamma_i,
  input  logic                  gamma_last_i,
  output logic                  scale_valid_o,
  input  logic                  scale_ready_i,
  output scale_bus_t            scale_o,
  output logic                  norm_valid_o,
  input  logic                  norm_ready_i,
  output act_bus_t              norm_o,
  output logic                  busy_o,
  output logic                  done_pulse_o
);

  localparam int unsigned CHUNK_ELEMS        = N_TILE;
  localparam int unsigned CHUNK_W            = CHUNK_ELEMS * SCALE_W;
  localparam int unsigned FEATURE_TILE_COUNT = D_MODEL / N_TILE;
  localparam int unsigned GAMMA_BEAT_COUNT   = D_MODEL / (DMA_BEAT_W / 16);
  // Q16.16 cannot represent 1e-5 exactly; one LSB (1/65536 ~= 1.53e-5) is the
  // nearest nonzero epsilon and is the correct RTL-side encoding to pass.
  localparam logic [SCALE_W-1:0] EPSILON_Q16 = 32'd1;
  localparam logic [15:0] FEATURE_COUNT_C    = 16'd2048;

  typedef enum logic [2:0] {
    RN_IDLE       = 3'd0,
    RN_CAPTURE    = 3'd1,
    RN_SEND_GAMMA = 3'd2,
    RN_SEND_ACT   = 3'd3,
    RN_RECV_CORE  = 3'd4,
    RN_OUT_SCALE  = 3'd5,
    RN_OUT_ACT    = 3'd6
  } rms_state_e;

  rms_state_e state_q;

  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] act_tile_data_q [0:FEATURE_TILE_COUNT-1];
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] norm_tile_data_q [0:FEATURE_TILE_COUNT-1];
  tile_tag_t                                     norm_tile_tag_q [0:FEATURE_TILE_COUNT-1];
  logic [DMA_BEAT_W-1:0]                         gamma_beats_q [0:GAMMA_BEAT_COUNT-1];
  logic [SCALE_W-1:0]                            input_scale_q;
  logic [SCALE_W-1:0]                            output_scale_q;
  tile_tag_t                                     scale_tag_q;
  logic [COUNT_W-1:0]                            row_count_q;
  logic [6:0]                                    act_tile_count_q;
  logic [7:0]                                    gamma_beat_count_q;
  logic                                          act_capture_done_q;
  logic                                          gamma_capture_done_q;
  logic [6:0]                                    send_gamma_idx_q;
  logic [6:0]                                    send_feature_idx_q;
  logic [COUNT_W-1:0]                            send_row_idx_q;
  logic [6:0]                                    recv_feature_idx_q;
  logic [COUNT_W-1:0]                            recv_row_idx_q;
  logic [6:0]                                    out_tile_idx_q;
  logic [6:0]                                    act_capture_idx_w;
  logic [7:0]                                    gamma_capture_idx_w;

  logic                                          core_start_q;
  logic                                          core_busy_w;
  logic                                          core_done_w;
  logic                                          core_act_valid_w;
  logic                                          core_act_ready_w;
  logic signed [CHUNK_W-1:0]                     core_act_chunk_w;
  logic                                          core_gamma_valid_w;
  logic                                          core_gamma_ready_w;
  logic signed [CHUNK_W-1:0]                     core_gamma_chunk_w;
  logic                                          core_out_valid_w;
  logic                                          core_out_ready_w;
  logic signed [CHUNK_W-1:0]                     core_out_chunk_w;

  scale_bus_t                                    scale_bus_d;
  act_bus_t                                      norm_bus_d;
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0]  norm_out_data_w;
  tile_tag_t                                     norm_out_tag_w;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0]  norm_out_unpack_w;
  wire signed [CHUNK_W-1:0]                      gamma_chunk_unpack_w [0:FEATURE_TILE_COUNT-1];

  function automatic logic [COUNT_W-1:0] row_count_from_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [ELEM_COUNT_W-1:0] effective_count;
    begin
      effective_count = (elem_count == '0) ? ELEM_COUNT_W'(ACT_VECTOR_ELEMS) : elem_count;
      row_count_from_elem_count = COUNT_W'((effective_count + N_TILE - 1) / N_TILE);
    end
  endfunction

  function automatic logic signed [31:0] dequantize_act_lane(
    input logic signed [ACT_W-1:0] act_val,
    input logic [SCALE_W-1:0]      scale_val
  );
    longint signed product;
    begin
      product = $signed(act_val) * $signed({1'b0, scale_val});
      if (product > 64'sd2147483647) begin
        dequantize_act_lane = 32'sh7fff_ffff;
      end else if (product < -64'sd2147483648) begin
        dequantize_act_lane = 32'sh8000_0000;
      end else begin
        dequantize_act_lane = product[31:0];
      end
    end
  endfunction

  function automatic logic signed [31:0] fp16_to_q16_16(
    input logic [15:0] fp16_val
  );
    logic sign_bit;
    logic [4:0] exp_bits;
    logic [9:0] frac_bits;
    logic [10:0] mantissa;
    integer shift_amt;
    longint signed mag;
    longint signed rounded_mag;
    longint signed remainder_mask;
    begin
      sign_bit = fp16_val[15];
      exp_bits = fp16_val[14:10];
      frac_bits = fp16_val[9:0];
      mag = 0;

      if (exp_bits == '0) begin
        if (frac_bits != '0) begin
          mag = (frac_bits + 8'd128) >>> 8;
        end
      end else if (exp_bits == 5'h1f) begin
        mag = 32'sh7fff_ffff;
      end else begin
        mantissa = {1'b1, frac_bits};
        shift_amt = integer'(exp_bits) - 15 + 6;
        if (shift_amt >= 0) begin
          mag = mantissa <<< shift_amt;
        end else begin
          remainder_mask = (1 <<< (-shift_amt)) - 1;
          rounded_mag = mantissa >>> (-shift_amt);
          if (((mantissa & remainder_mask) << 1) > (remainder_mask + 1)) begin
            rounded_mag = rounded_mag + 1;
          end else if ((((mantissa & remainder_mask) << 1) == (remainder_mask + 1)) && rounded_mag[0]) begin
            rounded_mag = rounded_mag + 1;
          end
          mag = rounded_mag;
        end
      end

      if (sign_bit) begin
        mag = -mag;
      end

      if (mag > 64'sd2147483647) begin
        fp16_to_q16_16 = 32'sh7fff_ffff;
      end else if (mag < -64'sd2147483648) begin
        fp16_to_q16_16 = 32'sh8000_0000;
      end else begin
        fp16_to_q16_16 = mag[31:0];
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

  function automatic tile_tag_t make_norm_tag(
    input tile_tag_t in_tag,
    input block_id_e block_id
  );
    tile_tag_t tag_d;
    begin
      tag_d = in_tag;
      tag_d.block_id = block_id;
      tag_d.gemm_mode = GEMM_NONE;
      make_norm_tag = tag_d;
    end
  endfunction

  assign act_ready_o        = ((state_q == RN_IDLE) || (state_q == RN_CAPTURE)) && !act_capture_done_q;
  assign gamma_ready_o      = ((state_q == RN_IDLE) || (state_q == RN_CAPTURE)) && !gamma_capture_done_q;
  assign scale_valid_o      = (state_q == RN_OUT_SCALE);
  assign norm_valid_o       = (state_q == RN_OUT_ACT);
  assign norm_o             = norm_bus_d;
  assign scale_o            = scale_bus_d;
  assign busy_o             = (state_q != RN_IDLE);
  assign core_gamma_valid_w = (state_q == RN_SEND_GAMMA);
  assign core_act_valid_w   = (state_q == RN_SEND_ACT);
  assign core_out_ready_w   = (state_q == RN_RECV_CORE);
  assign act_capture_idx_w  = (state_q == RN_IDLE) ? '0 : act_tile_count_q;
  assign gamma_capture_idx_w = (state_q == RN_IDLE) ? '0 : gamma_beat_count_q;

  always_comb begin
    scale_bus_d = '0;
    scale_bus_d.tag = scale_tag_q;
    scale_bus_d.tag.elem_count = ELEM_COUNT_W'(SCALE_VECTOR_ELEMS);
    scale_bus_d.tag.is_partial = 1'b0;
    scale_bus_d.tag.is_last = 1'b1;
    scale_bus_d.data = {SCALE_VECTOR_ELEMS{output_scale_q}};
  end

  always_comb begin
    core_gamma_chunk_w = '0;
    if (state_q == RN_SEND_GAMMA) begin
      core_gamma_chunk_w = gamma_chunk_unpack_w[send_gamma_idx_q];
    end
  end

  always_comb begin
    core_act_chunk_w = '0;
    if (state_q == RN_SEND_ACT) begin
      for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
        core_act_chunk_w[(elem * SCALE_W) +: SCALE_W] =
          dequantize_act_lane(
            act_tile_data_q[send_feature_idx_q][(((send_row_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W],
            input_scale_q
          );
      end
    end
  end

  always_comb begin
    norm_out_data_w = '0;
    norm_out_tag_w = '0;
    for (int tile_idx = 0; tile_idx < FEATURE_TILE_COUNT; tile_idx++) begin
      if (out_tile_idx_q == tile_idx[6:0]) begin
        norm_out_data_w = norm_tile_data_q[tile_idx];
        norm_out_tag_w = norm_tile_tag_q[tile_idx];
      end
    end
    norm_bus_d = '0;
    norm_bus_d.tag = norm_out_tag_w;
    norm_bus_d.data = norm_out_unpack_w;
  end

`ifndef SYNTHESIS
  always_comb begin
    if ((state_q == RN_IDLE) || (state_q == RN_CAPTURE)) begin
      if ((block_id_i != BLOCK_RMSNORM1) &&
          (block_id_i != BLOCK_RMSNORM2) &&
          (block_id_i != BLOCK_FINAL_RMSNORM)) begin
        $error("rmsnorm_wrapper requires RMSNorm block_id input");
      end
    end
  end
`endif

  rmsnorm_core_hls_ip u_rmsnorm_core_hls_ip (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .start_i(core_start_q),
    .row_count_i(row_count_q),
    .feature_count_i(FEATURE_COUNT_C),
    .epsilon_q16_i(EPSILON_Q16),
    .act_valid_i(core_act_valid_w),
    .act_ready_o(core_act_ready_w),
    .act_chunk_i(core_act_chunk_w),
    .gamma_valid_i(core_gamma_valid_w),
    .gamma_ready_o(core_gamma_ready_w),
    .gamma_chunk_i(core_gamma_chunk_w),
    .out_valid_o(core_out_valid_w),
    .out_ready_i(core_out_ready_w),
    .out_chunk_o(core_out_chunk_w),
    .busy_o(core_busy_w),
    .done_pulse_o(core_done_w)
  );

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;
    core_start_q <= 1'b0;

    if (!ap_rst_n) begin
      state_q              <= RN_IDLE;
      input_scale_q        <= '0;
      output_scale_q       <= '0;
      scale_tag_q          <= '0;
      row_count_q          <= '0;
      act_tile_count_q     <= '0;
      gamma_beat_count_q   <= '0;
      act_capture_done_q   <= 1'b0;
      gamma_capture_done_q <= 1'b0;
      send_gamma_idx_q     <= '0;
      send_feature_idx_q   <= '0;
      send_row_idx_q       <= '0;
      recv_feature_idx_q   <= '0;
      recv_row_idx_q       <= '0;
      out_tile_idx_q       <= '0;
      for (int idx = 0; idx < FEATURE_TILE_COUNT; idx++) begin
        act_tile_data_q[idx] <= '0;
        norm_tile_data_q[idx] <= '0;
        norm_tile_tag_q[idx] <= '0;
      end
      for (int idx = 0; idx < GAMMA_BEAT_COUNT; idx++) begin
        gamma_beats_q[idx] <= '0;
      end
    end else begin
      unique case (state_q)
        RN_IDLE: begin
          input_scale_q        <= input_scale_i;
          output_scale_q       <= output_scale_i;
          scale_tag_q          <= '0;
          row_count_q          <= '0;
          act_tile_count_q     <= '0;
          gamma_beat_count_q   <= '0;
          act_capture_done_q   <= 1'b0;
          gamma_capture_done_q <= 1'b0;
          send_gamma_idx_q     <= '0;
          send_feature_idx_q   <= '0;
          send_row_idx_q       <= '0;
          recv_feature_idx_q   <= '0;
          recv_row_idx_q       <= '0;
          out_tile_idx_q       <= '0;

          if (act_valid_i && act_ready_o) begin
            act_tile_data_q['0] <= act_i.data;
            norm_tile_data_q['0] <= '0;
            norm_tile_tag_q['0] <= make_norm_tag(act_i.tag, block_id_i);
            scale_tag_q <= make_norm_tag(act_i.tag, block_id_i);
            row_count_q <= row_count_from_elem_count(act_i.tag.elem_count);
            act_tile_count_q <= 7'd1;
            if (act_i.tag.is_last) begin
              act_capture_done_q <= 1'b1;
            end
          end

          if (gamma_valid_i && gamma_ready_o) begin
            gamma_beats_q['0] <= gamma_i;
            gamma_beat_count_q <= 8'd1;
            if (gamma_last_i) begin
              gamma_capture_done_q <= 1'b1;
            end
          end

          if ((act_valid_i && act_ready_o) || (gamma_valid_i && gamma_ready_o)) begin
            if ((act_valid_i && act_ready_o && act_i.tag.is_last) &&
                (gamma_valid_i && gamma_ready_o && gamma_last_i)) begin
              core_start_q <= 1'b1;
              send_gamma_idx_q <= '0;
              state_q <= RN_SEND_GAMMA;
            end else begin
              state_q <= RN_CAPTURE;
            end
          end
        end

        RN_CAPTURE: begin
          if (act_valid_i && act_ready_o) begin
            act_tile_data_q[act_capture_idx_w] <= act_i.data;
            norm_tile_data_q[act_capture_idx_w] <= '0;
            norm_tile_tag_q[act_capture_idx_w] <= make_norm_tag(act_i.tag, block_id_i);
            row_count_q <= row_count_from_elem_count(act_i.tag.elem_count);
            act_tile_count_q <= act_capture_idx_w + 1'b1;
            if (act_i.tag.is_last || (act_capture_idx_w == (FEATURE_TILE_COUNT - 1))) begin
              act_capture_done_q <= 1'b1;
            end
          end

          if (gamma_valid_i && gamma_ready_o) begin
            gamma_beats_q[gamma_capture_idx_w] <= gamma_i;
            gamma_beat_count_q <= gamma_capture_idx_w + 1'b1;
            if (gamma_last_i || (gamma_capture_idx_w == (GAMMA_BEAT_COUNT - 1))) begin
              gamma_capture_done_q <= 1'b1;
            end
          end

          if ((act_capture_done_q ||
               ((act_valid_i && act_ready_o) &&
                (act_i.tag.is_last || (act_capture_idx_w == (FEATURE_TILE_COUNT - 1))))) &&
              (gamma_capture_done_q ||
               ((gamma_valid_i && gamma_ready_o) &&
                (gamma_last_i || (gamma_capture_idx_w == (GAMMA_BEAT_COUNT - 1)))))) begin
            core_start_q <= 1'b1;
            send_gamma_idx_q <= '0;
            state_q <= RN_SEND_GAMMA;
          end
        end

        RN_SEND_GAMMA: begin
          if (core_gamma_valid_w && core_gamma_ready_w) begin
            if (send_gamma_idx_q == (FEATURE_TILE_COUNT - 1)) begin
              send_feature_idx_q <= '0;
              send_row_idx_q <= '0;
              state_q <= RN_SEND_ACT;
            end else begin
              send_gamma_idx_q <= send_gamma_idx_q + 1'b1;
            end
          end
        end

        RN_SEND_ACT: begin
          if (core_act_valid_w && core_act_ready_w) begin
            if (send_row_idx_q == (row_count_q - 1'b1)) begin
              send_row_idx_q <= '0;
              if (send_feature_idx_q == (FEATURE_TILE_COUNT - 1)) begin
                recv_feature_idx_q <= '0;
                recv_row_idx_q <= '0;
                state_q <= RN_RECV_CORE;
              end else begin
                send_feature_idx_q <= send_feature_idx_q + 1'b1;
              end
            end else begin
              send_row_idx_q <= send_row_idx_q + 1'b1;
            end
          end
        end

        RN_RECV_CORE: begin
          if (core_out_valid_w && core_out_ready_w) begin
            for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
              norm_tile_data_q[recv_feature_idx_q][(((recv_row_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W] <=
                quantize_fixed_lane(core_out_chunk_w[(elem * SCALE_W) +: SCALE_W], output_scale_q);
            end

            if (recv_row_idx_q == (row_count_q - 1'b1)) begin
              recv_row_idx_q <= '0;
              if (recv_feature_idx_q == (FEATURE_TILE_COUNT - 1)) begin
                out_tile_idx_q <= '0;
                state_q <= RN_OUT_SCALE;
              end else begin
                recv_feature_idx_q <= recv_feature_idx_q + 1'b1;
              end
            end else begin
              recv_row_idx_q <= recv_row_idx_q + 1'b1;
            end
          end
        end

        RN_OUT_SCALE: begin
          if (scale_valid_o && scale_ready_i) begin
            state_q <= RN_OUT_ACT;
          end
        end

        RN_OUT_ACT: begin
          if (norm_valid_o && norm_ready_i) begin
            if (out_tile_idx_q == (FEATURE_TILE_COUNT - 1)) begin
              done_pulse_o <= 1'b1;
              state_q <= RN_IDLE;
            end else begin
              out_tile_idx_q <= out_tile_idx_q + 1'b1;
            end
          end
        end

        default: begin
          state_q <= RN_IDLE;
        end
      endcase
    end
  end

  generate
    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_norm_unpack
      assign norm_out_unpack_w[lane] = norm_out_data_w[(lane * ACT_W) +: ACT_W];
    end
    for (genvar chunk = 0; chunk < FEATURE_TILE_COUNT; chunk++) begin : g_gamma_unpack
      for (genvar elem = 0; elem < 16; elem++) begin : g_gamma_elem
        assign gamma_chunk_unpack_w[chunk][(elem * SCALE_W) +: SCALE_W] =
          fp16_to_q16_16(gamma_beats_q[chunk * 2][(elem * 16) +: 16]);
        assign gamma_chunk_unpack_w[chunk][((elem + 16) * SCALE_W) +: SCALE_W] =
          fp16_to_q16_16(gamma_beats_q[(chunk * 2) + 1][(elem * 16) +: 16]);
      end
    end
  endgenerate

endmodule
