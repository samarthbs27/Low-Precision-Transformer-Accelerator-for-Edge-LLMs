import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module rope_unit (
  input  act_bus_t q_i,
  input  act_bus_t k_i,
  output act_bus_t q_o,
  output act_bus_t k_o
);

  localparam int unsigned SCALE_FRAC_W = 16;
  localparam longint unsigned ROUND_HALF = 16'd32768;

  logic signed [ACT_VECTOR_ELEMS-1:0][SCALE_W-1:0] cos_d;
  logic signed [ACT_VECTOR_ELEMS-1:0][SCALE_W-1:0] sin_d;
  logic [POS_W-1:0]                                 token_base_d;
  logic [COUNT_W-1:0]                               q_token_count_d;
  logic [COUNT_W-1:0]                               k_token_count_d;
  logic [COUNT_W-1:0]                               active_token_count_d;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0]     q_data_w;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0]     k_data_w;
  act_bus_t                                         q_bus_d;
  act_bus_t                                         k_bus_d;

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    begin
      if (elem_count == '0) begin
        effective_elem_count = ELEM_COUNT_W'(ACT_VECTOR_ELEMS);
      end else begin
        effective_elem_count = elem_count;
      end
    end
  endfunction

  function automatic logic [COUNT_W-1:0] token_count_from_tag(
    input tile_tag_t tag
  );
    logic [ELEM_COUNT_W-1:0] elem_count_eff;
    logic [COUNT_W-1:0]      elem_tokens;
    begin
      elem_count_eff = effective_elem_count(tag.elem_count);
      elem_tokens = COUNT_W'((elem_count_eff + HEAD_DIM - 1) / HEAD_DIM);

      if (tag.seq_count != '0) begin
        if (tag.seq_count < ROPE_CHUNK_TOKENS) begin
          token_count_from_tag = tag.seq_count;
        end else begin
          token_count_from_tag = COUNT_W'(ROPE_CHUNK_TOKENS);
        end
      end else if (elem_tokens < ROPE_CHUNK_TOKENS) begin
        token_count_from_tag = elem_tokens;
      end else begin
        token_count_from_tag = COUNT_W'(ROPE_CHUNK_TOKENS);
      end
    end
  endfunction

  function automatic logic [COUNT_W-1:0] min_token_count(
    input logic [COUNT_W-1:0] lhs,
    input logic [COUNT_W-1:0] rhs
  );
    begin
      if (lhs < rhs) begin
        min_token_count = lhs;
      end else begin
        min_token_count = rhs;
      end
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] rotate_scalar(
    input logic signed [ACT_W-1:0] curr_val,
    input logic signed [ACT_W-1:0] pair_val,
    input logic signed [SCALE_W-1:0] cos_val,
    input logic signed [SCALE_W-1:0] sin_val,
    input logic                      lower_half
  );
    longint signed curr_term;
    longint signed pair_term;
    longint signed sum_term;
    longint signed rounded_signed;
    longint unsigned abs_sum;
    longint unsigned quotient_mag;
    longint unsigned rounded_mag;
    logic [SCALE_FRAC_W-1:0] remainder_bits;
    begin
      curr_term = $signed(curr_val) * $signed(cos_val);
      pair_term = $signed(pair_val) * $signed(sin_val);
      if (lower_half) begin
        sum_term = curr_term - pair_term;
      end else begin
        sum_term = curr_term + pair_term;
      end

      if (sum_term < 0) begin
        abs_sum = -sum_term;
      end else begin
        abs_sum = sum_term;
      end

      quotient_mag = abs_sum >> SCALE_FRAC_W;
      remainder_bits = abs_sum[SCALE_FRAC_W-1:0];
      rounded_mag = quotient_mag;

      if (remainder_bits > ROUND_HALF[SCALE_FRAC_W-1:0]) begin
        rounded_mag = quotient_mag + 1;
      end else if ((remainder_bits == ROUND_HALF[SCALE_FRAC_W-1:0]) && quotient_mag[0]) begin
        rounded_mag = quotient_mag + 1;
      end

      if (sum_term < 0) begin
        rounded_signed = -rounded_mag;
      end else begin
        rounded_signed = rounded_mag;
      end

      if (rounded_signed > 127) begin
        rotate_scalar = 8'sd127;
      end else if (rounded_signed < -127) begin
        rotate_scalar = -8'sd127;
      end else begin
        rotate_scalar = ACT_W'(rounded_signed);
      end
    end
  endfunction

  assign q_token_count_d     = token_count_from_tag(q_i.tag);
  assign k_token_count_d     = token_count_from_tag(k_i.tag);
  assign token_base_d        = q_i.tag.token_base;
  assign active_token_count_d = min_token_count(q_token_count_d, k_token_count_d);
  assign q_o                 = q_bus_d;
  assign k_o                 = k_bus_d;

`ifndef SYNTHESIS
  always_comb begin
    if (q_i.tag.token_base != k_i.tag.token_base) begin
      $error("rope_unit requires matching Q/K token_base tags");
    end
  end
`endif

  rope_lut_rom u_rope_lut_rom (
    .token_base_i(token_base_d),
    .token_count_i(active_token_count_d),
    .cos_o(cos_d),
    .sin_o(sin_d)
  );

  always_comb begin
    q_bus_d = '0;
    k_bus_d = '0;

    q_bus_d.tag = q_i.tag;
    k_bus_d.tag = k_i.tag;
    q_bus_d.tag.block_id = BLOCK_ROPE;
    k_bus_d.tag.block_id = BLOCK_ROPE;
    q_bus_d.tag.gemm_mode = GEMM_Q;
    k_bus_d.tag.gemm_mode = GEMM_K;
    // Q and K for one RoPE slice must share position tags; the simulation-only
    // check above makes this integration contract explicit.
    q_bus_d.tag.seq_count = active_token_count_d;
    k_bus_d.tag.seq_count = active_token_count_d;
    q_bus_d.tag.elem_count = ELEM_COUNT_W'(active_token_count_d * HEAD_DIM);
    k_bus_d.tag.elem_count = ELEM_COUNT_W'(active_token_count_d * HEAD_DIM);
    q_bus_d.tag.is_partial = (active_token_count_d != ROPE_CHUNK_TOKENS);
    k_bus_d.tag.is_partial = (active_token_count_d != ROPE_CHUNK_TOKENS);
    q_bus_d.data = q_data_w;
    k_bus_d.data = k_data_w;
  end

  generate
    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_rope_lane
      localparam int unsigned TOKEN_LOCAL = lane / HEAD_DIM;
      localparam int unsigned DIM_LOCAL   = lane % HEAD_DIM;
      localparam bit LOWER_HALF = (DIM_LOCAL < ROPE_HALF_DIM);
      localparam int unsigned PAIR_DIM =
        LOWER_HALF ? (DIM_LOCAL + ROPE_HALF_DIM) : (DIM_LOCAL - ROPE_HALF_DIM);
      localparam int unsigned PAIR_LANE = (TOKEN_LOCAL * HEAD_DIM) + PAIR_DIM;

      assign q_data_w[lane] =
        (TOKEN_LOCAL < active_token_count_d) ?
          rotate_scalar(
            q_i.data[lane],
            q_i.data[PAIR_LANE],
            cos_d[lane],
            sin_d[lane],
            LOWER_HALF
          ) :
          '0;

      assign k_data_w[lane] =
        (TOKEN_LOCAL < active_token_count_d) ?
          rotate_scalar(
            k_i.data[lane],
            k_i.data[PAIR_LANE],
            cos_d[lane],
            sin_d[lane],
            LOWER_HALF
          ) :
          '0;
    end
  endgenerate

endmodule
