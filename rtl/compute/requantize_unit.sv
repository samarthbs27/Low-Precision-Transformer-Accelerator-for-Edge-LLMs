import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module requantize_unit (
  input  acc_bus_t acc_i,
  input  scale_bus_t scale_i,
  input  logic      nonnegative_only_i,
  output act_bus_t  act_o
);

  localparam int unsigned SCALE_FRAC_W = 16;
  localparam longint unsigned ROUND_HALF = 16'd32768;

  act_bus_t act_bus_d;
  logic [ELEM_COUNT_W-1:0] effective_elem_count_d;
  logic                    partial_tile_d;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] quantized_data_w;

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

  function automatic logic [BANK_ID_W-1:0] bank_index_from_lane(
    input int unsigned lane_idx
  );
    begin
      bank_index_from_lane = BANK_ID_W'(lane_idx / BANK_SLICE_INT8);
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] requantize_scalar(
    input logic signed [ACC_W-1:0] acc_val,
    input logic [SCALE_W-1:0]      scale_val,
    input logic                    nonnegative_only
  );
    longint signed   product;
    longint signed   scale_ext;
    longint signed   rounded_signed;
    longint unsigned abs_product;
    longint unsigned quotient_mag;
    longint unsigned rounded_mag;
    logic [SCALE_FRAC_W-1:0] remainder_bits;
    begin
      scale_ext   = $signed({1'b0, scale_val});
      product     = $signed(acc_val) * scale_ext;
      if (product < 0) begin
        abs_product = -product;
      end else begin
        abs_product = product;
      end
      quotient_mag  = abs_product >> SCALE_FRAC_W;
      remainder_bits = abs_product[SCALE_FRAC_W-1:0];
      rounded_mag    = quotient_mag;

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

      if (nonnegative_only) begin
        if (rounded_signed < 0) begin
          requantize_scalar = '0;
        end else if (rounded_signed > 127) begin
          requantize_scalar = 8'sd127;
        end else begin
          requantize_scalar = ACT_W'(rounded_signed);
        end
      end else begin
        if (rounded_signed > 127) begin
          requantize_scalar = 8'sd127;
        end else if (rounded_signed < -127) begin
          requantize_scalar = -8'sd127;
        end else begin
          requantize_scalar = ACT_W'(rounded_signed);
        end
      end
    end
  endfunction

  assign effective_elem_count_d = effective_elem_count(acc_i.tag.elem_count);
  assign partial_tile_d         = (effective_elem_count_d != ACT_VECTOR_ELEMS);
  assign act_o                  = act_bus_d;

  always @* begin
    act_bus_d            = '0;
    act_bus_d.tag        = acc_i.tag;
    act_bus_d.tag.elem_count = effective_elem_count_d;
    act_bus_d.tag.is_partial = partial_tile_d;
    act_bus_d.data       = quantized_data_w;
  end

  generate
    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_requant_lane
      localparam int unsigned SCALE_IDX = lane / BANK_SLICE_INT8;
      assign quantized_data_w[lane] =
        (lane < effective_elem_count_d) ?
          requantize_scalar(acc_i.data[lane], scale_i.data[SCALE_IDX], nonnegative_only_i) :
          '0;
    end
  endgenerate

endmodule
