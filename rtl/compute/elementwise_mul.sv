import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module elementwise_mul (
  input  logic      ap_clk,
  input  logic      ap_rst_n,
  input  logic      silu_valid_i,
  output logic      silu_ready_o,
  input  act_bus_t  silu_i,
  input  logic      up_valid_i,
  output logic      up_ready_o,
  input  act_bus_t  up_i,
  output logic      prod_valid_o,
  input  logic      prod_ready_i,
  output acc_bus_t  prod_o,
  output logic      busy_o,
  output logic      done_pulse_o
);

  act_bus_t silu_q;
  act_bus_t up_q;
  logic     silu_captured_q;
  logic     up_captured_q;

  acc_bus_t prod_d;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] prod_data_w;
  logic [ELEM_COUNT_W-1:0] effective_elem_count_w;

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

  function automatic logic tags_compatible(
    input tile_tag_t lhs_tag,
    input tile_tag_t rhs_tag
  );
    begin
      tags_compatible =
        (lhs_tag.layer_id == rhs_tag.layer_id) &&
        (lhs_tag.tile_id == rhs_tag.tile_id) &&
        (lhs_tag.token_base == rhs_tag.token_base) &&
        (lhs_tag.seq_count == rhs_tag.seq_count) &&
        (lhs_tag.q_head_id == rhs_tag.q_head_id) &&
        (lhs_tag.kv_head_id == rhs_tag.kv_head_id) &&
        (lhs_tag.elem_count == rhs_tag.elem_count) &&
        (lhs_tag.is_last == rhs_tag.is_last) &&
        (lhs_tag.is_partial == rhs_tag.is_partial);
    end
  endfunction

  assign silu_ready_o = !silu_captured_q;
  assign up_ready_o   = !up_captured_q;
  assign prod_valid_o = silu_captured_q && up_captured_q;
  assign busy_o       = silu_captured_q || up_captured_q;
  assign prod_o       = prod_d;

  assign effective_elem_count_w = effective_elem_count(silu_q.tag.elem_count);

  always_comb begin
    prod_d = '0;
    prod_d.tag = silu_q.tag;
    prod_d.tag.block_id = BLOCK_GLU_MUL;
    prod_d.tag.gemm_mode = GEMM_NONE;
    prod_d.tag.elem_count = effective_elem_count_w;
    prod_d.tag.is_partial = (effective_elem_count_w != ACT_VECTOR_ELEMS);
    prod_d.data = prod_data_w;
  end

`ifndef SYNTHESIS
  always_comb begin
    if (silu_captured_q && up_captured_q && !tags_compatible(silu_q.tag, up_q.tag)) begin
      $error("elementwise_mul requires matching SiLU/up tags");
    end
  end
`endif

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      silu_q          <= '0;
      up_q            <= '0;
      silu_captured_q <= 1'b0;
      up_captured_q   <= 1'b0;
    end else begin
      if (silu_valid_i && silu_ready_o) begin
        silu_q          <= silu_i;
        silu_captured_q <= 1'b1;
      end

      if (up_valid_i && up_ready_o) begin
        up_q          <= up_i;
        up_captured_q <= 1'b1;
      end

      if (prod_valid_o && prod_ready_i) begin
        silu_captured_q <= 1'b0;
        up_captured_q   <= 1'b0;
        done_pulse_o    <= 1'b1;
      end
    end
  end

  generate
    for (genvar lane = 0; lane < ACC_VECTOR_ELEMS; lane++) begin : g_mul_lane
      assign prod_data_w[lane] =
        (lane < effective_elem_count_w) ?
          ACC_W'($signed(silu_q.data[lane]) * $signed(up_q.data[lane])) :
          '0;
    end
  endgenerate

endmodule
