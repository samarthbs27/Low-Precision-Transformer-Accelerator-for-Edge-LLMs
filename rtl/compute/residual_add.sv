import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module residual_add (
  input  logic      ap_clk,
  input  logic      ap_rst_n,
  input  block_id_e block_id_i,
  input  logic      residual_valid_i,
  output logic      residual_ready_o,
  input  acc_bus_t  residual_i,
  input  logic      update_valid_i,
  output logic      update_ready_o,
  input  acc_bus_t  update_i,
  output logic      sum_valid_o,
  input  logic      sum_ready_i,
  output acc_bus_t  sum_o,
  output logic      busy_o,
  output logic      done_pulse_o
);

  acc_bus_t residual_q;
  acc_bus_t update_q;
  logic     residual_captured_q;
  logic     update_captured_q;
  acc_bus_t sum_d;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] sum_data_w;
  logic [ELEM_COUNT_W-1:0] effective_elem_count_w;

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] lhs_elem_count,
    input logic [ELEM_COUNT_W-1:0] rhs_elem_count
  );
    begin
      if (lhs_elem_count != '0) begin
        effective_elem_count = lhs_elem_count;
      end else if (rhs_elem_count != '0) begin
        effective_elem_count = rhs_elem_count;
      end else begin
        effective_elem_count = ELEM_COUNT_W'(ACC_VECTOR_ELEMS);
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

  assign residual_ready_o = !residual_captured_q;
  assign update_ready_o   = !update_captured_q;
  assign sum_valid_o      = residual_captured_q && update_captured_q;
  assign busy_o           = residual_captured_q || update_captured_q;
  assign sum_o            = sum_d;

  assign effective_elem_count_w =
    effective_elem_count(residual_q.tag.elem_count, update_q.tag.elem_count);

  always_comb begin
    sum_d = '0;
    sum_d.tag = residual_q.tag;
    sum_d.tag.block_id = block_id_i;
    sum_d.tag.gemm_mode = GEMM_NONE;
    sum_d.tag.elem_count = effective_elem_count_w;
    sum_d.tag.is_partial = (effective_elem_count_w != ACC_VECTOR_ELEMS);
    sum_d.data = sum_data_w;
  end

`ifndef SYNTHESIS
  always_comb begin
    if (residual_captured_q && update_captured_q && !tags_compatible(residual_q.tag, update_q.tag)) begin
      $error("residual_add requires matching tags on both inputs");
    end
  end
`endif

  generate
    for (genvar lane = 0; lane < ACC_VECTOR_ELEMS; lane++) begin : g_sum_lane
      // Residual accumulation is an unsaturated INT32 add by contract.
      assign sum_data_w[lane] =
        (lane < effective_elem_count_w) ?
          (residual_q.data[lane] + update_q.data[lane]) :
          '0;
    end
  endgenerate

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      residual_q          <= '0;
      update_q            <= '0;
      residual_captured_q <= 1'b0;
      update_captured_q   <= 1'b0;
    end else begin
      if (residual_valid_i && residual_ready_o) begin
        residual_q          <= residual_i;
        residual_captured_q <= 1'b1;
      end

      if (update_valid_i && update_ready_o) begin
        update_q          <= update_i;
        update_captured_q <= 1'b1;
      end

      if (sum_valid_o && sum_ready_i) begin
        residual_captured_q <= 1'b0;
        update_captured_q   <= 1'b0;
        done_pulse_o        <= 1'b1;
      end
    end
  end

endmodule
