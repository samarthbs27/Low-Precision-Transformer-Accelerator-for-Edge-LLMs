import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module shared_gemm_engine (
  input  logic      ap_clk,
  input  logic      ap_rst_n,
  input  gemm_mode_e gemm_mode_i,
  input  logic      clear_acc_i,
  input  logic      mac_valid_i,
  input  logic      emit_acc_i,
  input  logic      operands_valid_i,
  output logic      operands_ready_o,
  input  act_bus_t  act_i,
  input  wt_bus_t   wt_i,
  output logic      acc_valid_o,
  input  logic      acc_ready_i,
  output acc_bus_t  acc_o,
  output logic      busy_o
);

  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0]  acc_base_d;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0]  acc_mac_d;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0]  bank_load_d;
  tile_tag_t                                      resolved_tag_d;
  logic [ELEM_COUNT_W-1:0]                        active_lane_count_d;
  logic                                           load_bank_d;
  logic                                           clear_bank_d;
  logic                                           snapshot_capture_d;
  logic                                           operands_fire_d;
  logic                                           snapshot_valid_q;
  logic                                           dirty_q;
  acc_bus_t                                       bank_acc_q;
  acc_bus_t                                       snapshot_q;

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    begin
      if (elem_count == '0) begin
        effective_elem_count = ELEM_COUNT_W'(ACC_VECTOR_ELEMS);
      end else begin
        effective_elem_count = elem_count;
      end
    end
  endfunction

  function automatic logic [ELEM_COUNT_W-1:0] min_elem_count(
    input logic [ELEM_COUNT_W-1:0] lhs,
    input logic [ELEM_COUNT_W-1:0] rhs
  );
    logic [ELEM_COUNT_W-1:0] lhs_eff;
    logic [ELEM_COUNT_W-1:0] rhs_eff;
    begin
      lhs_eff = effective_elem_count(lhs);
      rhs_eff = effective_elem_count(rhs);
      if (lhs_eff < rhs_eff) begin
        min_elem_count = lhs_eff;
      end else begin
        min_elem_count = rhs_eff;
      end
    end
  endfunction

  function automatic block_id_e block_from_mode(
    input gemm_mode_e gemm_mode
  );
    begin
      unique case (gemm_mode)
        GEMM_Q:            block_from_mode = BLOCK_Q;
        GEMM_K:            block_from_mode = BLOCK_K;
        GEMM_V:            block_from_mode = BLOCK_V;
        GEMM_SCORE:        block_from_mode = BLOCK_SCORE;
        GEMM_WEIGHTED_SUM: block_from_mode = BLOCK_WEIGHTED_SUM;
        GEMM_O:            block_from_mode = BLOCK_O;
        GEMM_GATE:         block_from_mode = BLOCK_GATE;
        GEMM_UP:           block_from_mode = BLOCK_UP;
        GEMM_DOWN:         block_from_mode = BLOCK_DOWN;
        GEMM_LM_HEAD:      block_from_mode = BLOCK_LM_HEAD;
        default:           block_from_mode = BLOCK_NONE;
      endcase
    end
  endfunction

  function automatic tile_tag_t resolved_tag(
    input act_bus_t   act_bus,
    input wt_bus_t    wt_bus,
    input gemm_mode_e gemm_mode
  );
    tile_tag_t tag_tmp;
    logic [ELEM_COUNT_W-1:0] active_count;
    begin
      tag_tmp = act_bus.tag;
      active_count = min_elem_count(act_bus.tag.elem_count, wt_bus.tag.elem_count);
      tag_tmp.block_id   = block_from_mode(gemm_mode);
      tag_tmp.gemm_mode  = gemm_mode;
      tag_tmp.elem_count = active_count;
      tag_tmp.is_partial = (active_count != ACC_VECTOR_ELEMS);
      tag_tmp.is_last    = act_bus.tag.is_last && wt_bus.tag.is_last;
      resolved_tag       = tag_tmp;
    end
  endfunction

  assign operands_ready_o = !snapshot_valid_q;
  assign acc_valid_o      = snapshot_valid_q;
  assign acc_o            = snapshot_q;
  assign busy_o           = dirty_q || snapshot_valid_q;

  assign operands_fire_d   = operands_valid_i && operands_ready_o;
  assign active_lane_count_d = min_elem_count(act_i.tag.elem_count, wt_i.tag.elem_count);
  assign resolved_tag_d    = resolved_tag(act_i, wt_i, gemm_mode_i);
  assign load_bank_d       = operands_fire_d && mac_valid_i;
  assign clear_bank_d      = clear_acc_i && !load_bank_d;
  assign snapshot_capture_d = emit_acc_i && (!snapshot_valid_q || acc_ready_i);

  accumulator_bank u_accumulator_bank (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .clear_i(clear_bank_d),
    .load_i(load_bank_d),
    .load_data_i(bank_load_d),
    .load_tag_i(resolved_tag_d),
    .acc_o(bank_acc_q)
  );

  generate
    for (genvar lane = 0; lane < ACC_VECTOR_ELEMS; lane++) begin : g_lane_math
      assign acc_base_d[lane] =
        clear_acc_i ? '0 : bank_acc_q.data[lane];
      assign acc_mac_d[lane] =
        (load_bank_d && (lane < active_lane_count_d)) ?
          (acc_base_d[lane] + ($signed(act_i.data[lane]) * $signed(wt_i.data[lane]))) :
          acc_base_d[lane];
      assign bank_load_d[lane] = acc_mac_d[lane];
    end
  endgenerate

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      snapshot_q       <= '0;
      snapshot_valid_q <= 1'b0;
      dirty_q          <= 1'b0;
    end else begin
      if (snapshot_valid_q && acc_ready_i) begin
        snapshot_valid_q <= 1'b0;
      end

      if (clear_bank_d) begin
        dirty_q <= 1'b0;
      end

      if (load_bank_d) begin
        dirty_q <= 1'b1;
      end

      if (snapshot_capture_d) begin
        snapshot_q.tag <= load_bank_d ? resolved_tag_d : bank_acc_q.tag;
        snapshot_valid_q <= 1'b1;
        dirty_q <= 1'b0;

        if (load_bank_d) begin
          snapshot_q.data <= bank_load_d;
        end else begin
          snapshot_q.data <= bank_acc_q.data;
        end
      end
    end
  end

endmodule
