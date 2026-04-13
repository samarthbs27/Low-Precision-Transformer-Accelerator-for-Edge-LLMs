import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module gqa_router (
  input  logic                    select_k_i,
  input  logic                    select_v_i,
  input  logic [Q_HEAD_ID_W-1:0]  q_head_id_i,
  input  logic                    k_valid_i,
  output logic                    k_ready_o,
  input  act_bus_t                k_i,
  input  logic                    v_valid_i,
  output logic                    v_ready_o,
  input  act_bus_t                v_i,
  output logic                    routed_valid_o,
  input  logic                    routed_ready_i,
  output act_bus_t                routed_o,
  output logic                    route_error_o,
  output logic [KV_HEAD_ID_W-1:0] expected_kv_head_o
);

  act_bus_t routed_bus_d;
  logic     selected_valid_d;
  logic     select_error_d;

  function automatic logic [KV_HEAD_ID_W-1:0] expected_kv_head(
    input logic [Q_HEAD_ID_W-1:0] q_head_id
  );
    begin
      expected_kv_head = KV_HEAD_ID_W'(q_head_id / KV_GROUPS);
    end
  endfunction

  assign expected_kv_head_o = expected_kv_head(q_head_id_i);
  assign routed_o           = routed_bus_d;

  always_comb begin
    routed_bus_d    = '0;
    routed_valid_o  = 1'b0;
    route_error_o   = 1'b0;
    k_ready_o       = 1'b0;
    v_ready_o       = 1'b0;
    selected_valid_d = 1'b0;
    select_error_d  = select_k_i && select_v_i;

    unique case ({select_v_i, select_k_i})
      2'b01: begin
        routed_bus_d    = k_i;
        routed_bus_d.tag.block_id  = BLOCK_SCORE;
        routed_bus_d.tag.gemm_mode = GEMM_SCORE;
        routed_bus_d.tag.q_head_id = q_head_id_i;
        routed_bus_d.tag.kv_head_id = expected_kv_head_o;
        selected_valid_d = k_valid_i;
        routed_valid_o   = k_valid_i;
        k_ready_o        = routed_ready_i;
        route_error_o    = k_valid_i && (k_i.tag.kv_head_id != expected_kv_head_o);
      end

      2'b10: begin
        routed_bus_d    = v_i;
        routed_bus_d.tag.block_id  = BLOCK_WEIGHTED_SUM;
        routed_bus_d.tag.gemm_mode = GEMM_WEIGHTED_SUM;
        routed_bus_d.tag.q_head_id = q_head_id_i;
        routed_bus_d.tag.kv_head_id = expected_kv_head_o;
        selected_valid_d = v_valid_i;
        routed_valid_o   = v_valid_i;
        v_ready_o        = routed_ready_i;
        route_error_o    = v_valid_i && (v_i.tag.kv_head_id != expected_kv_head_o);
      end

      default: begin
        routed_bus_d   = '0;
        routed_valid_o = 1'b0;
      end
    endcase

    if (select_error_d) begin
      routed_valid_o = 1'b0;
      k_ready_o      = 1'b0;
      v_ready_o      = 1'b0;
      route_error_o  = 1'b1;
    end

    if (!selected_valid_d && !select_error_d) begin
      route_error_o = 1'b0;
    end
  end

endmodule
