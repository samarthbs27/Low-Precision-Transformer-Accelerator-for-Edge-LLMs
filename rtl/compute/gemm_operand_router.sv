import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module gemm_operand_router (
  input  gemm_mode_e gemm_mode_i,
  input  logic       act_valid_i,
  output logic       act_ready_o,
  input  act_bus_t   act_i,
  input  logic       weight_valid_i,
  output logic       weight_ready_o,
  input  wt_bus_t    weight_i,
  input  logic       score_valid_i,
  output logic       score_ready_o,
  input  act_bus_t   score_i,
  input  logic       kv_valid_i,
  output logic       kv_ready_o,
  input  act_bus_t   kv_i,
  output logic       operands_valid_o,
  input  logic       operands_ready_i,
  output act_bus_t   act_o,
  output wt_bus_t    wt_o
);

  function automatic wt_bus_t act_as_weight(
    input act_bus_t act_bus
  );
    wt_bus_t wt_tmp;
    begin
      wt_tmp      = '0;
      wt_tmp.data = act_bus.data;
      wt_tmp.tag  = act_bus.tag;
      act_as_weight = wt_tmp;
    end
  endfunction

  always @* begin
    act_ready_o      = 1'b0;
    weight_ready_o   = 1'b0;
    score_ready_o    = 1'b0;
    kv_ready_o       = 1'b0;
    operands_valid_o = 1'b0;
    act_o            = '0;
    wt_o             = '0;

    unique case (gemm_mode_i)
      GEMM_Q,
      GEMM_K,
      GEMM_V,
      GEMM_O,
      GEMM_GATE,
      GEMM_UP,
      GEMM_DOWN,
      GEMM_LM_HEAD: begin
        operands_valid_o = act_valid_i && weight_valid_i;
        act_ready_o      = operands_ready_i && weight_valid_i;
        weight_ready_o   = operands_ready_i && act_valid_i;
        act_o            = act_i;
        wt_o             = weight_i;
      end

      GEMM_SCORE: begin
        operands_valid_o = act_valid_i && kv_valid_i;
        act_ready_o      = operands_ready_i && kv_valid_i;
        kv_ready_o       = operands_ready_i && act_valid_i;
        act_o            = act_i;
        wt_o             = act_as_weight(kv_i);
      end

      GEMM_WEIGHTED_SUM: begin
        operands_valid_o = score_valid_i && kv_valid_i;
        score_ready_o    = operands_ready_i && kv_valid_i;
        kv_ready_o       = operands_ready_i && score_valid_i;
        act_o            = score_i;
        wt_o             = act_as_weight(kv_i);
      end

      default: begin
        operands_valid_o = 1'b0;
      end
    endcase
  end

endmodule
