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

  function automatic act_bus_t normalize_act_tag(
    input act_bus_t   act_bus,
    input gemm_mode_e gemm_mode
  );
    act_bus_t act_tmp;
    begin
      act_tmp               = act_bus;
      act_tmp.tag.block_id  = block_from_mode(gemm_mode);
      act_tmp.tag.gemm_mode = gemm_mode;
      normalize_act_tag     = act_tmp;
    end
  endfunction

  function automatic wt_bus_t act_as_weight(
    input act_bus_t   act_bus,
    input gemm_mode_e gemm_mode
  );
    wt_bus_t wt_tmp;
    begin
      wt_tmp      = '0;
      wt_tmp.data = act_bus.data;
      wt_tmp.tag  = act_bus.tag;
      wt_tmp.tag.block_id  = block_from_mode(gemm_mode);
      wt_tmp.tag.gemm_mode = gemm_mode;
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
        act_o            = normalize_act_tag(act_i, gemm_mode_i);
        wt_o             = weight_i;
        wt_o.tag.block_id  = block_from_mode(gemm_mode_i);
        wt_o.tag.gemm_mode = gemm_mode_i;
      end

      GEMM_SCORE: begin
        operands_valid_o = act_valid_i && kv_valid_i;
        act_ready_o      = operands_ready_i && kv_valid_i;
        kv_ready_o       = operands_ready_i && act_valid_i;
        act_o            = normalize_act_tag(act_i, gemm_mode_i);
        wt_o             = act_as_weight(kv_i, gemm_mode_i);
      end

      GEMM_WEIGHTED_SUM: begin
        operands_valid_o = score_valid_i && kv_valid_i;
        score_ready_o    = operands_ready_i && kv_valid_i;
        kv_ready_o       = operands_ready_i && score_valid_i;
        act_o            = normalize_act_tag(score_i, gemm_mode_i);
        wt_o             = act_as_weight(kv_i, gemm_mode_i);
      end

      default: begin
        operands_valid_o = 1'b0;
      end
    endcase
  end

endmodule
