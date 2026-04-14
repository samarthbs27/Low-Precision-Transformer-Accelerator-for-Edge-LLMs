import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module gemm_result_router (
  input  gemm_mode_e gemm_mode_i,
  input  logic       acc_valid_i,
  output logic       acc_ready_o,
  input  acc_bus_t   acc_i,
  input  logic       scale_valid_i,
  input  scale_bus_t scale_i,
  output logic       quant_valid_o,
  input  logic       quant_ready_i,
  output act_bus_t   quant_o,
  output logic       score_valid_o,
  input  logic       score_ready_i,
  output acc_bus_t   score_o,
  output logic       lmhead_valid_o,
  input  logic       lmhead_ready_i,
  output acc_bus_t   lmhead_o
);

  act_bus_t requantized_bus;
  act_bus_t quant_bus_d;
  acc_bus_t score_bus_d;
  acc_bus_t lmhead_bus_d;

  function automatic block_id_e block_from_mode(
    input gemm_mode_e gemm_mode
  );
    begin
      unique case (gemm_mode)
        GEMM_SCORE:        block_from_mode = BLOCK_SCORE;
        GEMM_LM_HEAD:      block_from_mode = BLOCK_LM_HEAD;
        GEMM_Q:            block_from_mode = BLOCK_Q;
        GEMM_K:            block_from_mode = BLOCK_K;
        GEMM_V:            block_from_mode = BLOCK_V;
        GEMM_WEIGHTED_SUM: block_from_mode = BLOCK_WEIGHTED_SUM;
        GEMM_O:            block_from_mode = BLOCK_O;
        GEMM_GATE:         block_from_mode = BLOCK_GATE;
        GEMM_UP:           block_from_mode = BLOCK_UP;
        GEMM_DOWN:         block_from_mode = BLOCK_DOWN;
        default:           block_from_mode = BLOCK_NONE;
      endcase
    end
  endfunction

  function automatic logic mode_routes_to_quant(
    input gemm_mode_e gemm_mode
  );
    begin
      unique case (gemm_mode)
        GEMM_Q,
        GEMM_K,
        GEMM_V,
        GEMM_WEIGHTED_SUM,
        GEMM_O,
        GEMM_GATE,
        GEMM_UP,
        GEMM_DOWN: mode_routes_to_quant = 1'b1;
        default:   mode_routes_to_quant = 1'b0;
      endcase
    end
  endfunction

  requantize_unit u_requantize_unit (
    .acc_i(acc_i),
    .scale_i(scale_i),
    .nonnegative_only_i(1'b0),
    .act_o(requantized_bus)
  );

  assign quant_o  = quant_bus_d;
  assign score_o  = score_bus_d;
  assign lmhead_o = lmhead_bus_d;

  always_comb begin
    quant_bus_d               = requantized_bus;
    quant_bus_d.tag.block_id  = block_from_mode(gemm_mode_i);
    quant_bus_d.tag.gemm_mode = gemm_mode_i;
    score_bus_d              = acc_i;
    score_bus_d.tag.block_id = block_from_mode(GEMM_SCORE);
    score_bus_d.tag.gemm_mode = GEMM_SCORE;
    lmhead_bus_d               = acc_i;
    lmhead_bus_d.tag.block_id  = block_from_mode(GEMM_LM_HEAD);
    lmhead_bus_d.tag.gemm_mode = GEMM_LM_HEAD;
  end

  always_comb begin
    quant_valid_o  = 1'b0;
    score_valid_o  = 1'b0;
    lmhead_valid_o = 1'b0;
    acc_ready_o    = 1'b0;

    unique case (gemm_mode_i)
      GEMM_SCORE: begin
        score_valid_o = acc_valid_i;
        acc_ready_o   = score_ready_i;
      end

      GEMM_LM_HEAD: begin
        lmhead_valid_o = acc_valid_i;
        acc_ready_o    = lmhead_ready_i;
      end

      default: begin
        if (mode_routes_to_quant(gemm_mode_i)) begin
          quant_valid_o = acc_valid_i && scale_valid_i;
          acc_ready_o   = quant_ready_i && scale_valid_i;
        end
      end
    endcase
  end

endmodule
