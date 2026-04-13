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

  assign quant_o  = requantized_bus;
  assign score_o  = acc_i;
  assign lmhead_o = acc_i;

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
