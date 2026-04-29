// Synthesis-only stub for rmsnorm_core_hls_ip.
// Matches port interface of the sim model in rmsnorm_core_hls_ip.sv.
// Replace with Vitis HLS-synthesized netlist before taping out.
import tinyllama_pkg::*;

module rmsnorm_core_hls_ip (
  input  logic                                   ap_clk,
  input  logic                                   ap_rst_n,
  input  logic                                   start_i,
  input  logic [COUNT_W-1:0]                     row_count_i,
  input  logic [15:0]                            feature_count_i,
  input  logic [31:0]                            epsilon_q16_i,
  input  logic                                   act_valid_i,
  output logic                                   act_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0]   act_chunk_i,
  input  logic                                   gamma_valid_i,
  output logic                                   gamma_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0]   gamma_chunk_i,
  output logic                                   out_valid_o,
  input  logic                                   out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0]   out_chunk_o,
  output logic                                   busy_o,
  output logic                                   done_pulse_o
);

  // 3-state placeholder FSM: accepts chunks until counter wraps, then pulses done.
  // D_MODEL/N_TILE * M_TILE * 2 ≈ 2048 cycles → 11-bit counter.
  typedef enum logic [1:0] { ST_IDLE = 2'd0, ST_RUN = 2'd1, ST_DONE = 2'd2 } stub_st_e;
  stub_st_e   state_q;
  logic [10:0] ctr_q;
  logic signed [(N_TILE * SCALE_W)-1:0] out_reg_q;

  always_ff @(posedge ap_clk or negedge ap_rst_n) begin
    if (!ap_rst_n) begin
      state_q <= ST_IDLE;
      ctr_q   <= '0;
      out_reg_q <= '0;
    end else unique case (state_q)
      ST_IDLE: if (start_i) begin
        state_q <= ST_RUN;
        ctr_q   <= '0;
      end
      ST_RUN: begin
        if (act_valid_i) out_reg_q <= act_chunk_i;
        if (&ctr_q) state_q <= ST_DONE;
        else        ctr_q   <= ctr_q + 1'b1;
      end
      ST_DONE: state_q <= ST_IDLE;
      default: state_q <= ST_IDLE;
    endcase
  end

  assign act_ready_o   = (state_q == ST_RUN);
  assign gamma_ready_o = (state_q == ST_RUN);
  assign out_valid_o   = (state_q == ST_DONE);
  assign out_chunk_o   = out_reg_q;
  assign busy_o        = (state_q == ST_RUN);
  assign done_pulse_o  = (state_q == ST_DONE);

endmodule
