import tinyllama_pkg::*;

module layer_controller (
  input  logic                 ap_clk,
  input  logic                 ap_rst_n,
  input  logic                 start_i,
  input  logic                 abort_req_i,
  input  runtime_mode_e        runtime_mode_i,
  input  logic                 layer_step_done_i,
  output logic                 busy_o,
  output logic                 run_done_o,
  output logic                 layer_start_o,
  output logic                 layer_ctx_valid_o,
  output runtime_mode_e        runtime_mode_o,
  output logic [LAYER_ID_W-1:0] layer_id_o,
  output logic [LAYER_ID_W-1:0] weight_layer_sel_o,
  output logic [LAYER_ID_W-1:0] kv_layer_sel_o
);

  runtime_mode_e          runtime_mode_q;
  logic [LAYER_ID_W-1:0]  layer_id_q;
  logic                   busy_q;

  assign busy_o            = busy_q;
  assign layer_ctx_valid_o = busy_q;
  assign runtime_mode_o    = runtime_mode_q;
  assign layer_id_o        = layer_id_q;
  assign weight_layer_sel_o = layer_id_q;
  assign kv_layer_sel_o     = layer_id_q;

  always_ff @(posedge ap_clk) begin
    run_done_o    <= 1'b0;
    layer_start_o <= 1'b0;

    if (!ap_rst_n) begin
      runtime_mode_q <= MODE_PREFILL;
      layer_id_q     <= '0;
      busy_q         <= 1'b0;
    end else begin
      if (busy_q && abort_req_i) begin
        run_done_o <= 1'b1;
        busy_q     <= 1'b0;
      end else if (!busy_q && start_i) begin
        runtime_mode_q <= runtime_mode_i;
        layer_id_q     <= '0;
        busy_q         <= 1'b1;
        layer_start_o  <= 1'b1;
      end else if (busy_q && layer_step_done_i) begin
        if (layer_id_q == (N_LAYERS - 1)) begin
          run_done_o <= 1'b1;
          busy_q     <= 1'b0;
        end else begin
          layer_id_q    <= layer_id_q + 1'b1;
          layer_start_o <= 1'b1;
        end
      end
    end
  end

endmodule
