import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module debug_capture_mux #(
  parameter int unsigned DEBUG_SOURCE_COUNT = 1
) (
  input  logic                                ap_clk,
  input  logic                                ap_rst_n,
  input  logic                                debug_enable_i,
  input  logic [LAYER_ID_W-1:0]               debug_layer_sel_i,
  input  logic [7:0]                          debug_step_sel_i,
  input  logic [DEBUG_SOURCE_COUNT-1:0]       src_valid_i,
  input  logic [(DEBUG_SOURCE_COUNT * $bits(dbg_bus_t))-1:0] src_dbg_flat_i,
  output logic                                dbg_valid_o,
  input  logic                                dbg_ready_i,
  output dbg_bus_t                            dbg_o,
  output logic                                drop_pulse_o
);

  localparam int unsigned DBG_SRC_W = $bits(dbg_bus_t);

  logic                    has_match_w;
  dbg_bus_t                selected_dbg_w;
  logic [BLOCK_ID_W-1:0]   step_block_sel_w;

  assign step_block_sel_w = debug_step_sel_i[BLOCK_ID_W-1:0];

  always_comb begin
    dbg_bus_t candidate_dbg;

    has_match_w   = 1'b0;
    selected_dbg_w = '0;

    for (int src_idx = 0; src_idx < DEBUG_SOURCE_COUNT; src_idx++) begin
      candidate_dbg = dbg_bus_t'(src_dbg_flat_i[(src_idx * DBG_SRC_W) +: DBG_SRC_W]);
      if (!has_match_w &&
          src_valid_i[src_idx] &&
          (candidate_dbg.tag.layer_id == debug_layer_sel_i) &&
          (candidate_dbg.tag.block_id == block_id_e'(step_block_sel_w))) begin
        has_match_w    = 1'b1;
        selected_dbg_w = candidate_dbg;
      end
    end
  end

  assign dbg_valid_o = debug_enable_i && has_match_w;
  assign dbg_o       = selected_dbg_w;

  always_ff @(posedge ap_clk) begin
    drop_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      drop_pulse_o <= 1'b0;
    end else if (debug_enable_i && has_match_w && !dbg_ready_i) begin
      drop_pulse_o <= 1'b1;
    end
  end

endmodule
