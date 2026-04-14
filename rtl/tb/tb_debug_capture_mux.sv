`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_debug_capture_mux;

  localparam int unsigned DEBUG_SOURCE_COUNT = 3;
  localparam int unsigned DBG_SRC_W = $bits(dbg_bus_t);

  logic                          clk;
  logic                          rst_n;
  logic                          debug_enable;
  logic [LAYER_ID_W-1:0]         debug_layer_sel;
  logic [7:0]                    debug_step_sel;
  logic [DEBUG_SOURCE_COUNT-1:0] src_valid;
  logic [(DEBUG_SOURCE_COUNT * DBG_SRC_W)-1:0] src_dbg_flat;
  logic                          dbg_valid;
  logic                          dbg_ready;
  dbg_bus_t                      dbg_bus;
  logic                          drop_pulse;

  debug_capture_mux #(
    .DEBUG_SOURCE_COUNT(DEBUG_SOURCE_COUNT)
  ) dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .debug_enable_i(debug_enable),
    .debug_layer_sel_i(debug_layer_sel),
    .debug_step_sel_i(debug_step_sel),
    .src_valid_i(src_valid),
    .src_dbg_flat_i(src_dbg_flat),
    .dbg_valid_o(dbg_valid),
    .dbg_ready_i(dbg_ready),
    .dbg_o(dbg_bus),
    .drop_pulse_o(drop_pulse)
  );

  always #5 clk = ~clk;

  task automatic clear_sources;
    begin
      src_valid = '0;
      src_dbg_flat = '0;
    end
  endtask

  task automatic set_source(
    input int unsigned src_idx,
    input logic        valid,
    input dbg_bus_t    dbg_src
  );
    begin
      src_valid[src_idx] = valid;
      src_dbg_flat[(src_idx * DBG_SRC_W) +: DBG_SRC_W] = dbg_src;
    end
  endtask

  initial begin
    dbg_bus_t src_dbg_local;

    clk             = 1'b0;
    rst_n           = 1'b0;
    debug_enable    = 1'b0;
    debug_layer_sel = '0;
    debug_step_sel  = '0;
    dbg_ready       = 1'b1;
    clear_sources();

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    debug_enable    = 1'b1;
    debug_layer_sel = LAYER_ID_W'(3);
    debug_step_sel  = 8'(BLOCK_Q);

    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_Q;
    src_dbg_local.tag.tile_id = TILE_ID_W'(7);
    src_dbg_local.data = 256'h1234;
    set_source(0, 1'b1, src_dbg_local);
    #1;
    if (!dbg_valid || (dbg_bus.tag.tile_id != TILE_ID_W'(7)) || (dbg_bus.data != 256'h1234)) begin
      $error("debug_capture_mux failed to forward matching source");
      $finish;
    end

    clear_sources();
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_Q;
    src_dbg_local.tag.tile_id = TILE_ID_W'(1);
    src_dbg_local.data = 256'h1111;
    set_source(0, 1'b1, src_dbg_local);
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_Q;
    src_dbg_local.tag.tile_id = TILE_ID_W'(2);
    src_dbg_local.data = 256'h2222;
    set_source(1, 1'b1, src_dbg_local);
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_V;
    src_dbg_local.data = 256'h3333;
    set_source(2, 1'b1, src_dbg_local);
    #1;
    if ((dbg_bus.tag.tile_id != TILE_ID_W'(1)) || (dbg_bus.data != 256'h1111)) begin
      $error("debug_capture_mux did not give priority to the lowest matching source index");
      $finish;
    end

    clear_sources();
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(2);
    src_dbg_local.tag.block_id = BLOCK_Q;
    set_source(1, 1'b1, src_dbg_local);
    #1;
    if (dbg_valid) begin
      $error("debug_capture_mux forwarded a layer-mismatched source");
      $finish;
    end

    clear_sources();
    dbg_ready = 1'b0;
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_Q;
    set_source(2, 1'b1, src_dbg_local);
    @(posedge clk);
    @(negedge clk);
    clear_sources();
    if (!drop_pulse) begin
      $error("debug_capture_mux failed to report a dropped capture");
      $finish;
    end
    @(posedge clk);
    @(negedge clk);
    if (drop_pulse) begin
      $error("debug_capture_mux drop_pulse did not clear after one cycle");
      $finish;
    end

    clear_sources();
    dbg_ready = 1'b1;
    debug_enable = 1'b0;
    src_dbg_local = '0;
    src_dbg_local.tag.layer_id = LAYER_ID_W'(3);
    src_dbg_local.tag.block_id = BLOCK_Q;
    set_source(0, 1'b1, src_dbg_local);
    #1;
    if (dbg_valid) begin
      $error("debug_capture_mux ignored debug_enable");
      $finish;
    end

    $display("PASS: tb_debug_capture_mux");
    $finish;
  end

endmodule
