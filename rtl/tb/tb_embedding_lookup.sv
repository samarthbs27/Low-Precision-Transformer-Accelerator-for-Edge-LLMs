`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_embedding_lookup;

  localparam int unsigned EMBED_ROW_BYTES = D_MODEL * 2;
  localparam int unsigned EMBED_ROW_BEATS = EMBED_ROW_BYTES / DMA_BEAT_BYTES;
  localparam string PREFILL_BASE = "sim/golden_traces/phase6/rtl/phase6_prefill_embedding_lookup_row0";
  localparam string DECODE_BASE  = "sim/golden_traces/phase6/rtl/phase6_decode_embedding_lookup_row0";

  logic                               clk;
  logic                               rst_n;
  logic [HBM_ADDR_W-1:0]              embedding_base_addr;
  logic                               token_valid;
  logic                               token_ready;
  token_bus_t                         token_bus;
  logic                               req_valid;
  logic                               req_ready;
  logic [HBM_ADDR_W-1:0]              req_base_addr;
  logic [31:0]                        req_byte_count;
  tensor_id_e                         req_tensor_id;
  logic [LAYER_ID_W-1:0]              req_layer_id;
  logic [TILE_ID_W-1:0]               req_tile_id;
  logic                               embed_row_valid;
  logic                               embed_row_ready;
  logic [DMA_BEAT_W-1:0]              embed_row_data;
  logic                               embed_row_last;
  logic                               row_valid;
  logic                               row_ready;
  logic [(D_MODEL * 16)-1:0]          row_fp16;
  token_bus_t                         row_meta;
  logic                               busy;
  logic                               done_pulse;
  logic                               saw_row;
  logic [(D_MODEL * 16)-1:0]          captured_row_fp16;
  token_bus_t                         captured_row_meta;

  logic [31:0]                        meta_mem [0:3];
  logic [DMA_BEAT_W-1:0]              row_beats_mem [0:EMBED_ROW_BEATS-1];

  embedding_lookup dut (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    .embedding_base_addr_i(embedding_base_addr),
    .token_valid_i(token_valid),
    .token_ready_o(token_ready),
    .token_i(token_bus),
    .req_valid_o(req_valid),
    .req_ready_i(req_ready),
    .req_base_addr_o(req_base_addr),
    .req_byte_count_o(req_byte_count),
    .req_tensor_id_o(req_tensor_id),
    .req_layer_id_o(req_layer_id),
    .req_tile_id_o(req_tile_id),
    .embed_row_valid_i(embed_row_valid),
    .embed_row_ready_o(embed_row_ready),
    .embed_row_i(embed_row_data),
    .embed_row_last_i(embed_row_last),
    .row_valid_o(row_valid),
    .row_ready_i(row_ready),
    .row_fp16_o(row_fp16),
    .row_meta_o(row_meta),
    .busy_o(busy),
    .done_pulse_o(done_pulse)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      saw_row <= 1'b0;
      captured_row_fp16 <= '0;
      captured_row_meta <= '0;
    end else if (row_valid && row_ready) begin
      saw_row <= 1'b1;
      captured_row_fp16 <= row_fp16;
      captured_row_meta <= row_meta;
    end
  end

  task automatic reset_capture;
    begin
      @(negedge clk);
      saw_row = 1'b0;
      captured_row_fp16 = '0;
      captured_row_meta = '0;
    end
  endtask

  task automatic load_case(input string base);
    begin
      $readmemh({base, ".meta.memh"}, meta_mem);
      $readmemh({base, ".row_beats_packed.memh"}, row_beats_mem);
    end
  endtask

  task automatic wait_for_row(input string case_name);
    int timeout_cycles;
    begin
      timeout_cycles = 0;
      while (!saw_row && (timeout_cycles < 256)) begin
        @(posedge clk);
        timeout_cycles++;
      end
      if (!saw_row) begin
        $error("embedding_lookup timed out waiting for row for %s", case_name);
        $finish;
      end
      @(posedge clk);
      if (busy || row_valid || done_pulse) begin
        $error("embedding_lookup did not return idle after row output for %s", case_name);
        $finish;
      end
    end
  endtask

  task automatic drive_row_beats;
    begin
      for (int beat_idx = 0; beat_idx < EMBED_ROW_BEATS; beat_idx++) begin
        @(negedge clk);
        embed_row_data = row_beats_mem[beat_idx];
        embed_row_last = (beat_idx == (EMBED_ROW_BEATS - 1));
        embed_row_valid = 1'b1;
        do begin
          @(posedge clk);
        end while (!embed_row_ready);
        @(negedge clk);
        embed_row_valid = 1'b0;
      end
      embed_row_last = 1'b0;
      embed_row_data = '0;
    end
  endtask

  task automatic run_case(input string case_name, input string base);
    begin
      load_case(base);

      reset_capture();
      token_bus = '0;
      token_bus.token_id = meta_mem[0];
      token_bus.token_count = meta_mem[1];
      token_bus.tag.layer_id = '0;
      token_bus.tag.block_id = BLOCK_EMBED;
      token_bus.tag.gemm_mode = GEMM_NONE;
      token_bus.tag.tile_id = TILE_ID_W'(meta_mem[2]);
      token_bus.tag.token_base = POS_W'(meta_mem[2]);
      token_bus.tag.seq_count = COUNT_W'(meta_mem[1]);
      token_bus.tag.elem_count = 16'd1;
      token_bus.tag.is_last = meta_mem[3][0];
      token_bus.tag.is_partial = 1'b1;

      @(negedge clk);
      token_valid = 1'b1;
      do begin
        @(posedge clk);
      end while (!token_ready);
      @(negedge clk);
      token_valid = 1'b0;

      if (!req_valid ||
          (req_base_addr != (embedding_base_addr + (HBM_ADDR_W'(meta_mem[0]) * EMBED_ROW_BYTES))) ||
          (req_byte_count != EMBED_ROW_BYTES) ||
          (req_tensor_id != TENSOR_EMBED) ||
          (req_tile_id != TILE_ID_W'(meta_mem[2]))) begin
        $error("embedding_lookup request mismatch for %s", case_name);
        $finish;
      end

      @(negedge clk);
      req_ready = 1'b1;
      @(posedge clk);
      @(negedge clk);
      req_ready = 1'b0;

      drive_row_beats();
      wait_for_row(case_name);

      for (int beat_idx = 0; beat_idx < EMBED_ROW_BEATS; beat_idx++) begin
        if (captured_row_fp16[(beat_idx * DMA_BEAT_W) +: DMA_BEAT_W] != row_beats_mem[beat_idx]) begin
          $error("embedding_lookup row data mismatch for %s at beat %0d", case_name, beat_idx);
          $finish;
        end
      end

      if ((captured_row_meta.token_id != meta_mem[0]) ||
          (captured_row_meta.token_count != meta_mem[1]) ||
          (captured_row_meta.tag.token_base != POS_W'(meta_mem[2])) ||
          (captured_row_meta.tag.is_last != meta_mem[3][0])) begin
        $error("embedding_lookup row metadata mismatch for %s", case_name);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    embedding_base_addr = 64'h0000_0000_1000_0000;
    token_valid = 1'b0;
    token_bus = '0;
    req_ready = 1'b0;
    embed_row_valid = 1'b0;
    embed_row_data = '0;
    embed_row_last = 1'b0;
    row_ready = 1'b1;
    saw_row = 1'b0;
    captured_row_fp16 = '0;
    captured_row_meta = '0;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (int beat_idx = 0; beat_idx < EMBED_ROW_BEATS; beat_idx++) begin
      row_beats_mem[beat_idx] = DMA_BEAT_W'(beat_idx);
    end

    reset_capture();
    token_bus.token_id = 32'd3;
    token_bus.token_count = 32'd2;
    token_bus.tag.block_id = BLOCK_EMBED;
    token_bus.tag.tile_id = TILE_ID_W'(1);
    token_bus.tag.token_base = POS_W'(1);
    token_bus.tag.seq_count = COUNT_W'(2);
    token_bus.tag.elem_count = 16'd1;
    token_bus.tag.is_last = 1'b0;
    token_bus.tag.is_partial = 1'b1;
    @(negedge clk);
    token_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!token_ready);
    @(negedge clk);
    token_valid = 1'b0;

    if ((req_base_addr != (embedding_base_addr + (64'd3 * EMBED_ROW_BYTES))) ||
        (req_byte_count != EMBED_ROW_BYTES) ||
        (req_tensor_id != TENSOR_EMBED)) begin
      $error("embedding_lookup directed request mismatch");
      $finish;
    end

    @(negedge clk);
    req_ready = 1'b1;
    @(posedge clk);
    @(negedge clk);
    req_ready = 1'b0;
    drive_row_beats();
    wait_for_row("directed");
    if ((captured_row_fp16[0 +: DMA_BEAT_W] != row_beats_mem[0]) ||
        (captured_row_fp16[((EMBED_ROW_BEATS - 1) * DMA_BEAT_W) +: DMA_BEAT_W] != row_beats_mem[EMBED_ROW_BEATS - 1])) begin
      $error("embedding_lookup directed row assembly mismatch");
      $finish;
    end

    run_case("phase6_prefill_embedding_lookup_row0", PREFILL_BASE);
    run_case("phase6_decode_embedding_lookup_row0", DECODE_BASE);

    $display("PASS: tb_embedding_lookup");
    $finish;
  end

endmodule
