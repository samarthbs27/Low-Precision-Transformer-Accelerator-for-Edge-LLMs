`timescale 1ns/1ps

import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tb_kernel_top_smoke;

  localparam int unsigned TRACE_META_WORDS = 12;
  localparam int unsigned TRACE_PROMPT_MAX = 16;
  localparam int unsigned SCALE_BEATS = (SCALE_VECTOR_ELEMS * SCALE_W) / DMA_BEAT_W;

  typedef enum logic [1:0] {
    RD_NONE   = 2'd0,
    RD_CMD    = 2'd1,
    RD_PROMPT = 2'd2,
    RD_EMBED  = 2'd3
  } rd_kind_e;

  logic [31:0] trace_meta [0:TRACE_META_WORDS-1];
  logic [31:0] trace_command_words [0:HOST_BLOCK_WORDS-1];
  logic [31:0] trace_prompt_tokens [0:TRACE_PROMPT_MAX-1];

  logic                  clk;
  logic                  rst_n;
  logic [AXIL_ADDR_W-1:0] s_axi_awaddr;
  logic                  s_axi_awvalid;
  logic                  s_axi_awready;
  logic [AXIL_DATA_W-1:0] s_axi_wdata;
  logic [AXIL_STRB_W-1:0] s_axi_wstrb;
  logic                  s_axi_wvalid;
  logic                  s_axi_wready;
  logic [1:0]            s_axi_bresp;
  logic                  s_axi_bvalid;
  logic                  s_axi_bready;
  logic [AXIL_ADDR_W-1:0] s_axi_araddr;
  logic                  s_axi_arvalid;
  logic                  s_axi_arready;
  logic [AXIL_DATA_W-1:0] s_axi_rdata;
  logic [1:0]            s_axi_rresp;
  logic                  s_axi_rvalid;
  logic                  s_axi_rready;
  logic                  interrupt;

  logic                  shell_rd_desc_valid;
  logic                  shell_rd_desc_ready;
  dma_desc_t             shell_rd_desc;
  logic                  shell_rd_data_valid;
  logic                  shell_rd_data_ready;
  logic [DMA_BEAT_W-1:0] shell_rd_data;
  logic                  shell_rd_burst_active_q;
  rd_kind_e              shell_rd_kind_q;
  tensor_id_e            shell_rd_tensor_id_q;
  logic [15:0]           shell_rd_beat_idx_q;
  logic [15:0]           shell_rd_beats_total_q;
  logic                  shell_wr_desc_valid;
  logic                  shell_wr_desc_ready;
  dma_desc_t             shell_wr_desc;
  logic                  shell_wr_data_valid;
  logic                  shell_wr_data_ready;
  logic [DMA_BEAT_W-1:0] shell_wr_data;

  int unsigned           command_read_count;
  int unsigned           prompt_read_count;
  int unsigned           embed_read_count;
  int unsigned           scale_read_count;
  int unsigned           launch_status_write_count;
  int unsigned           final_status_write_count;
  int unsigned           generated_write_count;
  logic [DMA_BEAT_W-1:0] final_status_payload;
  logic                  irq_seen;

  function automatic logic [DMA_BEAT_W-1:0] pack_command_beat;
    logic [DMA_BEAT_W-1:0] beat_value;
    begin
      beat_value = '0;
      for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
        beat_value[(word_idx*AXIL_DATA_W) +: AXIL_DATA_W] = trace_command_words[word_idx];
      end
      return beat_value;
    end
  endfunction

  function automatic logic [DMA_BEAT_W-1:0] pack_unit_scale_beat;
    logic [DMA_BEAT_W-1:0] beat_value;
    begin
      beat_value = '0;
      for (int lane = 0; lane < (DMA_BEAT_W / SCALE_W); lane++) begin
        beat_value[(lane * SCALE_W) +: SCALE_W] = 32'h0001_0000;
      end
      return beat_value;
    end
  endfunction

  function automatic logic [DMA_BEAT_W-1:0] pack_prompt_beat(
    input int unsigned beat_idx,
    input int unsigned prompt_count
  );
    logic [DMA_BEAT_W-1:0] beat_value;
    int unsigned token_idx;
    begin
      beat_value = '0;
      for (int lane = 0; lane < TOKENS_PER_DMA_BEAT; lane++) begin
        token_idx = (beat_idx * TOKENS_PER_DMA_BEAT) + lane;
        if (token_idx < prompt_count) begin
          beat_value[(lane*TOKEN_W) +: TOKEN_W] = trace_prompt_tokens[token_idx];
        end
      end
      return beat_value;
    end
  endfunction

  tinyllama_u55c_kernel_top dut (
    .ap_clk              (clk),
    .ap_rst_n            (rst_n),
    .s_axi_awaddr        (s_axi_awaddr),
    .s_axi_awvalid       (s_axi_awvalid),
    .s_axi_awready       (s_axi_awready),
    .s_axi_wdata         (s_axi_wdata),
    .s_axi_wstrb         (s_axi_wstrb),
    .s_axi_wvalid        (s_axi_wvalid),
    .s_axi_wready        (s_axi_wready),
    .s_axi_bresp         (s_axi_bresp),
    .s_axi_bvalid        (s_axi_bvalid),
    .s_axi_bready        (s_axi_bready),
    .s_axi_araddr        (s_axi_araddr),
    .s_axi_arvalid       (s_axi_arvalid),
    .s_axi_arready       (s_axi_arready),
    .s_axi_rdata         (s_axi_rdata),
    .s_axi_rresp         (s_axi_rresp),
    .s_axi_rvalid        (s_axi_rvalid),
    .s_axi_rready        (s_axi_rready),
    .interrupt           (interrupt),
    .shell_rd_desc_valid_o(shell_rd_desc_valid),
    .shell_rd_desc_ready_i(shell_rd_desc_ready),
    .shell_rd_desc_o     (shell_rd_desc),
    .shell_rd_data_valid_i(shell_rd_data_valid),
    .shell_rd_data_ready_o(shell_rd_data_ready),
    .shell_rd_data_i     (shell_rd_data),
    .shell_wr_desc_valid_o(shell_wr_desc_valid),
    .shell_wr_desc_ready_i(shell_wr_desc_ready),
    .shell_wr_desc_o     (shell_wr_desc),
    .shell_wr_data_valid_o(shell_wr_data_valid),
    .shell_wr_data_ready_i(shell_wr_data_ready),
    .shell_wr_data_o     (shell_wr_data)
  );

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      shell_rd_burst_active_q <= 1'b0;
      shell_rd_kind_q         <= RD_NONE;
      shell_rd_tensor_id_q    <= TENSOR_NONE;
      shell_rd_beat_idx_q     <= '0;
      shell_rd_beats_total_q  <= '0;
      shell_rd_data_valid     <= 1'b0;
      shell_rd_data           <= '0;
      command_read_count      <= 0;
      prompt_read_count       <= 0;
      embed_read_count        <= 0;
      scale_read_count        <= 0;
      launch_status_write_count <= 0;
      final_status_write_count <= 0;
      generated_write_count   <= 0;
      final_status_payload    <= '0;
      irq_seen                <= 1'b0;
    end else begin
      if (interrupt) begin
        irq_seen <= 1'b1;
      end

      if (shell_rd_burst_active_q && !shell_rd_data_valid) begin
        shell_rd_data_valid <= 1'b1;
        unique case (shell_rd_kind_q)
          RD_CMD:    shell_rd_data <= pack_command_beat();
          RD_PROMPT: shell_rd_data <= pack_prompt_beat(shell_rd_beat_idx_q, trace_meta[4]);
          RD_EMBED:  shell_rd_data <= (shell_rd_tensor_id_q == TENSOR_SCALE_META) ? pack_unit_scale_beat() : '0;
          default:   shell_rd_data <= '0;
        endcase
      end else if (shell_rd_data_valid && shell_rd_data_ready) begin
        if (shell_rd_beat_idx_q + 1 >= shell_rd_beats_total_q) begin
          shell_rd_data_valid <= 1'b0;
          shell_rd_burst_active_q <= 1'b0;
          shell_rd_kind_q <= RD_NONE;
          shell_rd_beat_idx_q <= '0;
        end else begin
          shell_rd_beat_idx_q <= shell_rd_beat_idx_q + 1'b1;
          unique case (shell_rd_kind_q)
            RD_CMD:    shell_rd_data <= pack_command_beat();
            RD_PROMPT: shell_rd_data <= pack_prompt_beat(shell_rd_beat_idx_q + 1'b1, trace_meta[4]);
            RD_EMBED:  shell_rd_data <= (shell_rd_tensor_id_q == TENSOR_SCALE_META) ? pack_unit_scale_beat() : '0;
            default:   shell_rd_data <= '0;
          endcase
        end
      end

      if (shell_rd_desc_valid && shell_rd_desc_ready) begin
        if (shell_rd_desc.addr == {trace_meta[1], trace_meta[0]}) begin
          command_read_count <= command_read_count + 1;
          shell_rd_kind_q <= RD_CMD;
        end else if ((shell_rd_desc.region == REGION_HOST_IO) && (shell_rd_desc.tensor_id == TENSOR_NONE)) begin
          prompt_read_count <= prompt_read_count + 1;
          shell_rd_kind_q <= RD_PROMPT;
        end else if ((shell_rd_desc.region == REGION_EMBED_META) &&
                     ((shell_rd_desc.tensor_id == TENSOR_EMBED) ||
                      (shell_rd_desc.tensor_id == TENSOR_SCALE_META))) begin
          if (shell_rd_desc.tensor_id == TENSOR_SCALE_META) begin
            scale_read_count <= scale_read_count + 1;
          end else begin
            embed_read_count <= embed_read_count + 1;
          end
          shell_rd_kind_q <= RD_EMBED;
        end else begin
          $error("tb_kernel_top_smoke saw unexpected read descriptor tensor=%0d region=%0d addr=0x%0h",
                 shell_rd_desc.tensor_id,
                 shell_rd_desc.region,
                 shell_rd_desc.addr);
          $finish;
        end
        shell_rd_burst_active_q <= 1'b1;
        shell_rd_tensor_id_q <= shell_rd_desc.tensor_id;
        shell_rd_beat_idx_q <= '0;
        shell_rd_beats_total_q <= (shell_rd_desc.byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
        shell_rd_data_valid <= 1'b0;
      end

      if (shell_wr_desc_valid && shell_wr_data_valid && shell_wr_desc_ready && shell_wr_data_ready) begin
        if (shell_wr_desc.addr == {trace_meta[3], trace_meta[2]}) begin
          if (shell_wr_data[STATUS_BUSY_BIT]) begin
            launch_status_write_count <= launch_status_write_count + 1;
          end else begin
            final_status_write_count <= final_status_write_count + 1;
            final_status_payload <= shell_wr_data;
          end
        end else if (shell_wr_desc.addr >= {trace_command_words[HOST_CMD_WORD_GEN_BASE_HI], trace_command_words[HOST_CMD_WORD_GEN_BASE_LO]}) begin
          if (shell_wr_data[TOKEN_W-1:0] != (32'd1000 + generated_write_count)) begin
            $error("generated token mismatch at index %0d: expected %0d got %0d",
                   generated_write_count,
                   32'd1000 + generated_write_count,
                   shell_wr_data[TOKEN_W-1:0]);
            $finish;
          end
          generated_write_count <= generated_write_count + 1;
        end
      end
    end
  end

  assign shell_rd_desc_ready = 1'b1;
  assign shell_wr_desc_ready = 1'b1;
  assign shell_wr_data_ready = 1'b1;

  task automatic axi_write(
    input logic [AXIL_ADDR_W-1:0] addr,
    input logic [AXIL_DATA_W-1:0] data
  );
    int unsigned wait_cycles;
    begin
      @(negedge clk);
      s_axi_awaddr  <= addr;
      s_axi_awvalid <= 1'b1;
      s_axi_wdata   <= data;
      s_axi_wstrb   <= '1;
      s_axi_wvalid  <= 1'b1;
      wait_cycles = 0;
      while (!(s_axi_awready && s_axi_wready)) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI write timeout at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_awvalid <= 1'b0;
      s_axi_wvalid  <= 1'b0;
      s_axi_bready  <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_bvalid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI write response timeout at addr 0x%0h", addr);
          $finish;
        end
      end
      @(negedge clk);
      s_axi_bready <= 1'b0;
    end
  endtask

  task automatic axi_read(
    input logic [AXIL_ADDR_W-1:0] addr,
    output logic [AXIL_DATA_W-1:0] data
  );
    int unsigned wait_cycles;
    begin
      @(negedge clk);
      s_axi_araddr  <= addr;
      s_axi_arvalid <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_arready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI read timeout at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_arvalid <= 1'b0;
      s_axi_rready  <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_rvalid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI read response timeout at addr 0x%0h", addr);
          $finish;
        end
      end
      data = s_axi_rdata;
      @(negedge clk);
      s_axi_rready <= 1'b0;
    end
  endtask

  logic [31:0] readback;

  initial begin
    $readmemh("sim/golden_traces/phase8/rtl/phase8_prefill_decode_runtime.meta.memh", trace_meta);
    $readmemh("sim/golden_traces/phase8/rtl/phase8_prefill_decode_runtime.command_words.memh", trace_command_words);
    $readmemh("sim/golden_traces/phase8/rtl/phase8_prefill_decode_runtime.prompt_tokens.memh", trace_prompt_tokens);

    clk          = 1'b0;
    rst_n        = 1'b0;
    s_axi_awaddr = '0;
    s_axi_awvalid = 1'b0;
    s_axi_wdata  = '0;
    s_axi_wstrb  = '0;
    s_axi_wvalid = 1'b0;
    s_axi_bready = 1'b0;
    s_axi_araddr = '0;
    s_axi_arvalid = 1'b0;
    s_axi_rready = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    axi_write(REGW_CMD_BASE_LO << 2, trace_meta[0]);
    axi_write(REGW_CMD_BASE_HI << 2, trace_meta[1]);
    axi_write(REGW_STATUS_BASE_LO << 2, trace_meta[2]);
    axi_write(REGW_STATUS_BASE_HI << 2, trace_meta[3]);
    axi_write(REGW_PROMPT_TOKEN_COUNT << 2, trace_meta[4]);
    axi_write(REGW_MAX_NEW_TOKENS << 2, trace_meta[5]);
    axi_write(REGW_EOS_TOKEN_ID << 2, trace_meta[6]);
    axi_write(REGW_CONTROL << 2, 32'h0000_0001);

    begin : wait_done
      int unsigned wait_cycles;
      wait_cycles = 0;
      while (final_status_write_count == 0) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 40000) begin
          $error("tb_kernel_top_smoke timeout waiting for final status write");
          $finish;
        end
      end
    end

    axi_read(REGW_STATUS << 2, readback);
    if (!readback[STATUS_DONE_BIT] ||
        !readback[STATUS_STOP_VALID_BIT] ||
        (readback[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] != STOP_REASON_MAX_TOKENS)) begin
      $error("AXI status readback mismatch: 0x%08h", readback);
      $finish;
    end

    axi_read(REGW_GENERATED_TOKEN_COUNT << 2, readback);
    if (readback != 32'd2) begin
      $error("generated-token-count readback expected 2, got %0d", readback);
      $finish;
    end

    axi_read(REGW_LAST_TOKEN_ID << 2, readback);
    if (readback != 32'd1001) begin
      $error("last-token-id readback expected 1001, got %0d", readback);
      $finish;
    end

    if (!irq_seen) begin
      $error("interrupt expected during runtime completion");
      $finish;
    end
    if (command_read_count != 1) begin
      $error("expected one command read, got %0d", command_read_count);
      $finish;
    end
    if (prompt_read_count != trace_meta[9]) begin
      $error("expected %0d prompt reads, got %0d", trace_meta[9], prompt_read_count);
      $finish;
    end
    if (scale_read_count != 1 || embed_read_count != trace_meta[4]) begin
      $error("expected one scale read and %0d embedding reads, got scale=%0d embed=%0d",
             trace_meta[4],
             scale_read_count,
             embed_read_count);
      $finish;
    end
    if (generated_write_count != 2) begin
      $error("expected 2 generated token writes, got %0d", generated_write_count);
      $finish;
    end
    if (launch_status_write_count != 1 || final_status_write_count != 1) begin
      $error("expected one launch and one final status write, got launch=%0d final=%0d",
             launch_status_write_count, final_status_write_count);
      $finish;
    end
    if (final_status_payload[31:0] != trace_meta[11] ||
        final_status_payload[(HOST_STATUS_WORD_GEN_COUNT*32) +: 32] != 32'd2 ||
        final_status_payload[(HOST_STATUS_WORD_LAST_TOKEN*32) +: 32] != 32'd1001) begin
      $error("final status payload mismatch");
      $finish;
    end

    $display("PASS: tb_kernel_top_smoke");
    $finish;
  end

endmodule
