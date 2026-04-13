`timescale 1ns/1ps

import tinyllama_pkg::*;

module tb_axi_lite_ctrl_slave;

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

  logic                       reg_wr_en;
  logic [REG_WORD_ADDR_W-1:0] reg_wr_addr;
  logic [AXIL_DATA_W-1:0]     reg_wr_data;
  logic [AXIL_STRB_W-1:0]     reg_wr_strb;
  logic                       reg_rd_en;
  logic [REG_WORD_ADDR_W-1:0] reg_rd_addr;
  logic [AXIL_DATA_W-1:0]     reg_rd_data;

  logic                       hw_busy;
  logic                       hw_done_pulse;
  logic                       hw_error_valid;
  error_code_e                hw_error_code;
  logic                       hw_stop_valid;
  stop_reason_e               hw_stop_reason;
  logic [COUNT_W-1:0]         hw_generated_token_count;
  logic [TOKEN_W-1:0]         hw_last_token_id;
  logic [LAYER_ID_W-1:0]      hw_current_layer;
  block_id_e                  hw_current_block;

  logic                       start_pulse;
  runtime_mode_e              launch_mode;
  logic                       abort_req;
  logic [HBM_ADDR_W-1:0]      cmd_base_addr;
  logic [HBM_ADDR_W-1:0]      status_base_addr;
  logic [HBM_ADDR_W-1:0]      debug_base_addr;
  logic [COUNT_W-1:0]         prompt_token_count;
  logic [COUNT_W-1:0]         max_new_tokens;
  logic [TOKEN_W-1:0]         eos_token_id;
  logic                       debug_enable;
  logic [LAYER_ID_W-1:0]      debug_layer_sel;
  logic [DEBUG_CFG_STEP_W-1:0] debug_step_sel;

  axi_lite_ctrl_slave dut_axi (
    .ap_clk       (clk),
    .ap_rst_n     (rst_n),
    .s_axi_awaddr (s_axi_awaddr),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_awready(s_axi_awready),
    .s_axi_wdata  (s_axi_wdata),
    .s_axi_wstrb  (s_axi_wstrb),
    .s_axi_wvalid (s_axi_wvalid),
    .s_axi_wready (s_axi_wready),
    .s_axi_bresp  (s_axi_bresp),
    .s_axi_bvalid (s_axi_bvalid),
    .s_axi_bready (s_axi_bready),
    .s_axi_araddr (s_axi_araddr),
    .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata  (s_axi_rdata),
    .s_axi_rresp  (s_axi_rresp),
    .s_axi_rvalid (s_axi_rvalid),
    .s_axi_rready (s_axi_rready),
    .reg_wr_en    (reg_wr_en),
    .reg_wr_addr  (reg_wr_addr),
    .reg_wr_data  (reg_wr_data),
    .reg_wr_strb  (reg_wr_strb),
    .reg_rd_en    (reg_rd_en),
    .reg_rd_addr  (reg_rd_addr),
    .reg_rd_data  (reg_rd_data)
  );

  kernel_reg_file dut_regfile (
    .ap_clk                  (clk),
    .ap_rst_n                (rst_n),
    .reg_wr_en               (reg_wr_en),
    .reg_wr_addr             (reg_wr_addr),
    .reg_wr_data             (reg_wr_data),
    .reg_wr_strb             (reg_wr_strb),
    .reg_rd_en               (reg_rd_en),
    .reg_rd_addr             (reg_rd_addr),
    .reg_rd_data             (reg_rd_data),
    .hw_busy_i               (hw_busy),
    .hw_done_pulse_i         (hw_done_pulse),
    .hw_error_valid_i        (hw_error_valid),
    .hw_error_code_i         (hw_error_code),
    .hw_stop_valid_i         (hw_stop_valid),
    .hw_stop_reason_i        (hw_stop_reason),
    .hw_generated_token_count_i(hw_generated_token_count),
    .hw_last_token_id_i      (hw_last_token_id),
    .hw_current_layer_i      (hw_current_layer),
    .hw_current_block_i      (hw_current_block),
    .start_pulse_o           (start_pulse),
    .launch_mode_o           (launch_mode),
    .abort_req_o             (abort_req),
    .cmd_base_addr_o         (cmd_base_addr),
    .status_base_addr_o      (status_base_addr),
    .debug_base_addr_o       (debug_base_addr),
    .prompt_token_count_o    (prompt_token_count),
    .max_new_tokens_o        (max_new_tokens),
    .eos_token_id_o          (eos_token_id),
    .debug_enable_o          (debug_enable),
    .debug_layer_sel_o       (debug_layer_sel),
    .debug_step_sel_o        (debug_step_sel)
  );

  always #5 clk = ~clk;

  task automatic axi_write_same_cycle(
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
          $error("AXI same-cycle write timeout waiting for awready/wready at addr 0x%0h", addr);
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
          $error("AXI same-cycle write timeout waiting for bvalid at addr 0x%0h", addr);
          $finish;
        end
      end
      if (s_axi_bresp != 2'b00) begin
        $error("AXI write response expected OKAY, got %0b", s_axi_bresp);
        $finish;
      end
      @(negedge clk);
      s_axi_bready <= 1'b0;
    end
  endtask

  task automatic axi_write_aw_first(
    input logic [AXIL_ADDR_W-1:0] addr,
    input logic [AXIL_DATA_W-1:0] data
  );
    int unsigned wait_cycles;
    begin
      @(negedge clk);
      s_axi_awaddr  <= addr;
      s_axi_awvalid <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_awready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI AW-first timeout waiting for awready at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_awvalid <= 1'b0;

      @(negedge clk);
      s_axi_wdata  <= data;
      s_axi_wstrb  <= '1;
      s_axi_wvalid <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_wready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI AW-first timeout waiting for wready at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_wvalid <= 1'b0;

      s_axi_bready = 1'b1;
      wait_cycles = 0;
      while (!s_axi_bvalid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI AW-first timeout waiting for bvalid at addr 0x%0h", addr);
          $finish;
        end
      end
      if (s_axi_bresp != 2'b00) begin
        $error("AXI AW-first write response expected OKAY, got %0b", s_axi_bresp);
        $finish;
      end
      @(negedge clk);
      s_axi_bready <= 1'b0;
    end
  endtask

  task automatic axi_write_w_first(
    input logic [AXIL_ADDR_W-1:0] addr,
    input logic [AXIL_DATA_W-1:0] data
  );
    int unsigned wait_cycles;
    begin
      @(negedge clk);
      s_axi_wdata  <= data;
      s_axi_wstrb  <= '1;
      s_axi_wvalid <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_wready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI W-first timeout waiting for wready at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_wvalid <= 1'b0;

      @(negedge clk);
      s_axi_awaddr  <= addr;
      s_axi_awvalid <= 1'b1;
      wait_cycles = 0;
      while (!s_axi_awready) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI W-first timeout waiting for awready at addr 0x%0h", addr);
          $finish;
        end
      end
      @(posedge clk);
      s_axi_awvalid <= 1'b0;

      s_axi_bready = 1'b1;
      wait_cycles = 0;
      while (!s_axi_bvalid) begin
        @(negedge clk);
        wait_cycles++;
        if (wait_cycles > 64) begin
          $error("AXI W-first timeout waiting for bvalid at addr 0x%0h", addr);
          $finish;
        end
      end
      if (s_axi_bresp != 2'b00) begin
        $error("AXI W-first write response expected OKAY, got %0b", s_axi_bresp);
        $finish;
      end
      @(negedge clk);
      s_axi_bready <= 1'b0;
    end
  endtask

  task automatic axi_read(
    input  logic [AXIL_ADDR_W-1:0] addr,
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
          $error("AXI read timeout waiting for arready at addr 0x%0h", addr);
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
          $error("AXI read timeout waiting for rvalid at addr 0x%0h", addr);
          $finish;
        end
      end
      if (s_axi_rresp != 2'b00) begin
        $error("AXI read response expected OKAY, got %0b", s_axi_rresp);
        $finish;
      end
      data = s_axi_rdata;
      @(negedge clk);
      s_axi_rready <= 1'b0;
    end
  endtask

  logic [31:0] readback;

  initial begin
    clk                     = 1'b0;
    rst_n                   = 1'b0;
    s_axi_awaddr            = '0;
    s_axi_awvalid           = 1'b0;
    s_axi_wdata             = '0;
    s_axi_wstrb             = '0;
    s_axi_wvalid            = 1'b0;
    s_axi_bready            = 1'b0;
    s_axi_araddr            = '0;
    s_axi_arvalid           = 1'b0;
    s_axi_rready            = 1'b0;
    hw_busy                 = 1'b0;
    hw_done_pulse           = 1'b0;
    hw_error_valid          = 1'b0;
    hw_error_code           = ERROR_NONE;
    hw_stop_valid           = 1'b0;
    hw_stop_reason          = STOP_REASON_NONE;
    hw_generated_token_count = '0;
    hw_last_token_id        = '0;
    hw_current_layer        = '0;
    hw_current_block        = BLOCK_NONE;
    readback                = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    axi_write_same_cycle({REGW_CMD_BASE_LO, 2'b00}, 32'h1122_3344);
    axi_read({REGW_CMD_BASE_LO, 2'b00}, readback);
    if (readback !== 32'h1122_3344) begin
      $error("REGW_CMD_BASE_LO expected 0x11223344, got 0x%08h", readback);
      $finish;
    end

    axi_write_aw_first({REGW_CMD_BASE_HI, 2'b00}, 32'h5566_7788);
    if (cmd_base_addr !== 64'h5566_7788_1122_3344) begin
      $error("cmd_base_addr expected 0x5566778811223344, got 0x%016h", cmd_base_addr);
      $finish;
    end

    axi_write_w_first({REGW_STATUS_BASE_LO, 2'b00}, 32'hABCD_EF01);
    axi_read({REGW_STATUS_BASE_LO, 2'b00}, readback);
    if (readback !== 32'hABCD_EF01) begin
      $error("REGW_STATUS_BASE_LO expected 0xABCDEF01, got 0x%08h", readback);
      $finish;
    end

    axi_write_same_cycle({REGW_PROMPT_TOKEN_COUNT, 2'b00}, 32'd128);
    axi_write_same_cycle({REGW_MAX_NEW_TOKENS, 2'b00}, 32'd7);
    axi_write_same_cycle({REGW_EOS_TOKEN_ID, 2'b00}, 32'd2);
    axi_write_same_cycle({REGW_DEBUG_CFG, 2'b00}, (1 << DEBUG_CFG_ENABLE_BIT) | (5 << DEBUG_CFG_LAYER_LSB) | (9 << DEBUG_CFG_STEP_LSB));

    if (prompt_token_count !== 128) begin
      $error("prompt_token_count expected 128, got %0d", prompt_token_count);
      $finish;
    end
    if (max_new_tokens !== 7) begin
      $error("max_new_tokens expected 7, got %0d", max_new_tokens);
      $finish;
    end
    if (eos_token_id !== 2) begin
      $error("eos_token_id expected 2, got %0d", eos_token_id);
      $finish;
    end
    if (!debug_enable || (debug_layer_sel != 5) || (debug_step_sel != 9)) begin
      $error("debug cfg mismatch enable=%0b layer=%0d step=%0d", debug_enable, debug_layer_sel, debug_step_sel);
      $finish;
    end

    axi_write_same_cycle({REGW_CONTROL, 2'b00}, (1 << CTRL_MODE_BIT) | (1 << CTRL_START_BIT));
    if (!start_pulse) begin
      $error("start_pulse expected HIGH after control write");
      $finish;
    end
    if (launch_mode != MODE_DECODE) begin
      $error("launch_mode expected MODE_DECODE, got %0d", launch_mode);
      $finish;
    end
    @(negedge clk);
    if (start_pulse) begin
      $error("start_pulse expected LOW after one cycle");
      $finish;
    end

    hw_busy = 1'b1;
    axi_read({REGW_STATUS, 2'b00}, readback);
    if (!readback[STATUS_BUSY_BIT]) begin
      $error("STATUS busy bit expected HIGH");
      $finish;
    end

    hw_stop_reason = STOP_REASON_MAX_TOKENS;
    hw_stop_valid  = 1'b1;
    hw_done_pulse  = 1'b1;
    @(negedge clk);
    hw_stop_valid  = 1'b0;
    hw_done_pulse  = 1'b0;
    hw_busy        = 1'b0;

    axi_read({REGW_STATUS, 2'b00}, readback);
    if (!readback[STATUS_DONE_BIT] || !readback[STATUS_STOP_VALID_BIT]) begin
      $error("STATUS done/stop_valid bits expected HIGH, got 0x%08h", readback);
      $finish;
    end
    if (readback[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] != STOP_REASON_MAX_TOKENS) begin
      $error("STATUS stop_reason expected MAX_TOKENS, got %0d", readback[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB]);
      $finish;
    end

    hw_generated_token_count = 3;
    hw_last_token_id         = 32'd42;
    hw_current_layer         = 5;
    hw_current_block         = BLOCK_GATE;
    @(negedge clk);

    axi_read({REGW_GENERATED_TOKEN_COUNT, 2'b00}, readback);
    if (readback != 3) begin
      $error("REGW_GENERATED_TOKEN_COUNT expected 3, got %0d", readback);
      $finish;
    end

    axi_read({REGW_LAST_TOKEN_ID, 2'b00}, readback);
    if (readback != 42) begin
      $error("REGW_LAST_TOKEN_ID expected 42, got %0d", readback);
      $finish;
    end

    axi_read({REGW_CURRENT_LAYER, 2'b00}, readback);
    if (readback != 5) begin
      $error("REGW_CURRENT_LAYER expected 5, got %0d", readback);
      $finish;
    end

    axi_read({REGW_CURRENT_BLOCK, 2'b00}, readback);
    if (readback != BLOCK_GATE) begin
      $error("REGW_CURRENT_BLOCK expected BLOCK_GATE, got %0d", readback);
      $finish;
    end

    hw_error_code  = ERROR_INTERNAL_ASSERT;
    hw_error_valid = 1'b1;
    @(negedge clk);
    hw_error_valid = 1'b0;

    axi_read({REGW_STATUS, 2'b00}, readback);
    if (!readback[STATUS_ERROR_BIT]) begin
      $error("STATUS error bit expected HIGH");
      $finish;
    end
    if (readback[STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB] != ERROR_INTERNAL_ASSERT) begin
      $error("STATUS error_code expected ERROR_INTERNAL_ASSERT, got %0d", readback[STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB]);
      $finish;
    end

    axi_write_same_cycle({REGW_CONTROL, 2'b00}, (1 << CTRL_START_BIT));
    if (!start_pulse) begin
      $error("start_pulse expected HIGH when clearing sticky status with START");
      $finish;
    end
    @(negedge clk);
    if (start_pulse) begin
      $error("start_pulse expected LOW one cycle after sticky-clear START");
      $finish;
    end

    axi_read({REGW_STATUS, 2'b00}, readback);
    if (readback[STATUS_DONE_BIT] || readback[STATUS_ERROR_BIT] || readback[STATUS_STOP_VALID_BIT]) begin
      $error("STATUS sticky bits expected LOW after START clear, got 0x%08h", readback);
      $finish;
    end
    if (readback[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] != STOP_REASON_NONE) begin
      $error("STATUS stop_reason expected NONE after START clear, got %0d", readback[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB]);
      $finish;
    end
    if (readback[STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB] != ERROR_NONE) begin
      $error("STATUS error_code expected NONE after START clear, got %0d", readback[STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB]);
      $finish;
    end

    axi_write_same_cycle({REGW_CONTROL, 2'b00}, (1 << CTRL_ABORT_REQ_BIT));
    if (!abort_req) begin
      $error("abort_req expected HIGH after abort control write");
      $finish;
    end

    axi_read({REGW_VERSION, 2'b00}, readback);
    if (readback != RTL_VERSION_WORD) begin
      $error("REGW_VERSION expected 0x%08h, got 0x%08h", RTL_VERSION_WORD, readback);
      $finish;
    end

    $display("PASS: tb_axi_lite_ctrl_slave");
    $finish;
  end

endmodule
