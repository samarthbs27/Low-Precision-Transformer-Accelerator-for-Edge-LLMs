import tinyllama_pkg::*;

module kernel_reg_file (
  input  logic                       ap_clk,
  input  logic                       ap_rst_n,
  input  logic                       reg_wr_en,
  input  logic [REG_WORD_ADDR_W-1:0] reg_wr_addr,
  input  logic [AXIL_DATA_W-1:0]     reg_wr_data,
  input  logic [AXIL_STRB_W-1:0]     reg_wr_strb,
  input  logic                       reg_rd_en,
  input  logic [REG_WORD_ADDR_W-1:0] reg_rd_addr,
  output logic [AXIL_DATA_W-1:0]     reg_rd_data,
  input  logic                       hw_busy_i,
  input  logic                       hw_done_pulse_i,
  input  logic                       hw_error_valid_i,
  input  error_code_e                hw_error_code_i,
  input  logic                       hw_stop_valid_i,
  input  stop_reason_e               hw_stop_reason_i,
  input  logic [COUNT_W-1:0]         hw_generated_token_count_i,
  input  logic [TOKEN_W-1:0]         hw_last_token_id_i,
  input  logic [LAYER_ID_W-1:0]      hw_current_layer_i,
  input  block_id_e                  hw_current_block_i,
  output logic                       start_pulse_o,
  output runtime_mode_e              launch_mode_o,
  output logic                       abort_req_o,
  output logic [HBM_ADDR_W-1:0]      cmd_base_addr_o,
  output logic [HBM_ADDR_W-1:0]      status_base_addr_o,
  output logic [HBM_ADDR_W-1:0]      debug_base_addr_o,
  output logic [COUNT_W-1:0]         prompt_token_count_o,
  output logic [COUNT_W-1:0]         max_new_tokens_o,
  output logic [TOKEN_W-1:0]         eos_token_id_o,
  output logic                       debug_enable_o,
  output logic [LAYER_ID_W-1:0]      debug_layer_sel_o,
  output logic [DEBUG_CFG_STEP_W-1:0] debug_step_sel_o
);

  logic [31:0] cmd_base_lo_q;
  logic [31:0] cmd_base_hi_q;
  logic [31:0] status_base_lo_q;
  logic [31:0] status_base_hi_q;
  logic [31:0] debug_base_lo_q;
  logic [31:0] debug_base_hi_q;
  logic [31:0] prompt_token_count_q;
  logic [31:0] max_new_tokens_q;
  logic [31:0] eos_token_id_q;
  logic [31:0] debug_cfg_q;
  logic [31:0] generated_token_count_q;
  logic [31:0] last_token_id_q;
  logic [31:0] current_layer_q;
  logic [31:0] current_block_q;
  logic        done_sticky_q;
  logic        error_sticky_q;
  logic        stop_valid_q;
  stop_reason_e stop_reason_q;
  error_code_e  error_code_q;
  runtime_mode_e launch_mode_q;
  logic         abort_req_q;

  function automatic logic [31:0] apply_wstrb32(
    input logic [31:0] curr,
    input logic [31:0] wdata,
    input logic [AXIL_STRB_W-1:0] wstrb
  );
    logic [31:0] result;
    int unsigned idx;
    begin
      result = curr;
      for (idx = 0; idx < AXIL_STRB_W; idx++) begin
        if (wstrb[idx]) begin
          result[idx*8 +: 8] = wdata[idx*8 +: 8];
        end
      end
      return result;
    end
  endfunction

  logic [31:0] status_word;

  assign status_word[STATUS_BUSY_BIT] = hw_busy_i;
  assign status_word[STATUS_DONE_BIT] = done_sticky_q;
  assign status_word[STATUS_ERROR_BIT] = error_sticky_q;
  assign status_word[STATUS_STOP_VALID_BIT] = stop_valid_q;
  assign status_word[STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] = stop_reason_q;
  assign status_word[STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB] = error_code_q;
  assign status_word[31:STATUS_ERROR_CODE_MSB+1] = '0;
  assign status_word[STATUS_ERROR_CODE_LSB-1:STATUS_STOP_REASON_MSB+1] = '0;

  assign launch_mode_o        = launch_mode_q;
  assign abort_req_o          = abort_req_q;
  assign cmd_base_addr_o      = {cmd_base_hi_q, cmd_base_lo_q};
  assign status_base_addr_o   = {status_base_hi_q, status_base_lo_q};
  assign debug_base_addr_o    = {debug_base_hi_q, debug_base_lo_q};
  assign prompt_token_count_o = prompt_token_count_q[COUNT_W-1:0];
  assign max_new_tokens_o     = max_new_tokens_q[COUNT_W-1:0];
  assign eos_token_id_o       = eos_token_id_q[TOKEN_W-1:0];
  assign debug_enable_o       = debug_cfg_q[DEBUG_CFG_ENABLE_BIT];
  assign debug_layer_sel_o    = debug_cfg_q[DEBUG_CFG_LAYER_MSB:DEBUG_CFG_LAYER_LSB];
  assign debug_step_sel_o     = debug_cfg_q[DEBUG_CFG_STEP_MSB:DEBUG_CFG_STEP_LSB];

  always_ff @(posedge ap_clk) begin
    start_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      cmd_base_lo_q           <= '0;
      cmd_base_hi_q           <= '0;
      status_base_lo_q        <= '0;
      status_base_hi_q        <= '0;
      debug_base_lo_q         <= '0;
      debug_base_hi_q         <= '0;
      prompt_token_count_q    <= '0;
      max_new_tokens_q        <= '0;
      eos_token_id_q          <= '0;
      debug_cfg_q             <= '0;
      generated_token_count_q <= '0;
      last_token_id_q         <= '0;
      current_layer_q         <= '0;
      current_block_q         <= '0;
      done_sticky_q           <= 1'b0;
      error_sticky_q          <= 1'b0;
      stop_valid_q            <= 1'b0;
      stop_reason_q           <= STOP_REASON_NONE;
      error_code_q            <= ERROR_NONE;
      launch_mode_q           <= MODE_PREFILL;
      abort_req_q             <= 1'b0;
    end else begin
      generated_token_count_q <= {{(32-COUNT_W){1'b0}}, hw_generated_token_count_i};
      last_token_id_q         <= hw_last_token_id_i;
      current_layer_q         <= {{(32-LAYER_ID_W){1'b0}}, hw_current_layer_i};
      current_block_q         <= {{(32-BLOCK_ID_W){1'b0}}, hw_current_block_i};

      if (hw_done_pulse_i) begin
        done_sticky_q <= 1'b1;
        abort_req_q   <= 1'b0;
      end

      if (hw_stop_valid_i) begin
        stop_valid_q  <= 1'b1;
        stop_reason_q <= hw_stop_reason_i;
      end

      if (hw_error_valid_i) begin
        error_sticky_q <= 1'b1;
        error_code_q   <= hw_error_code_i;
      end

      if (reg_wr_en) begin
        unique case (reg_wr_addr)
          REGW_CONTROL: begin
            if (reg_wr_strb[CTRL_START_BIT/8] && reg_wr_data[CTRL_START_BIT]) begin
              start_pulse_o <= 1'b1;
              done_sticky_q <= 1'b0;
              error_sticky_q <= 1'b0;
              stop_valid_q  <= 1'b0;
              stop_reason_q <= STOP_REASON_NONE;
              error_code_q  <= ERROR_NONE;
              abort_req_q   <= 1'b0;
            end

            if (reg_wr_strb[CTRL_MODE_BIT/8]) begin
              launch_mode_q <= runtime_mode_e'(reg_wr_data[CTRL_MODE_BIT]);
            end

            if (reg_wr_strb[CTRL_ABORT_REQ_BIT/8] && reg_wr_data[CTRL_ABORT_REQ_BIT]) begin
              abort_req_q <= 1'b1;
            end
          end

          REGW_CMD_BASE_LO: begin
            cmd_base_lo_q <= apply_wstrb32(cmd_base_lo_q, reg_wr_data, reg_wr_strb);
          end

          REGW_CMD_BASE_HI: begin
            cmd_base_hi_q <= apply_wstrb32(cmd_base_hi_q, reg_wr_data, reg_wr_strb);
          end

          REGW_STATUS_BASE_LO: begin
            status_base_lo_q <= apply_wstrb32(status_base_lo_q, reg_wr_data, reg_wr_strb);
          end

          REGW_STATUS_BASE_HI: begin
            status_base_hi_q <= apply_wstrb32(status_base_hi_q, reg_wr_data, reg_wr_strb);
          end

          REGW_DEBUG_BASE_LO: begin
            debug_base_lo_q <= apply_wstrb32(debug_base_lo_q, reg_wr_data, reg_wr_strb);
          end

          REGW_DEBUG_BASE_HI: begin
            debug_base_hi_q <= apply_wstrb32(debug_base_hi_q, reg_wr_data, reg_wr_strb);
          end

          REGW_PROMPT_TOKEN_COUNT: begin
            prompt_token_count_q <= apply_wstrb32(prompt_token_count_q, reg_wr_data, reg_wr_strb);
          end

          REGW_MAX_NEW_TOKENS: begin
            max_new_tokens_q <= apply_wstrb32(max_new_tokens_q, reg_wr_data, reg_wr_strb);
          end

          REGW_EOS_TOKEN_ID: begin
            eos_token_id_q <= apply_wstrb32(eos_token_id_q, reg_wr_data, reg_wr_strb);
          end

          REGW_DEBUG_CFG: begin
            debug_cfg_q <= apply_wstrb32(debug_cfg_q, reg_wr_data, reg_wr_strb);
          end

          default: begin
          end
        endcase
      end
    end
  end

  always_comb begin
    reg_rd_data = '0;

    unique case (reg_rd_addr)
      REGW_CONTROL: begin
        reg_rd_data[CTRL_MODE_BIT]      = launch_mode_q;
        reg_rd_data[CTRL_ABORT_REQ_BIT] = abort_req_q;
      end

      REGW_STATUS: begin
        reg_rd_data = status_word;
      end

      REGW_CMD_BASE_LO: begin
        reg_rd_data = cmd_base_lo_q;
      end

      REGW_CMD_BASE_HI: begin
        reg_rd_data = cmd_base_hi_q;
      end

      REGW_STATUS_BASE_LO: begin
        reg_rd_data = status_base_lo_q;
      end

      REGW_STATUS_BASE_HI: begin
        reg_rd_data = status_base_hi_q;
      end

      REGW_DEBUG_BASE_LO: begin
        reg_rd_data = debug_base_lo_q;
      end

      REGW_DEBUG_BASE_HI: begin
        reg_rd_data = debug_base_hi_q;
      end

      REGW_PROMPT_TOKEN_COUNT: begin
        reg_rd_data = prompt_token_count_q;
      end

      REGW_MAX_NEW_TOKENS: begin
        reg_rd_data = max_new_tokens_q;
      end

      REGW_EOS_TOKEN_ID: begin
        reg_rd_data = eos_token_id_q;
      end

      REGW_DEBUG_CFG: begin
        reg_rd_data = debug_cfg_q;
      end

      REGW_GENERATED_TOKEN_COUNT: begin
        reg_rd_data = generated_token_count_q;
      end

      REGW_LAST_TOKEN_ID: begin
        reg_rd_data = last_token_id_q;
      end

      REGW_CURRENT_LAYER: begin
        reg_rd_data = current_layer_q;
      end

      REGW_CURRENT_BLOCK: begin
        reg_rd_data = current_block_q;
      end

      REGW_VERSION: begin
        reg_rd_data = RTL_VERSION_WORD;
      end

      default: begin
        reg_rd_data = '0;
      end
    endcase
  end

endmodule
