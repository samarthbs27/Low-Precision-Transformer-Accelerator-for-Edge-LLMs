import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module host_cmd_status_mgr (
  input  logic                   ap_clk,
  input  logic                   ap_rst_n,
  input  logic                   start_i,
  input  logic [HBM_ADDR_W-1:0]  cmd_base_addr_i,
  input  logic [HBM_ADDR_W-1:0]  status_base_addr_i,
  input  logic                   busy_i,
  input  logic                   done_pulse_i,
  input  logic                   error_valid_i,
  input  error_code_e            error_code_i,
  input  logic                   stop_valid_i,
  input  stop_reason_e           stop_reason_i,
  input  logic [COUNT_W-1:0]     generated_token_count_i,
  input  logic [TOKEN_W-1:0]     last_token_id_i,
  input  logic [LAYER_ID_W-1:0]  current_layer_i,
  input  block_id_e              current_block_i,
  output logic                   command_info_valid_o,
  output logic [HBM_ADDR_W-1:0]  prompt_tokens_base_addr_o,
  output logic [HBM_ADDR_W-1:0]  generated_tokens_base_addr_o,
  output logic [COUNT_W-1:0]     generated_tokens_capacity_o,
  output logic                   cmd_read_desc_valid_o,
  input  logic                   cmd_read_desc_ready_i,
  output dma_desc_t              cmd_read_desc_o,
  input  logic                   cmd_read_data_valid_i,
  input  logic [DMA_BEAT_W-1:0]  cmd_read_data_i,
  output logic                   cmd_read_data_ready_o,
  output logic                   status_write_desc_valid_o,
  input  logic                   status_write_desc_ready_i,
  output dma_desc_t              status_write_desc_o,
  output logic                   status_write_data_valid_o,
  input  logic                   status_write_data_ready_i,
  output logic [DMA_BEAT_W-1:0]  status_write_data_o
);

  logic        cmd_wait_data_q;
  logic        cmd_desc_issued_q;
  logic        status_done_sticky_q;
  logic        status_error_sticky_q;
  logic        status_stop_sticky_q;
  stop_reason_e status_stop_reason_q;
  error_code_e  status_error_code_q;
  logic [DMA_BEAT_W-1:0] status_payload_q;
  logic        status_desc_pending_q;
  logic        status_data_pending_q;
  logic        status_done_event;
  logic        status_error_event;
  logic        status_stop_event;
  stop_reason_e status_stop_reason_event;
  error_code_e  status_error_code_event;

  logic [31:0] status_words [0:HOST_BLOCK_WORDS-1];
  logic [31:0] cmd_words [0:HOST_BLOCK_WORDS-1];

  assign cmd_read_desc_o.region         = REGION_HOST_IO;
  assign cmd_read_desc_o.tensor_id      = TENSOR_NONE;
  assign cmd_read_desc_o.write_not_read = 1'b0;
  assign cmd_read_desc_o.pseudo_channel = HOST_IO_PC_ID;
  assign cmd_read_desc_o.addr           = cmd_base_addr_i;
  assign cmd_read_desc_o.burst_len      = 16'd1;
  assign cmd_read_desc_o.byte_count     = HOST_BLOCK_BYTES;
  assign cmd_read_desc_o.layer_id       = '0;
  assign cmd_read_desc_o.kv_head_id     = '0;
  assign cmd_read_desc_o.tile_id        = '0;

  assign status_write_desc_o.region         = REGION_HOST_IO;
  assign status_write_desc_o.tensor_id      = TENSOR_NONE;
  assign status_write_desc_o.write_not_read = 1'b1;
  assign status_write_desc_o.pseudo_channel = HOST_IO_PC_ID;
  assign status_write_desc_o.addr           = status_base_addr_i;
  assign status_write_desc_o.burst_len      = 16'd1;
  assign status_write_desc_o.byte_count     = HOST_BLOCK_BYTES;
  assign status_write_desc_o.layer_id       = '0;
  assign status_write_desc_o.kv_head_id     = '0;
  assign status_write_desc_o.tile_id        = '0;

  assign cmd_read_desc_valid_o     = cmd_desc_issued_q && !cmd_wait_data_q;
  assign cmd_read_data_ready_o     = cmd_wait_data_q;
  assign status_write_desc_valid_o = status_desc_pending_q;
  assign status_write_data_valid_o = status_data_pending_q;
  assign status_write_data_o       = status_payload_q;
  assign status_done_event         = status_done_sticky_q  || done_pulse_i;
  assign status_error_event        = status_error_sticky_q || error_valid_i;
  assign status_stop_event         = status_stop_sticky_q  || stop_valid_i;

  always_comb begin
    status_stop_reason_event = status_stop_reason_q;
    status_error_code_event  = status_error_code_q;

    if (stop_valid_i) begin
      status_stop_reason_event = stop_reason_i;
    end

    if (error_valid_i) begin
      status_error_code_event = error_code_i;
    end

    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      cmd_words[word_idx] = cmd_read_data_i[word_idx*AXIL_DATA_W +: AXIL_DATA_W];
    end

    for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
      status_words[word_idx] = '0;
    end

    status_words[HOST_STATUS_WORD_STATUS][STATUS_BUSY_BIT] = busy_i;
    status_words[HOST_STATUS_WORD_STATUS][STATUS_DONE_BIT] = status_done_event;
    status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_BIT] = status_error_event;
    status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_VALID_BIT] = status_stop_event;
    status_words[HOST_STATUS_WORD_STATUS][STATUS_STOP_REASON_MSB:STATUS_STOP_REASON_LSB] = status_stop_reason_event;
    status_words[HOST_STATUS_WORD_STATUS][STATUS_ERROR_CODE_MSB:STATUS_ERROR_CODE_LSB] = status_error_code_event;
    status_words[HOST_STATUS_WORD_GEN_COUNT] = generated_token_count_i;
    status_words[HOST_STATUS_WORD_LAST_TOKEN] = last_token_id_i;
    status_words[HOST_STATUS_WORD_CUR_LAYER] = {{(32-LAYER_ID_W){1'b0}}, current_layer_i};
    status_words[HOST_STATUS_WORD_CUR_BLOCK] = {{(32-BLOCK_ID_W){1'b0}}, current_block_i};
    status_words[HOST_STATUS_WORD_VERSION] = RTL_VERSION_WORD;
  end

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      command_info_valid_o        <= 1'b0;
      prompt_tokens_base_addr_o   <= '0;
      generated_tokens_base_addr_o <= '0;
      generated_tokens_capacity_o <= '0;
      cmd_wait_data_q             <= 1'b0;
      cmd_desc_issued_q           <= 1'b0;
      status_done_sticky_q        <= 1'b0;
      status_error_sticky_q       <= 1'b0;
      status_stop_sticky_q        <= 1'b0;
      status_stop_reason_q        <= STOP_REASON_NONE;
      status_error_code_q         <= ERROR_NONE;
      status_payload_q            <= '0;
      status_desc_pending_q       <= 1'b0;
      status_data_pending_q       <= 1'b0;
    end else begin
      if (start_i) begin
        command_info_valid_o   <= 1'b0;
        cmd_desc_issued_q      <= 1'b1;
        cmd_wait_data_q        <= 1'b0;
        status_done_sticky_q   <= 1'b0;
        status_error_sticky_q  <= 1'b0;
        status_stop_sticky_q   <= 1'b0;
        status_stop_reason_q   <= STOP_REASON_NONE;
        status_error_code_q    <= ERROR_NONE;
      end

      if (cmd_read_desc_valid_o && cmd_read_desc_ready_i) begin
        cmd_wait_data_q <= 1'b1;
      end

      if (cmd_wait_data_q && cmd_read_data_valid_i) begin
        prompt_tokens_base_addr_o    <= {cmd_words[HOST_CMD_WORD_PROMPT_BASE_HI], cmd_words[HOST_CMD_WORD_PROMPT_BASE_LO]};
        generated_tokens_base_addr_o <= {cmd_words[HOST_CMD_WORD_GEN_BASE_HI], cmd_words[HOST_CMD_WORD_GEN_BASE_LO]};
        generated_tokens_capacity_o  <= cmd_words[HOST_CMD_WORD_GEN_CAPACITY][COUNT_W-1:0];
        command_info_valid_o         <= 1'b1;
        cmd_wait_data_q              <= 1'b0;
        cmd_desc_issued_q            <= 1'b0;
      end

      if (done_pulse_i) begin
        status_done_sticky_q <= 1'b1;
      end

      if (error_valid_i) begin
        status_error_sticky_q <= 1'b1;
        status_error_code_q   <= error_code_i;
      end

      if (stop_valid_i) begin
        status_stop_sticky_q <= 1'b1;
        status_stop_reason_q <= stop_reason_i;
      end

      if (done_pulse_i || error_valid_i || stop_valid_i) begin
        for (int word_idx = 0; word_idx < HOST_BLOCK_WORDS; word_idx++) begin
          status_payload_q[word_idx*AXIL_DATA_W +: AXIL_DATA_W] <= status_words[word_idx];
        end
        status_desc_pending_q <= 1'b1;
        status_data_pending_q <= 1'b1;
      end

      if (status_desc_pending_q && status_write_desc_ready_i) begin
        status_desc_pending_q <= 1'b0;
      end

      if (status_data_pending_q && status_write_data_ready_i) begin
        status_data_pending_q <= 1'b0;
      end
    end
  end

endmodule
