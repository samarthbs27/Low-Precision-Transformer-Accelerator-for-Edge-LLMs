import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module tinyllama_u55c_kernel_top (
  input  logic                        ap_clk,
  input  logic                        ap_rst_n,
  input  logic [AXIL_ADDR_W-1:0]      s_axi_awaddr,
  input  logic                        s_axi_awvalid,
  output logic                        s_axi_awready,
  input  logic [AXIL_DATA_W-1:0]      s_axi_wdata,
  input  logic [AXIL_STRB_W-1:0]      s_axi_wstrb,
  input  logic                        s_axi_wvalid,
  output logic                        s_axi_wready,
  output logic [1:0]                  s_axi_bresp,
  output logic                        s_axi_bvalid,
  input  logic                        s_axi_bready,
  input  logic [AXIL_ADDR_W-1:0]      s_axi_araddr,
  input  logic                        s_axi_arvalid,
  output logic                        s_axi_arready,
  output logic [AXIL_DATA_W-1:0]      s_axi_rdata,
  output logic [1:0]                  s_axi_rresp,
  output logic                        s_axi_rvalid,
  input  logic                        s_axi_rready,
  output logic                        interrupt,
  output logic                        shell_rd_desc_valid_o,
  input  logic                        shell_rd_desc_ready_i,
  output dma_desc_t                   shell_rd_desc_o,
  input  logic                        shell_rd_data_valid_i,
  output logic                        shell_rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0]       shell_rd_data_i,
  output logic                        shell_wr_desc_valid_o,
  input  logic                        shell_wr_desc_ready_i,
  output dma_desc_t                   shell_wr_desc_o,
  output logic                        shell_wr_data_valid_o,
  input  logic                        shell_wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0]       shell_wr_data_o
);

  localparam logic [HBM_ADDR_W-1:0] EMBEDDING_BASE_ADDR = 64'h0000_0000_1000_0000;
  localparam logic [HBM_ADDR_W-1:0] EMBEDDING_SCALE_META_BASE_ADDR = 64'h0000_0000_0400_0000;
  localparam logic [HBM_ADDR_W-1:0] FINAL_RMS_GAMMA_BASE_ADDR = 64'h0000_0000_0800_0000;
  localparam logic [HBM_ADDR_W-1:0] LM_HEAD_BASE_ADDR = 64'h0000_0000_2000_0000;
  // Keep the runtime final-RMSNorm output scale on a stable dedicated source.
  // This is a temporary fixed integration contract until a real configured
  // final-RMSNorm output-scale path is wired into the runtime top.
  localparam logic [SCALE_W-1:0] FINAL_RMS_OUTPUT_SCALE_Q16 = 32'h0001_0000;

  logic                       reg_wr_en;
  logic [REG_WORD_ADDR_W-1:0] reg_wr_addr;
  logic [AXIL_DATA_W-1:0]     reg_wr_data;
  logic [AXIL_STRB_W-1:0]     reg_wr_strb;
  logic                       reg_rd_en;
  logic [REG_WORD_ADDR_W-1:0] reg_rd_addr;
  logic [AXIL_DATA_W-1:0]     reg_rd_data;

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

  logic                       hw_busy;
  logic                       hw_done_pulse;
  logic                       hw_error_valid;
  error_code_e                hw_error_code;
  logic                       hw_stop_valid;
  stop_reason_e               hw_stop_reason;
  logic                       stop_now_w;
  stop_reason_e               stop_reason_w;
  logic [COUNT_W-1:0]         hw_generated_token_count;
  logic [TOKEN_W-1:0]         hw_last_token_id;
  logic [LAYER_ID_W-1:0]      hw_current_layer;
  block_id_e                  hw_current_block;

  logic                       command_info_valid;
  logic [HBM_ADDR_W-1:0]      prompt_tokens_base_addr;
  logic [HBM_ADDR_W-1:0]      generated_tokens_base_addr;
  logic [COUNT_W-1:0]         generated_tokens_capacity;

  logic                       ctrl_done_pulse;
  logic                       ctrl_error_pulse;
  logic                       ctrl_stop_valid;
  stop_reason_e               ctrl_stop_reason;
  error_code_e                ctrl_error_code;
  logic [COUNT_W-1:0]         ctrl_generated_token_count;
  runtime_mode_e              ctrl_active_mode;
  logic                       prompt_read_start;
  logic                       token_writer_start;
  logic                       embedding_start;
  logic                       runtime_layer_start;
  logic                       lm_head_start;
  logic                       argmax_start;

  logic                       prompt_busy;
  logic                       prompt_done_pulse;
  logic                       prompt_error_valid;
  error_code_e                prompt_error_code;
  logic                       prompt_rd_desc_valid;
  logic                       prompt_rd_desc_ready;
  dma_desc_t                  prompt_rd_desc;
  logic                       prompt_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      prompt_rd_data;
  logic                       prompt_rd_data_ready;
  logic                       prompt_token_valid;
  logic                       prompt_token_ready;
  token_bus_t                 prompt_token;
  logic                       frontend_launch;
  logic                       frontend_token_valid;
  logic                       frontend_token_ready;
  token_bus_t                 frontend_token;
  logic                       decode_token_pending_q;
  token_bus_t                 decode_token_q;
  logic                       frontend_rd_desc_valid;
  logic                       frontend_rd_desc_ready;
  dma_desc_t                  frontend_rd_desc;
  logic                       frontend_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      frontend_rd_data;
  logic                       frontend_rd_data_ready;
  logic                       embed_rd_desc_valid;
  logic                       embed_rd_desc_ready;
  dma_desc_t                  embed_rd_desc;
  logic                       embed_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      embed_rd_data;
  logic                       embed_rd_data_ready;
  logic                       embed_scale_valid;
  scale_bus_t                 embed_scale_bus;
  logic                       embed_scale_ready;
  logic                       embed_act_valid;
  act_bus_t                   embed_act_bus;
  logic                       embed_act_ready;
  logic                       embed_busy;
  logic                       embed_done_pulse;

  logic                       gen_writer_busy;
  logic                       gen_wr_desc_valid;
  logic                       gen_wr_desc_ready;
  dma_desc_t                  gen_wr_desc;
  logic                       gen_wr_data_valid;
  logic                       gen_wr_data_ready;
  logic [DMA_BEAT_W-1:0]      gen_wr_data;

  logic                       host_cmd_rd_desc_valid;
  logic                       host_cmd_rd_desc_ready;
  dma_desc_t                  host_cmd_rd_desc;
  logic                       host_cmd_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      host_cmd_rd_data;
  logic                       host_cmd_rd_data_ready;
  logic                       host_status_wr_desc_valid;
  logic                       host_status_wr_desc_ready;
  dma_desc_t                  host_status_wr_desc;
  logic                       host_status_wr_data_valid;
  logic                       host_status_wr_data_ready;
  logic [DMA_BEAT_W-1:0]      host_status_wr_data;

  logic                       layer_busy;
  logic                       layer_run_done;
  logic                       per_layer_start;
  logic                       layer_ctx_valid;
  logic                       block_valid;
  logic                       block_start;
  runtime_mode_e              layer_runtime_mode;
  logic [LAYER_ID_W-1:0]      layer_id;
  logic [LAYER_ID_W-1:0]      weight_layer_sel;
  logic [LAYER_ID_W-1:0]      kv_layer_sel;
  block_id_e                  layer_block_id;
  logic [Q_HEAD_ID_W-1:0]     q_head_id;
  logic [KV_HEAD_ID_W-1:0]    kv_head_id;
  logic                       decoder_context_valid;
  logic                       decoder_block_done;
  logic                       decoder_final_scale_valid;
  scale_bus_t                 decoder_final_scale_bus;
  logic                       decoder_final_scale_ready;
  logic                       decoder_final_act_valid;
  act_bus_t                   decoder_final_act_bus;
  logic                       decoder_final_act_ready;
  logic                       decoder_final_hidden_done_pulse;
  logic                       decoder_busy;
  logic                       final_rms_rd_desc_valid;
  logic                       final_rms_rd_desc_ready;
  dma_desc_t                  final_rms_rd_desc;
  logic                       final_rms_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      final_rms_rd_data;
  logic                       final_rms_rd_data_ready;
  logic                       final_rms_scale_valid;
  scale_bus_t                 final_rms_scale_bus;
  logic                       final_rms_scale_ready;
  logic                       final_rms_act_valid;
  act_bus_t                   final_rms_act_bus;
  logic                       final_rms_act_ready;
  logic                       final_rms_done_pulse;
  logic                       final_rms_busy;
  logic                       runtime_lm_head_done_pulse;
  logic                       runtime_token_valid;
  logic                       runtime_token_ready;
  logic [TOKEN_W-1:0]         runtime_token_id;
  logic signed [ACC_W-1:0]    runtime_token_logit;
  logic                       runtime_token_fire_q;
  logic [TOKEN_W-1:0]         runtime_token_fire_id_q;
  logic                       runtime_lm_context_valid;
  logic                       runtime_lm_busy;
  logic                       runtime_argmax_busy;
  logic                       lm_head_rd_desc_valid;
  logic                       lm_head_rd_desc_ready;
  dma_desc_t                  lm_head_rd_desc;
  logic                       lm_head_rd_data_valid;
  logic [DMA_BEAT_W-1:0]      lm_head_rd_data;
  logic                       lm_head_rd_data_ready;

  dma_desc_t                  zero_desc;
  logic [DMA_BEAT_W-1:0]      zero_data;

  typedef enum logic [1:0] {
    EMRD_NONE     = 2'd0,
    EMRD_FRONTEND = 2'd1,
    EMRD_FINAL    = 2'd2,
    EMRD_LM_HEAD  = 2'd3
  } embed_rd_sel_e;

  embed_rd_sel_e              embed_rd_sel_w;
  embed_rd_sel_e              embed_rd_active_client_q;
  logic                       embed_rd_active_q;
  logic [15:0]                embed_rd_beats_remaining_q;

  function automatic logic [15:0] embed_beats_from_byte_count(
    input logic [31:0] byte_count
  );
    logic [31:0] beats_32;
    begin
      if (byte_count == '0) begin
        beats_32 = 32'd1;
      end else begin
        beats_32 = (byte_count + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
      end

      if (beats_32 == '0) begin
        embed_beats_from_byte_count = 16'd1;
      end else if (beats_32 > 16'hffff) begin
        embed_beats_from_byte_count = 16'hffff;
      end else begin
        embed_beats_from_byte_count = beats_32[15:0];
      end
    end
  endfunction

  assign zero_desc = '0;
  assign zero_data = '0;
  assign frontend_launch = start_pulse || embedding_start;
  assign frontend_token_valid = decode_token_pending_q || prompt_token_valid;
  assign frontend_token = decode_token_pending_q ? decode_token_q : prompt_token;
  assign prompt_token_ready = frontend_token_ready && !decode_token_pending_q;

  axi_lite_ctrl_slave u_axi_lite_ctrl_slave (
    .ap_clk       (ap_clk),
    .ap_rst_n     (ap_rst_n),
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

  kernel_reg_file u_kernel_reg_file (
    .ap_clk                   (ap_clk),
    .ap_rst_n                 (ap_rst_n),
    .reg_wr_en                (reg_wr_en),
    .reg_wr_addr              (reg_wr_addr),
    .reg_wr_data              (reg_wr_data),
    .reg_wr_strb              (reg_wr_strb),
    .reg_rd_en                (reg_rd_en),
    .reg_rd_addr              (reg_rd_addr),
    .reg_rd_data              (reg_rd_data),
    .hw_busy_i                (hw_busy),
    .hw_done_pulse_i          (hw_done_pulse),
    .hw_error_valid_i         (hw_error_valid),
    .hw_error_code_i          (hw_error_code),
    .hw_stop_valid_i          (hw_stop_valid),
    .hw_stop_reason_i         (hw_stop_reason),
    .hw_generated_token_count_i(hw_generated_token_count),
    .hw_last_token_id_i       (hw_last_token_id),
    .hw_current_layer_i       (hw_current_layer),
    .hw_current_block_i       (hw_current_block),
    .start_pulse_o            (start_pulse),
    .launch_mode_o            (launch_mode),
    .abort_req_o              (abort_req),
    .cmd_base_addr_o          (cmd_base_addr),
    .status_base_addr_o       (status_base_addr),
    .debug_base_addr_o        (debug_base_addr),
    .prompt_token_count_o     (prompt_token_count),
    .max_new_tokens_o         (max_new_tokens),
    .eos_token_id_o           (eos_token_id),
    .debug_enable_o           (debug_enable),
    .debug_layer_sel_o        (debug_layer_sel),
    .debug_step_sel_o         (debug_step_sel)
  );

  host_cmd_status_mgr u_host_cmd_status_mgr (
    .ap_clk                    (ap_clk),
    .ap_rst_n                  (ap_rst_n),
    .start_i                   (start_pulse),
    .cmd_base_addr_i           (cmd_base_addr),
    .status_base_addr_i        (status_base_addr),
    .busy_i                    (hw_busy),
    .done_pulse_i              (hw_done_pulse),
    .error_valid_i             (hw_error_valid),
    .error_code_i              (hw_error_code),
    .stop_valid_i              (hw_stop_valid),
    .stop_reason_i             (hw_stop_reason),
    .generated_token_count_i   (hw_generated_token_count),
    .last_token_id_i           (hw_last_token_id),
    .current_layer_i           (hw_current_layer),
    .current_block_i           (hw_current_block),
    .command_info_valid_o      (command_info_valid),
    .prompt_tokens_base_addr_o (prompt_tokens_base_addr),
    .generated_tokens_base_addr_o(generated_tokens_base_addr),
    .generated_tokens_capacity_o(generated_tokens_capacity),
    .cmd_read_desc_valid_o     (host_cmd_rd_desc_valid),
    .cmd_read_desc_ready_i     (host_cmd_rd_desc_ready),
    .cmd_read_desc_o           (host_cmd_rd_desc),
    .cmd_read_data_valid_i     (host_cmd_rd_data_valid),
    .cmd_read_data_i           (host_cmd_rd_data),
    .cmd_read_data_ready_o     (host_cmd_rd_data_ready),
    .status_write_desc_valid_o (host_status_wr_desc_valid),
    .status_write_desc_ready_i (host_status_wr_desc_ready),
    .status_write_desc_o       (host_status_wr_desc),
    .status_write_data_valid_o (host_status_wr_data_valid),
    .status_write_data_ready_i (host_status_wr_data_ready),
    .status_write_data_o       (host_status_wr_data)
  );

  prefill_decode_controller u_prefill_decode_controller (
    .ap_clk                   (ap_clk),
    .ap_rst_n                 (ap_rst_n),
    .start_i                  (start_pulse),
    .abort_req_i              (abort_req),
    .launch_mode_i            (launch_mode),
    .prompt_token_count_i     (prompt_token_count),
    .max_new_tokens_i         (max_new_tokens),
    .command_info_valid_i     (command_info_valid),
    .prompt_read_done_i       (embed_done_pulse),
    .layer_pass_done_i        (layer_run_done),
    .lm_head_done_i           (runtime_lm_head_done_pulse),
    .token_valid_i            (runtime_token_fire_q),
    .token_id_i               (runtime_token_fire_id_q),
    .stop_now_i               (stop_now_w),
    .stop_reason_i            (stop_reason_w),
    .error_valid_i            (prompt_error_valid),
    .error_code_i             (prompt_error_code),
    .busy_o                   (hw_busy),
    .done_pulse_o             (ctrl_done_pulse),
    .error_pulse_o            (ctrl_error_pulse),
    .stop_valid_o             (ctrl_stop_valid),
    .stop_reason_o            (ctrl_stop_reason),
    .error_code_o             (ctrl_error_code),
    .generated_token_count_o  (ctrl_generated_token_count),
    .active_mode_o            (ctrl_active_mode),
    .prefill_active_o         (),
    .decode_active_o          (),
    .prompt_read_start_o      (prompt_read_start),
    .token_writer_start_o     (token_writer_start),
    .embedding_start_o        (embedding_start),
    .layer_start_o            (runtime_layer_start),
    .lm_head_start_o          (lm_head_start),
    .argmax_start_o           (argmax_start)
  );

  stop_condition_unit u_stop_condition_unit (
    .abort_req_i             (abort_req),
    .emitted_token_valid_i   (runtime_token_fire_q),
    .emitted_token_id_i      (runtime_token_fire_id_q),
    .generated_token_count_i (ctrl_generated_token_count),
    .max_new_tokens_i        (max_new_tokens),
    .eos_token_id_i          (eos_token_id),
    .stop_now_o              (stop_now_w),
    .stop_reason_o           (stop_reason_w)
  );

  prompt_token_reader u_prompt_token_reader (
    .ap_clk                  (ap_clk),
    .ap_rst_n                (ap_rst_n),
    .start_i                 (prompt_read_start),
    .prompt_tokens_base_addr_i(prompt_tokens_base_addr),
    .prompt_token_count_i    (prompt_token_count),
    .busy_o                  (prompt_busy),
    .done_pulse_o            (prompt_done_pulse),
    .error_valid_o           (prompt_error_valid),
    .error_code_o            (prompt_error_code),
    .rd_desc_valid_o         (prompt_rd_desc_valid),
    .rd_desc_ready_i         (prompt_rd_desc_ready),
    .rd_desc_o               (prompt_rd_desc),
    .rd_data_valid_i         (prompt_rd_data_valid),
    .rd_data_i               (prompt_rd_data),
    .rd_data_ready_o         (prompt_rd_data_ready),
    .token_valid_o           (prompt_token_valid),
    .token_ready_i           (prompt_token_ready),
    .token_o                 (prompt_token)
  );

  runtime_embedding_frontend u_runtime_embedding_frontend (
    .ap_clk               (ap_clk),
    .ap_rst_n             (ap_rst_n),
    .launch_i             (frontend_launch),
    .embedding_base_addr_i(EMBEDDING_BASE_ADDR),
    .scale_meta_base_addr_i(EMBEDDING_SCALE_META_BASE_ADDR),
    .token_valid_i        (frontend_token_valid),
    .token_ready_o        (frontend_token_ready),
    .token_i              (frontend_token),
    .rd_desc_valid_o      (frontend_rd_desc_valid),
    .rd_desc_ready_i      (frontend_rd_desc_ready),
    .rd_desc_o            (frontend_rd_desc),
    .rd_data_valid_i      (frontend_rd_data_valid),
    .rd_data_ready_o      (frontend_rd_data_ready),
    .rd_data_i            (frontend_rd_data),
    .scale_valid_o        (embed_scale_valid),
    .scale_ready_i        (embed_scale_ready),
    .scale_o              (embed_scale_bus),
    .act_valid_o          (embed_act_valid),
    .act_ready_i          (embed_act_ready),
    .act_o                (embed_act_bus),
    .busy_o               (embed_busy),
    .done_pulse_o         (embed_done_pulse)
  );

  runtime_decoder_datapath u_runtime_decoder_datapath (
    .ap_clk                  (ap_clk),
    .ap_rst_n                (ap_rst_n),
    .launch_i                (frontend_launch),
    .abort_req_i             (abort_req),
    .embed_scale_valid_i     (embed_scale_valid),
    .embed_scale_ready_o     (embed_scale_ready),
    .embed_scale_i           (embed_scale_bus),
    .embed_act_valid_i       (embed_act_valid),
    .embed_act_ready_o       (embed_act_ready),
    .embed_act_i             (embed_act_bus),
    .block_valid_i           (block_valid),
    .block_start_i           (block_start),
    .runtime_mode_i          (layer_runtime_mode),
    .layer_id_i              (layer_id),
    .block_id_i              (layer_block_id),
    .q_head_id_i             (q_head_id),
    .kv_head_id_i            (kv_head_id),
    .context_valid_o         (decoder_context_valid),
    .block_done_o            (decoder_block_done),
    .final_scale_valid_o     (decoder_final_scale_valid),
    .final_scale_ready_i     (decoder_final_scale_ready),
    .final_scale_o           (decoder_final_scale_bus),
    .final_act_valid_o       (decoder_final_act_valid),
    .final_act_ready_i       (decoder_final_act_ready),
    .final_act_o             (decoder_final_act_bus),
    .final_hidden_done_pulse_o(decoder_final_hidden_done_pulse),
    .busy_o                  (decoder_busy)
  );

  runtime_final_rmsnorm_tail u_runtime_final_rmsnorm_tail (
    .ap_clk              (ap_clk),
    .ap_rst_n            (ap_rst_n),
    .launch_i            (frontend_launch),
    .abort_req_i         (abort_req),
    .gamma_base_addr_i   (FINAL_RMS_GAMMA_BASE_ADDR),
    .output_scale_i      (FINAL_RMS_OUTPUT_SCALE_Q16),
    .rd_desc_valid_o     (final_rms_rd_desc_valid),
    .rd_desc_ready_i     (final_rms_rd_desc_ready),
    .rd_desc_o           (final_rms_rd_desc),
    .rd_data_valid_i     (final_rms_rd_data_valid),
    .rd_data_ready_o     (final_rms_rd_data_ready),
    .rd_data_i           (final_rms_rd_data),
    .hidden_scale_valid_i(decoder_final_scale_valid),
    .hidden_scale_ready_o(decoder_final_scale_ready),
    .hidden_scale_i      (decoder_final_scale_bus),
    .hidden_act_valid_i  (decoder_final_act_valid),
    .hidden_act_ready_o  (decoder_final_act_ready),
    .hidden_act_i        (decoder_final_act_bus),
    .norm_scale_valid_o  (final_rms_scale_valid),
    .norm_scale_ready_i  (final_rms_scale_ready),
    .norm_scale_o        (final_rms_scale_bus),
    .norm_act_valid_o    (final_rms_act_valid),
    .norm_act_ready_i    (final_rms_act_ready),
    .norm_act_o          (final_rms_act_bus),
    .norm_done_pulse_o   (final_rms_done_pulse),
    .busy_o              (final_rms_busy)
  );

  runtime_lm_head_tail u_runtime_lm_head_tail (
    .ap_clk                (ap_clk),
    .ap_rst_n              (ap_rst_n),
    .launch_i              (frontend_launch),
    .abort_req_i           (abort_req),
    .start_i               (lm_head_start || argmax_start),
    .lmhead_base_addr_i    (LM_HEAD_BASE_ADDR),
    .rd_desc_valid_o       (lm_head_rd_desc_valid),
    .rd_desc_ready_i       (lm_head_rd_desc_ready),
    .rd_desc_o             (lm_head_rd_desc),
    .rd_data_valid_i       (lm_head_rd_data_valid),
    .rd_data_ready_o       (lm_head_rd_data_ready),
    .rd_data_i             (lm_head_rd_data),
    .hidden_scale_valid_i  (final_rms_scale_valid),
    .hidden_scale_ready_o  (final_rms_scale_ready),
    .hidden_scale_i        (final_rms_scale_bus),
    .hidden_act_valid_i    (final_rms_act_valid),
    .hidden_act_ready_o    (final_rms_act_ready),
    .hidden_act_i          (final_rms_act_bus),
    .hidden_done_pulse_i   (final_rms_done_pulse),
    .lm_head_done_pulse_o  (runtime_lm_head_done_pulse),
    .token_valid_o         (runtime_token_valid),
    .token_ready_i         (runtime_token_ready),
    .token_id_o            (runtime_token_id),
    .token_logit_o         (runtime_token_logit),
    .context_valid_o       (runtime_lm_context_valid),
    .lm_head_busy_o        (runtime_lm_busy),
    .argmax_busy_o         (runtime_argmax_busy),
    .busy_o                ()
  );

  always_comb begin
    embed_rd_sel_w = EMRD_NONE;
    embed_rd_desc = '0;
    if (frontend_rd_desc_valid) begin
      embed_rd_sel_w = EMRD_FRONTEND;
      embed_rd_desc = frontend_rd_desc;
    end else if (final_rms_rd_desc_valid) begin
      embed_rd_sel_w = EMRD_FINAL;
      embed_rd_desc = final_rms_rd_desc;
    end else if (lm_head_rd_desc_valid) begin
      embed_rd_sel_w = EMRD_LM_HEAD;
      embed_rd_desc = lm_head_rd_desc;
    end
  end

  assign embed_rd_desc_valid = !embed_rd_active_q && (embed_rd_sel_w != EMRD_NONE);
  assign frontend_rd_desc_ready = !embed_rd_active_q && embed_rd_desc_ready &&
                                  (embed_rd_sel_w == EMRD_FRONTEND);
  assign final_rms_rd_desc_ready = !embed_rd_active_q && embed_rd_desc_ready &&
                                   (embed_rd_sel_w == EMRD_FINAL);
  assign lm_head_rd_desc_ready = !embed_rd_active_q && embed_rd_desc_ready &&
                                 (embed_rd_sel_w == EMRD_LM_HEAD);
  assign frontend_rd_data_valid = embed_rd_active_q &&
                                  (embed_rd_active_client_q == EMRD_FRONTEND) &&
                                  embed_rd_data_valid;
  assign final_rms_rd_data_valid = embed_rd_active_q &&
                                   (embed_rd_active_client_q == EMRD_FINAL) &&
                                   embed_rd_data_valid;
  assign lm_head_rd_data_valid = embed_rd_active_q &&
                                 (embed_rd_active_client_q == EMRD_LM_HEAD) &&
                                 embed_rd_data_valid;
  assign frontend_rd_data = embed_rd_data;
  assign final_rms_rd_data = embed_rd_data;
  assign lm_head_rd_data = embed_rd_data;
  assign embed_rd_data_ready = embed_rd_active_q &&
                               (((embed_rd_active_client_q == EMRD_FRONTEND) && frontend_rd_data_ready) ||
                                ((embed_rd_active_client_q == EMRD_FINAL) && final_rms_rd_data_ready) ||
                                ((embed_rd_active_client_q == EMRD_LM_HEAD) && lm_head_rd_data_ready));

  generated_token_writer u_generated_token_writer (
    .ap_clk                    (ap_clk),
    .ap_rst_n                  (ap_rst_n),
    .start_i                   (token_writer_start),
    .generated_tokens_base_addr_i(generated_tokens_base_addr),
    .generated_tokens_capacity_i(generated_tokens_capacity),
    .token_valid_i             (runtime_token_valid),
    .token_id_i                (runtime_token_id),
    .token_ready_o             (runtime_token_ready),
    .busy_o                    (gen_writer_busy),
    .wr_desc_valid_o           (gen_wr_desc_valid),
    .wr_desc_ready_i           (gen_wr_desc_ready),
    .wr_desc_o                 (gen_wr_desc),
    .wr_data_valid_o           (gen_wr_data_valid),
    .wr_data_ready_i           (gen_wr_data_ready),
    .wr_data_o                 (gen_wr_data)
  );

  layer_controller u_layer_controller (
    .ap_clk            (ap_clk),
    .ap_rst_n          (ap_rst_n),
    .start_i           (runtime_layer_start),
    .abort_req_i       (abort_req),
    .runtime_mode_i    (ctrl_active_mode),
    .block_done_i      (decoder_block_done),
    .busy_o            (layer_busy),
    .run_done_o        (layer_run_done),
    .layer_start_o     (per_layer_start),
    .layer_ctx_valid_o (layer_ctx_valid),
    .block_valid_o     (block_valid),
    .block_start_o     (block_start),
    .runtime_mode_o    (layer_runtime_mode),
    .layer_id_o        (layer_id),
    .weight_layer_sel_o(weight_layer_sel),
    .kv_layer_sel_o    (kv_layer_sel),
    .block_id_o        (layer_block_id),
    .q_head_id_o       (q_head_id),
    .kv_head_id_o      (kv_head_id)
  );

  hbm_port_router u_hbm_port_router (
    .ap_clk                     (ap_clk),
    .ap_rst_n                   (ap_rst_n),
    .host_cmd_rd_desc_valid_i   (host_cmd_rd_desc_valid),
    .host_cmd_rd_desc_ready_o   (host_cmd_rd_desc_ready),
    .host_cmd_rd_desc_i         (host_cmd_rd_desc),
    .host_cmd_rd_data_valid_o   (host_cmd_rd_data_valid),
    .host_cmd_rd_data_ready_i   (host_cmd_rd_data_ready),
    .host_cmd_rd_data_o         (host_cmd_rd_data),
    .prompt_rd_desc_valid_i     (prompt_rd_desc_valid),
    .prompt_rd_desc_ready_o     (prompt_rd_desc_ready),
    .prompt_rd_desc_i           (prompt_rd_desc),
    .prompt_rd_data_valid_o     (prompt_rd_data_valid),
    .prompt_rd_data_ready_i     (prompt_rd_data_ready),
    .prompt_rd_data_o           (prompt_rd_data),
    .weight_rd_desc_valid_i     (1'b0),
    .weight_rd_desc_ready_o     (),
    .weight_rd_desc_i           (zero_desc),
    .weight_rd_data_valid_o     (),
    .weight_rd_data_ready_i     (1'b0),
    .weight_rd_data_o           (),
    .embed_lm_rd_desc_valid_i   (embed_rd_desc_valid),
    .embed_lm_rd_desc_ready_o   (embed_rd_desc_ready),
    .embed_lm_rd_desc_i         (embed_rd_desc),
    .embed_lm_rd_data_valid_o   (embed_rd_data_valid),
    .embed_lm_rd_data_ready_i   (embed_rd_data_ready),
    .embed_lm_rd_data_o         (embed_rd_data),
    .kv_rd_desc_valid_i         (1'b0),
    .kv_rd_desc_ready_o         (),
    .kv_rd_desc_i               (zero_desc),
    .kv_rd_data_valid_o         (),
    .kv_rd_data_ready_i         (1'b0),
    .kv_rd_data_o               (),
    .host_status_wr_desc_valid_i(host_status_wr_desc_valid),
    .host_status_wr_desc_ready_o(host_status_wr_desc_ready),
    .host_status_wr_desc_i      (host_status_wr_desc),
    .host_status_wr_data_valid_i(host_status_wr_data_valid),
    .host_status_wr_data_ready_o(host_status_wr_data_ready),
    .host_status_wr_data_i      (host_status_wr_data),
    .gen_token_wr_desc_valid_i  (gen_wr_desc_valid),
    .gen_token_wr_desc_ready_o  (gen_wr_desc_ready),
    .gen_token_wr_desc_i        (gen_wr_desc),
    .gen_token_wr_data_valid_i  (gen_wr_data_valid),
    .gen_token_wr_data_ready_o  (gen_wr_data_ready),
    .gen_token_wr_data_i        (gen_wr_data),
    .kv_wr_desc_valid_i         (1'b0),
    .kv_wr_desc_ready_o         (),
    .kv_wr_desc_i               (zero_desc),
    .kv_wr_data_valid_i         (1'b0),
    .kv_wr_data_ready_o         (),
    .kv_wr_data_i               (zero_data),
    .debug_wr_desc_valid_i      (1'b0),
    .debug_wr_desc_ready_o      (),
    .debug_wr_desc_i            (zero_desc),
    .debug_wr_data_valid_i      (1'b0),
    .debug_wr_data_ready_o      (),
    .debug_wr_data_i            (zero_data),
    .shell_rd_desc_valid_o      (shell_rd_desc_valid_o),
    .shell_rd_desc_ready_i      (shell_rd_desc_ready_i),
    .shell_rd_desc_o            (shell_rd_desc_o),
    .shell_rd_data_valid_i      (shell_rd_data_valid_i),
    .shell_rd_data_ready_o      (shell_rd_data_ready_o),
    .shell_rd_data_i            (shell_rd_data_i),
    .shell_wr_desc_valid_o      (shell_wr_desc_valid_o),
    .shell_wr_desc_ready_i      (shell_wr_desc_ready_i),
    .shell_wr_desc_o            (shell_wr_desc_o),
    .shell_wr_data_valid_o      (shell_wr_data_valid_o),
    .shell_wr_data_ready_i      (shell_wr_data_ready_i),
    .shell_wr_data_o            (shell_wr_data_o)
  );

  assign hw_done_pulse = ctrl_done_pulse;
  assign hw_error_valid = ctrl_error_pulse || prompt_error_valid;
  assign hw_stop_valid = ctrl_stop_valid;
  assign hw_stop_reason = ctrl_stop_reason;
  assign hw_generated_token_count = ctrl_generated_token_count;
  assign interrupt = hw_done_pulse || hw_error_valid || hw_stop_valid;

  always_comb begin
    if (prompt_error_valid) begin
      hw_error_code = prompt_error_code;
    end else begin
      hw_error_code = ctrl_error_code;
    end

    hw_current_layer = layer_busy ? layer_id : LAYER_ID_W'(0);

    if (layer_busy) begin
      hw_current_block = layer_block_id;
    end else if (embed_busy || prompt_busy) begin
      hw_current_block = BLOCK_EMBED;
    end else if (final_rms_busy || final_rms_scale_valid || final_rms_act_valid) begin
      hw_current_block = BLOCK_FINAL_RMSNORM;
    end else if (runtime_lm_busy) begin
      hw_current_block = BLOCK_LM_HEAD;
    end else if (runtime_argmax_busy || runtime_token_valid) begin
      hw_current_block = BLOCK_ARGMAX;
    end else begin
      hw_current_block = BLOCK_NONE;
    end
  end

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      embed_rd_active_q <= 1'b0;
      embed_rd_active_client_q <= EMRD_NONE;
      embed_rd_beats_remaining_q <= '0;
      runtime_token_fire_q <= 1'b0;
      runtime_token_fire_id_q <= '0;
      decode_token_pending_q <= 1'b0;
      decode_token_q <= '0;
      hw_last_token_id   <= '0;
    end else begin
      runtime_token_fire_q <= 1'b0;

      if (start_pulse || abort_req) begin
        embed_rd_active_q <= 1'b0;
        embed_rd_active_client_q <= EMRD_NONE;
        embed_rd_beats_remaining_q <= '0;
        decode_token_pending_q <= 1'b0;
      end else if (decode_token_pending_q && frontend_token_ready) begin
        decode_token_pending_q <= 1'b0;
      end

      if (!embed_rd_active_q && embed_rd_desc_valid && embed_rd_desc_ready) begin
        embed_rd_active_q <= 1'b1;
        embed_rd_active_client_q <= embed_rd_sel_w;
        embed_rd_beats_remaining_q <= embed_beats_from_byte_count(embed_rd_desc.byte_count);
      end

      if (embed_rd_active_q && embed_rd_data_valid && embed_rd_data_ready) begin
        if (embed_rd_beats_remaining_q <= 16'd1) begin
          embed_rd_active_q <= 1'b0;
          embed_rd_active_client_q <= EMRD_NONE;
          embed_rd_beats_remaining_q <= '0;
        end else begin
          embed_rd_beats_remaining_q <= embed_rd_beats_remaining_q - 1'b1;
        end
      end

      if (runtime_token_valid && runtime_token_ready) begin
        runtime_token_fire_q <= 1'b1;
        runtime_token_fire_id_q <= runtime_token_id;
        hw_last_token_id <= runtime_token_id;
      end

      if (runtime_token_fire_q && !stop_now_w) begin
        decode_token_pending_q <= 1'b1;
        decode_token_q.token_id <= runtime_token_fire_id_q;
        decode_token_q.token_count <= COUNT_W'(prompt_token_count + ctrl_generated_token_count + COUNT_W'(1));
        decode_token_q.tag.layer_id <= '0;
        decode_token_q.tag.block_id <= BLOCK_EMBED;
        decode_token_q.tag.gemm_mode <= GEMM_NONE;
        decode_token_q.tag.tile_id <= TILE_ID_W'(prompt_token_count + ctrl_generated_token_count);
        decode_token_q.tag.token_base <= POS_W'(prompt_token_count + ctrl_generated_token_count);
        decode_token_q.tag.seq_count <= COUNT_W'(1);
        decode_token_q.tag.q_head_id <= '0;
        decode_token_q.tag.kv_head_id <= '0;
        decode_token_q.tag.elem_count <= 16'd1;
        decode_token_q.tag.is_last <= 1'b1;
        decode_token_q.tag.is_partial <= 1'b1;
      end

      if (start_pulse) begin
        hw_last_token_id <= '0;
      end
    end
  end

endmodule
