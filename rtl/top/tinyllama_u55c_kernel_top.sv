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
  token_bus_t                 prompt_token;

  logic                       gen_writer_busy;
  logic                       gen_writer_token_ready;
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
  logic                       sim_block_done_q;

  logic                       sim_lm_head_done_q;
  logic                       sim_token_valid_q;
  logic [TOKEN_W-1:0]         sim_token_id_q;
  logic                       sim_lm_pending_q;
  logic                       sim_token_pending_q;
  dma_desc_t                  zero_desc;
  logic [DMA_BEAT_W-1:0]      zero_data;

  assign zero_desc = '0;
  assign zero_data = '0;

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
    .prompt_read_done_i       (prompt_done_pulse),
    .layer_pass_done_i        (layer_run_done),
    .lm_head_done_i           (sim_lm_head_done_q),
    .token_valid_i            (sim_token_valid_q),
    .token_id_i               (sim_token_id_q),
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
    .emitted_token_valid_i   (sim_token_valid_q),
    .emitted_token_id_i      (sim_token_id_q),
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
    .token_ready_i           (1'b1),
    .token_o                 (prompt_token)
  );

  generated_token_writer u_generated_token_writer (
    .ap_clk                    (ap_clk),
    .ap_rst_n                  (ap_rst_n),
    .start_i                   (token_writer_start),
    .generated_tokens_base_addr_i(generated_tokens_base_addr),
    .generated_tokens_capacity_i(generated_tokens_capacity),
    .token_valid_i             (sim_token_valid_q),
    .token_id_i                (sim_token_id_q),
    .token_ready_o             (gen_writer_token_ready),
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
    .block_done_i      (sim_block_done_q),
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
    .embed_lm_rd_desc_valid_i   (1'b0),
    .embed_lm_rd_desc_ready_o   (),
    .embed_lm_rd_desc_i         (zero_desc),
    .embed_lm_rd_data_valid_o   (),
    .embed_lm_rd_data_ready_i   (1'b0),
    .embed_lm_rd_data_o         (),
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
    end else if (prompt_busy) begin
      hw_current_block = BLOCK_EMBED;
    end else if (sim_lm_pending_q) begin
      hw_current_block = BLOCK_LM_HEAD;
    end else if (sim_token_pending_q || sim_token_valid_q) begin
      hw_current_block = BLOCK_ARGMAX;
    end else begin
      hw_current_block = BLOCK_NONE;
    end
  end

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      sim_block_done_q   <= 1'b0;
      sim_lm_head_done_q <= 1'b0;
      sim_token_valid_q  <= 1'b0;
      sim_token_id_q     <= '0;
      sim_lm_pending_q   <= 1'b0;
      sim_token_pending_q <= 1'b0;
      hw_last_token_id   <= '0;
    end else begin
      sim_block_done_q   <= block_start;
      sim_lm_head_done_q <= 1'b0;
      sim_token_valid_q  <= 1'b0;

      if (lm_head_start) begin
        sim_lm_pending_q    <= 1'b1;
        sim_token_pending_q <= 1'b0;
        sim_token_id_q      <= TOKEN_W'(32'd1000) + TOKEN_W'(ctrl_generated_token_count);
      end else if (sim_lm_pending_q) begin
        sim_lm_pending_q    <= 1'b0;
        sim_lm_head_done_q  <= 1'b1;
        sim_token_pending_q <= 1'b1;
      end else if (sim_token_pending_q && gen_writer_token_ready) begin
        sim_token_pending_q <= 1'b0;
        sim_token_valid_q   <= 1'b1;
        hw_last_token_id    <= sim_token_id_q;
      end

      if (start_pulse) begin
        hw_last_token_id <= '0;
      end
    end
  end

endmodule
