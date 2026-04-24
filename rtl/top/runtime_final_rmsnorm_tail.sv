import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module runtime_final_rmsnorm_tail (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  launch_i,
  input  logic                  abort_req_i,
  input  logic [HBM_ADDR_W-1:0] gamma_base_addr_i,
  input  logic [SCALE_W-1:0]    output_scale_i,
  output logic                  rd_desc_valid_o,
  input  logic                  rd_desc_ready_i,
  output dma_desc_t             rd_desc_o,
  input  logic                  rd_data_valid_i,
  output logic                  rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] rd_data_i,
  input  logic                  hidden_scale_valid_i,
  output logic                  hidden_scale_ready_o,
  input  scale_bus_t            hidden_scale_i,
  input  logic                  hidden_act_valid_i,
  output logic                  hidden_act_ready_o,
  input  act_bus_t              hidden_act_i,
  output logic                  norm_scale_valid_o,
  input  logic                  norm_scale_ready_i,
  output scale_bus_t            norm_scale_o,
  output logic                  norm_act_valid_o,
  input  logic                  norm_act_ready_i,
  output act_bus_t              norm_act_o,
  output logic                  norm_done_pulse_o,
  output logic                  busy_o
);

  localparam int unsigned FINAL_GAMMA_BYTES = D_MODEL * 2;

  logic                  helper_rst_n;
  logic                  gamma_req_valid_q;
  logic                  gamma_req_done_q;
  logic                  gamma_reader_busy;
  logic                  gamma_reader_done_pulse;
  logic                  gamma_valid;
  logic                  gamma_ready;
  logic [DMA_BEAT_W-1:0] gamma_beat;
  logic                  gamma_last;
  logic                  scale_seen_q;
  logic [SCALE_W-1:0]    input_scale_q;
  logic                  rms_act_ready;
  logic                  rms_busy;

  assign helper_rst_n = ap_rst_n && !abort_req_i;
  assign busy_o = gamma_req_valid_q || gamma_reader_busy || rms_busy || scale_seen_q;

  embedding_lmhead_dma_reader u_gamma_reader (
    .ap_clk            (ap_clk),
    .ap_rst_n          (helper_rst_n),
    .req_valid_i       (gamma_req_valid_q),
    .req_ready_o       (),
    .base_addr_i       (gamma_base_addr_i),
    .byte_count_i      (32'(FINAL_GAMMA_BYTES)),
    .layer_id_i        (LAYER_ID_W'(N_LAYERS - 1)),
    .tensor_id_i       (tensor_id_e'(TENSOR_FINAL_RMS_GAMMA)),
    .tile_id_i         (TILE_ID_W'(0)),
    .busy_o            (gamma_reader_busy),
    .done_pulse_o      (gamma_reader_done_pulse),
    .rd_desc_valid_o   (rd_desc_valid_o),
    .rd_desc_ready_i   (rd_desc_ready_i),
    .rd_desc_o         (rd_desc_o),
    .rd_data_valid_i   (rd_data_valid_i),
    .rd_data_i         (rd_data_i),
    .rd_data_ready_o   (rd_data_ready_o),
    .embed_row_valid_o (),
    .embed_row_ready_i (1'b0),
    .embed_row_o       (),
    .embed_row_last_o  (),
    .gamma_valid_o     (gamma_valid),
    .gamma_ready_i     (gamma_ready),
    .gamma_o           (gamma_beat),
    .gamma_last_o      (gamma_last),
    .lmhead_wt_valid_o (),
    .lmhead_wt_ready_i (1'b0),
    .lmhead_wt_o       (),
    .scale_valid_o     (),
    .scale_ready_i     (1'b0),
    .scale_tensor_id_o (),
    .scale_o           ()
  );

  rmsnorm_wrapper u_final_rmsnorm_wrapper (
    .ap_clk        (ap_clk),
    .ap_rst_n      (helper_rst_n),
    .block_id_i    (block_id_e'(BLOCK_FINAL_RMSNORM)),
    .act_valid_i   (hidden_act_valid_i && scale_seen_q),
    .act_ready_o   (rms_act_ready),
    .act_i         (hidden_act_i),
    .input_scale_i (input_scale_q),
    .output_scale_i(output_scale_i),
    .gamma_valid_i (gamma_valid),
    .gamma_ready_o (gamma_ready),
    .gamma_i       (gamma_beat),
    .gamma_last_i  (gamma_last),
    .scale_valid_o (norm_scale_valid_o),
    .scale_ready_i (norm_scale_ready_i),
    .scale_o       (norm_scale_o),
    .norm_valid_o  (norm_act_valid_o),
    .norm_ready_i  (norm_act_ready_i),
    .norm_o        (norm_act_o),
    .busy_o        (rms_busy),
    .done_pulse_o  (norm_done_pulse_o)
  );

  always_ff @(posedge ap_clk) begin
    if (!helper_rst_n) begin
      gamma_req_valid_q <= 1'b0;
      gamma_req_done_q <= 1'b0;
      scale_seen_q <= 1'b0;
      input_scale_q <= '0;
    end else begin
      if (launch_i) begin
        gamma_req_valid_q <= 1'b1;
        gamma_req_done_q <= 1'b0;
        scale_seen_q <= 1'b0;
        input_scale_q <= '0;
      end else if (rd_desc_valid_o && rd_desc_ready_i) begin
        gamma_req_valid_q <= 1'b0;
      end

      if (gamma_reader_done_pulse) begin
        gamma_req_done_q <= 1'b1;
      end

      if (hidden_scale_valid_i && hidden_scale_ready_o) begin
        scale_seen_q <= 1'b1;
        input_scale_q <= hidden_scale_i.data[0];
      end

      if (norm_done_pulse_o) begin
        scale_seen_q <= 1'b0;
      end
    end
  end

  assign hidden_scale_ready_o = !scale_seen_q;
  assign hidden_act_ready_o = scale_seen_q && rms_act_ready;

endmodule
