import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module runtime_embedding_frontend (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  launch_i,
  input  logic [HBM_ADDR_W-1:0] embedding_base_addr_i,
  input  logic [HBM_ADDR_W-1:0] scale_meta_base_addr_i,
  input  logic                  token_valid_i,
  output logic                  token_ready_o,
  input  token_bus_t            token_i,
  output logic                  rd_desc_valid_o,
  input  logic                  rd_desc_ready_i,
  output dma_desc_t             rd_desc_o,
  input  logic                  rd_data_valid_i,
  output logic                  rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] rd_data_i,
  output logic                  scale_valid_o,
  input  logic                  scale_ready_i,
  output scale_bus_t            scale_o,
  output logic                  act_valid_o,
  input  logic                  act_ready_i,
  output act_bus_t              act_o,
  output logic                  busy_o,
  output logic                  done_pulse_o
);

  localparam int unsigned SCALE_META_BYTES = SCALE_VECTOR_ELEMS * (SCALE_W / 8);

  typedef enum logic [1:0] {
    REF_UNARMED    = 2'd0,
    REF_SCALE_REQ  = 2'd1,
    REF_SCALE_WAIT = 2'd2,
    REF_READY      = 2'd3
  } ref_state_e;

  ref_state_e                state_q;
  logic                      reader_req_valid;
  logic                      reader_req_ready;
  logic [HBM_ADDR_W-1:0]     reader_base_addr;
  logic [31:0]               reader_byte_count;
  logic [LAYER_ID_W-1:0]     reader_layer_id;
  tensor_id_e                reader_tensor_id;
  logic [TILE_ID_W-1:0]      reader_tile_id;
  logic                      reader_busy;
  logic                      reader_done_pulse;
  logic                      reader_embed_row_valid;
  logic                      reader_embed_row_ready;
  logic [DMA_BEAT_W-1:0]     reader_embed_row;
  logic                      reader_embed_row_last;
  logic                      reader_scale_valid;
  logic                      reader_scale_ready;
  scale_bus_t                reader_scale_bus;
  logic                      lookup_req_valid;
  logic                      lookup_req_ready;
  logic [HBM_ADDR_W-1:0]     lookup_req_base_addr;
  logic [31:0]               lookup_req_byte_count;
  tensor_id_e                lookup_req_tensor_id;
  logic [LAYER_ID_W-1:0]     lookup_req_layer_id;
  logic [TILE_ID_W-1:0]      lookup_req_tile_id;
  logic                      lookup_token_ready;
  logic                      lookup_row_valid;
  logic                      lookup_row_ready;
  logic [(D_MODEL * 16)-1:0] lookup_row_fp16;
  token_bus_t                lookup_row_meta;
  logic                      lookup_busy;
  logic                      quant_busy;
  logic                      quant_done_pulse;

  assign reader_req_valid = (state_q == REF_SCALE_REQ) || ((state_q == REF_READY) && lookup_req_valid);
  assign reader_base_addr = (state_q == REF_SCALE_REQ) ? scale_meta_base_addr_i : lookup_req_base_addr;
  assign reader_byte_count = (state_q == REF_SCALE_REQ) ? 32'(SCALE_META_BYTES) : lookup_req_byte_count;
  assign reader_layer_id = (state_q == REF_SCALE_REQ) ? '0 : lookup_req_layer_id;
  assign reader_tile_id = (state_q == REF_SCALE_REQ) ? '0 : lookup_req_tile_id;
  assign lookup_req_ready = (state_q == REF_READY) && reader_req_ready;
  assign token_ready_o = (state_q == REF_READY) && lookup_token_ready;
  assign busy_o = (state_q != REF_UNARMED && state_q != REF_READY) ||
                  lookup_busy || reader_busy || quant_busy;
  assign done_pulse_o = quant_done_pulse;

  always_comb begin
    if (state_q == REF_SCALE_REQ) begin
      reader_tensor_id = TENSOR_SCALE_META;
    end else begin
      reader_tensor_id = lookup_req_tensor_id;
    end
  end

  embedding_lmhead_dma_reader u_embedding_lmhead_dma_reader (
    .ap_clk           (ap_clk),
    .ap_rst_n         (ap_rst_n),
    .req_valid_i      (reader_req_valid),
    .req_ready_o      (reader_req_ready),
    .base_addr_i      (reader_base_addr),
    .byte_count_i     (reader_byte_count),
    .layer_id_i       (reader_layer_id),
    .tensor_id_i      (reader_tensor_id),
    .tile_id_i        (reader_tile_id),
    .busy_o           (reader_busy),
    .done_pulse_o     (reader_done_pulse),
    .rd_desc_valid_o  (rd_desc_valid_o),
    .rd_desc_ready_i  (rd_desc_ready_i),
    .rd_desc_o        (rd_desc_o),
    .rd_data_valid_i  (rd_data_valid_i),
    .rd_data_i        (rd_data_i),
    .rd_data_ready_o  (rd_data_ready_o),
    .embed_row_valid_o(reader_embed_row_valid),
    .embed_row_ready_i(reader_embed_row_ready),
    .embed_row_o      (reader_embed_row),
    .embed_row_last_o (reader_embed_row_last),
    .gamma_valid_o    (),
    .gamma_ready_i    (1'b0),
    .gamma_o          (),
    .gamma_last_o     (),
    .lmhead_wt_valid_o(),
    .lmhead_wt_ready_i(1'b0),
    .lmhead_wt_o      (),
    .scale_valid_o    (reader_scale_valid),
    .scale_ready_i    (reader_scale_ready),
    .scale_tensor_id_o(),
    .scale_o          (reader_scale_bus)
  );

  embedding_lookup u_embedding_lookup (
    .ap_clk               (ap_clk),
    .ap_rst_n             (ap_rst_n),
    .embedding_base_addr_i(embedding_base_addr_i),
    .token_valid_i        (token_valid_i),
    .token_ready_o        (lookup_token_ready),
    .token_i              (token_i),
    .req_valid_o          (lookup_req_valid),
    .req_ready_i          (lookup_req_ready),
    .req_base_addr_o      (lookup_req_base_addr),
    .req_byte_count_o     (lookup_req_byte_count),
    .req_tensor_id_o      (lookup_req_tensor_id),
    .req_layer_id_o       (lookup_req_layer_id),
    .req_tile_id_o        (lookup_req_tile_id),
    .embed_row_valid_i    (reader_embed_row_valid),
    .embed_row_ready_o    (reader_embed_row_ready),
    .embed_row_i          (reader_embed_row),
    .embed_row_last_i     (reader_embed_row_last),
    .row_valid_o          (lookup_row_valid),
    .row_ready_i          (lookup_row_ready),
    .row_fp16_o           (lookup_row_fp16),
    .row_meta_o           (lookup_row_meta),
    .busy_o               (lookup_busy),
    .done_pulse_o         ()
  );

  embedding_quantizer u_embedding_quantizer (
    .ap_clk           (ap_clk),
    .ap_rst_n         (ap_rst_n),
    .row_valid_i      (lookup_row_valid),
    .row_ready_o      (lookup_row_ready),
    .row_fp16_i       (lookup_row_fp16),
    .row_meta_i       (lookup_row_meta),
    .scale_valid_i    (reader_scale_valid),
    .scale_ready_o    (reader_scale_ready),
    .scale_i          (reader_scale_bus),
    .scale_out_valid_o(scale_valid_o),
    .scale_out_ready_i(scale_ready_i),
    .scale_out_o      (scale_o),
    .act_valid_o      (act_valid_o),
    .act_ready_i      (act_ready_i),
    .act_o            (act_o),
    .busy_o           (quant_busy),
    .done_pulse_o     (quant_done_pulse)
  );

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      state_q <= REF_UNARMED;
    end else begin
      if (launch_i) begin
        state_q <= REF_SCALE_REQ;
      end else begin
        unique case (state_q)
          REF_UNARMED: begin
            state_q <= REF_UNARMED;
          end

          REF_SCALE_REQ: begin
            if (reader_req_valid && reader_req_ready) begin
              state_q <= REF_SCALE_WAIT;
            end
          end

          REF_SCALE_WAIT: begin
            if (reader_done_pulse) begin
              state_q <= REF_READY;
            end
          end

          REF_READY: begin
            state_q <= REF_READY;
          end

          default: begin
            state_q <= REF_UNARMED;
          end
        endcase
      end
    end
  end

endmodule
