import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module embedding_lmhead_dma_reader (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  req_valid_i,
  output logic                  req_ready_o,
  input  logic [HBM_ADDR_W-1:0] base_addr_i,
  input  logic [31:0]           byte_count_i,
  input  logic [LAYER_ID_W-1:0] layer_id_i,
  input  tensor_id_e            tensor_id_i,
  input  logic [TILE_ID_W-1:0]  tile_id_i,
  output logic                  busy_o,
  output logic                  done_pulse_o,
  output logic                  rd_desc_valid_o,
  input  logic                  rd_desc_ready_i,
  output dma_desc_t             rd_desc_o,
  input  logic                  rd_data_valid_i,
  input  logic [DMA_BEAT_W-1:0] rd_data_i,
  output logic                  rd_data_ready_o,
  output logic                  embed_row_valid_o,
  input  logic                  embed_row_ready_i,
  output logic [DMA_BEAT_W-1:0] embed_row_o,
  output logic                  embed_row_last_o,
  output logic                  gamma_valid_o,
  input  logic                  gamma_ready_i,
  output logic [DMA_BEAT_W-1:0] gamma_o,
  output logic                  gamma_last_o,
  output logic                  lmhead_wt_valid_o,
  input  logic                  lmhead_wt_ready_i,
  output wt_bus_t               lmhead_wt_o,
  output logic                  scale_valid_o,
  input  logic                  scale_ready_i,
  output tensor_id_e            scale_tensor_id_o,
  output scale_bus_t            scale_o
);

  localparam int unsigned SCALE_META_BEATS = (SCALE_VECTOR_ELEMS * SCALE_W) / DMA_BEAT_W;

  typedef enum logic [2:0] {
    EDR_IDLE         = 3'd0,
    EDR_ISSUE_DESC   = 3'd1,
    EDR_WAIT_DATA    = 3'd2,
    EDR_STREAM_RAW   = 3'd3,
    EDR_STREAM_WT    = 3'd4,
    EDR_STREAM_SCALE = 3'd5,
    EDR_DONE         = 3'd6
  } edr_state_e;

  edr_state_e             state_q;
  dma_desc_t              req_desc_q;
  tensor_id_e             tensor_id_q;
  logic [DMA_BEAT_W-1:0]  beat_data_q;
  logic [DMA_BEAT_W-1:0]  scale_beats_q [0:SCALE_META_BEATS-1];
  logic [15:0]            beats_total_q;
  logic [15:0]            beats_seen_q;
  logic [31:0]            bytes_remaining_q;
  logic [15:0]            beat_bytes_q;

  function automatic logic [31:0] effective_byte_count(
    input logic [31:0] byte_count
  );
    begin
      effective_byte_count = (byte_count == '0) ? DMA_BEAT_BYTES : byte_count;
    end
  endfunction

  function automatic logic [15:0] beats_from_bytes(
    input logic [31:0] byte_count
  );
    logic [31:0] beats_32;
    begin
      beats_32 = (effective_byte_count(byte_count) + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
      beats_from_bytes = beats_32[15:0];
    end
  endfunction

  function automatic logic [15:0] bytes_for_beat(
    input logic [31:0] bytes_remaining
  );
    begin
      if (bytes_remaining > DMA_BEAT_BYTES) begin
        bytes_for_beat = DMA_BEAT_BYTES;
      end else begin
        bytes_for_beat = bytes_remaining[15:0];
      end
    end
  endfunction

  function automatic logic is_gamma_tensor(
    input tensor_id_e tensor_id
  );
    begin
      unique case (tensor_id)
        TENSOR_FINAL_RMS_GAMMA,
        TENSOR_RMSNORM1_GAMMA,
        TENSOR_RMSNORM2_GAMMA: is_gamma_tensor = 1'b1;
        default:               is_gamma_tensor = 1'b0;
      endcase
    end
  endfunction

  assign busy_o            = (state_q != EDR_IDLE);
  assign req_ready_o       = (state_q == EDR_IDLE);
  assign rd_desc_valid_o   = (state_q == EDR_ISSUE_DESC);
  assign rd_data_ready_o   = (state_q == EDR_WAIT_DATA);
  assign embed_row_o       = beat_data_q;
  assign embed_row_last_o  = (beats_seen_q == beats_total_q);
  assign gamma_o           = beat_data_q;
  assign gamma_last_o      = (beats_seen_q == beats_total_q);
  assign scale_tensor_id_o = tensor_id_q;
  assign rd_desc_o         = req_desc_q;

  assign embed_row_valid_o = (state_q == EDR_STREAM_RAW) && (tensor_id_q == TENSOR_EMBED);
  assign gamma_valid_o     = (state_q == EDR_STREAM_RAW) && is_gamma_tensor(tensor_id_q);
  assign lmhead_wt_valid_o = (state_q == EDR_STREAM_WT) && (tensor_id_q == TENSOR_LM_HEAD);
  assign scale_valid_o     = (state_q == EDR_STREAM_SCALE) && (tensor_id_q == TENSOR_SCALE_META);

  always @* begin
    lmhead_wt_o = '0;
    lmhead_wt_o.tag.layer_id   = req_desc_q.layer_id;
    lmhead_wt_o.tag.block_id   = BLOCK_LM_HEAD;
    lmhead_wt_o.tag.gemm_mode  = GEMM_LM_HEAD;
    lmhead_wt_o.tag.tile_id    = req_desc_q.tile_id;
    lmhead_wt_o.tag.elem_count = beat_bytes_q;
    lmhead_wt_o.tag.is_partial = (beat_bytes_q != WEIGHT_VECTOR_ELEMS);
    lmhead_wt_o.tag.is_last    = (beats_seen_q == beats_total_q);
    lmhead_wt_o.data[DMA_BEAT_BYTES-1:0] = beat_data_q;

    scale_o = '0;
    scale_o.tag.layer_id   = req_desc_q.layer_id;
    scale_o.tag.block_id   = BLOCK_REQUANTIZE;
    scale_o.tag.gemm_mode  = GEMM_NONE;
    scale_o.tag.tile_id    = req_desc_q.tile_id;
    scale_o.tag.elem_count = effective_byte_count(req_desc_q.byte_count) / (SCALE_W / 8);
    scale_o.tag.is_partial = 1'b0;
    scale_o.tag.is_last    = 1'b1;
    scale_o.data[(DMA_BEAT_W / SCALE_W)-1:0] = scale_beats_q[0];
    if (SCALE_META_BEATS > 1) begin
      scale_o.data[(2 * DMA_BEAT_W / SCALE_W)-1:(DMA_BEAT_W / SCALE_W)] = scale_beats_q[1];
    end
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q           <= EDR_IDLE;
      req_desc_q        <= '0;
      tensor_id_q       <= TENSOR_NONE;
      beat_data_q       <= '0;
      for (int idx = 0; idx < SCALE_META_BEATS; idx++) begin
        scale_beats_q[idx] <= '0;
      end
      beats_total_q     <= '0;
      beats_seen_q      <= '0;
      bytes_remaining_q <= '0;
      beat_bytes_q      <= '0;
    end else begin
      unique case (state_q)
        EDR_IDLE: begin
          if (req_valid_i) begin
            req_desc_q.region         <= (tensor_id_i == TENSOR_LM_HEAD) ? REGION_LM_HEAD : REGION_EMBED_META;
            req_desc_q.tensor_id      <= tensor_id_i;
            req_desc_q.write_not_read <= 1'b0;
            if (tensor_id_i == TENSOR_LM_HEAD) begin
              req_desc_q.pseudo_channel <= PC_ID_W'(18 + tile_id_i[1:0]);
            end else begin
              req_desc_q.pseudo_channel <= PC_ID_W'(16 + tile_id_i[0]);
            end
            req_desc_q.addr           <= base_addr_i;
            req_desc_q.burst_len      <= beats_from_bytes(byte_count_i);
            req_desc_q.byte_count     <= effective_byte_count(byte_count_i);
            req_desc_q.layer_id       <= layer_id_i;
            req_desc_q.kv_head_id     <= '0;
            req_desc_q.tile_id        <= tile_id_i;
            tensor_id_q               <= tensor_id_i;
            beats_total_q             <= beats_from_bytes(byte_count_i);
            beats_seen_q              <= '0;
            bytes_remaining_q         <= effective_byte_count(byte_count_i);
            beat_bytes_q              <= '0;
            for (int idx = 0; idx < SCALE_META_BEATS; idx++) begin
              scale_beats_q[idx] <= '0;
            end
            state_q                   <= EDR_ISSUE_DESC;
          end
        end

        EDR_ISSUE_DESC: begin
          if (rd_desc_valid_o && rd_desc_ready_i) begin
            state_q <= EDR_WAIT_DATA;
          end
        end

        EDR_WAIT_DATA: begin
          if (rd_data_valid_i && rd_data_ready_o) begin
            beat_data_q       <= rd_data_i;
            beat_bytes_q      <= bytes_for_beat(bytes_remaining_q);
            beats_seen_q      <= beats_seen_q + 1'b1;
            bytes_remaining_q <= bytes_remaining_q - bytes_for_beat(bytes_remaining_q);

            if (tensor_id_q == TENSOR_SCALE_META) begin
              scale_beats_q[beats_seen_q] <= rd_data_i;
              if ((beats_seen_q + 1'b1) == beats_total_q) begin
                state_q <= EDR_STREAM_SCALE;
              end
            end else if (tensor_id_q == TENSOR_LM_HEAD) begin
              state_q <= EDR_STREAM_WT;
            end else begin
              state_q <= EDR_STREAM_RAW;
            end
          end
        end

        EDR_STREAM_RAW: begin
          if ((embed_row_valid_o && embed_row_ready_i) || (gamma_valid_o && gamma_ready_i)) begin
            if (beats_seen_q == beats_total_q) begin
              done_pulse_o <= 1'b1;
              state_q      <= EDR_DONE;
            end else begin
              state_q <= EDR_WAIT_DATA;
            end
          end
        end

        EDR_STREAM_WT: begin
          if (lmhead_wt_valid_o && lmhead_wt_ready_i) begin
            if (beats_seen_q == beats_total_q) begin
              done_pulse_o <= 1'b1;
              state_q      <= EDR_DONE;
            end else begin
              state_q <= EDR_WAIT_DATA;
            end
          end
        end

        EDR_STREAM_SCALE: begin
          if (scale_valid_o && scale_ready_i) begin
            done_pulse_o <= 1'b1;
            state_q      <= EDR_DONE;
          end
        end

        EDR_DONE: begin
          state_q <= EDR_IDLE;
        end

        default: begin
          state_q <= EDR_IDLE;
        end
      endcase
    end
  end

endmodule
