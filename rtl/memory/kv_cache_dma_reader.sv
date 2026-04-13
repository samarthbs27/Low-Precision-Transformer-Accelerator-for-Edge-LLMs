import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module kv_cache_dma_reader (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  req_valid_i,
  output logic                  req_ready_o,
  input  dma_desc_t             req_desc_i,
  output logic                  busy_o,
  output logic                  done_pulse_o,
  output logic                  rd_desc_valid_o,
  input  logic                  rd_desc_ready_i,
  output dma_desc_t             rd_desc_o,
  input  logic                  rd_data_valid_i,
  input  logic [DMA_BEAT_W-1:0] rd_data_i,
  output logic                  rd_data_ready_o,
  output logic                  kv_valid_o,
  input  logic                  kv_ready_i,
  output logic                  kv_is_v_o,
  output act_bus_t              kv_tile_o
);

  typedef enum logic [2:0] {
    KDR_IDLE       = 3'd0,
    KDR_ISSUE_DESC = 3'd1,
    KDR_WAIT_DATA  = 3'd2,
    KDR_STREAM     = 3'd3,
    KDR_DONE       = 3'd4
  } kdr_state_e;

  kdr_state_e             state_q;
  dma_desc_t              req_desc_q;
  logic [DMA_BEAT_W-1:0]  beat_data_q;
  logic [15:0]            beats_total_q;
  logic [15:0]            beats_seen_q;
  logic [31:0]            bytes_remaining_q;
  logic [15:0]            beat_bytes_q;

  function automatic logic [15:0] beats_from_bytes(
    input logic [31:0] byte_count
  );
    logic [31:0] beats_32;
    begin
      beats_32 = (((byte_count == '0) ? DMA_BEAT_BYTES : byte_count) + DMA_BEAT_BYTES - 1) / DMA_BEAT_BYTES;
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

  assign busy_o          = (state_q != KDR_IDLE);
  assign req_ready_o     = (state_q == KDR_IDLE);
  assign rd_desc_valid_o = (state_q == KDR_ISSUE_DESC);
  assign rd_data_ready_o = (state_q == KDR_WAIT_DATA);
  assign rd_desc_o       = req_desc_q;
  assign kv_valid_o      = (state_q == KDR_STREAM);
  assign kv_is_v_o       = (req_desc_q.region == REGION_V_CACHE);

  always @* begin
    kv_tile_o = '0;
    kv_tile_o.tag.layer_id   = req_desc_q.layer_id;
    kv_tile_o.tag.block_id   = kv_is_v_o ? BLOCK_V : BLOCK_K;
    kv_tile_o.tag.tile_id    = req_desc_q.tile_id;
    kv_tile_o.tag.kv_head_id = req_desc_q.kv_head_id;
    kv_tile_o.tag.elem_count = beat_bytes_q;
    kv_tile_o.tag.is_partial = (beat_bytes_q != ACT_VECTOR_ELEMS);
    kv_tile_o.tag.is_last    = (beats_seen_q == beats_total_q);
    kv_tile_o.data[DMA_BEAT_BYTES-1:0] = beat_data_q;
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q           <= KDR_IDLE;
      req_desc_q        <= '0;
      beat_data_q       <= '0;
      beats_total_q     <= '0;
      beats_seen_q      <= '0;
      bytes_remaining_q <= '0;
      beat_bytes_q      <= '0;
    end else begin
      unique case (state_q)
        KDR_IDLE: begin
          if (req_valid_i) begin
            req_desc_q        <= req_desc_i;
            beats_total_q     <= beats_from_bytes(req_desc_i.byte_count);
            beats_seen_q      <= '0;
            bytes_remaining_q <= (req_desc_i.byte_count == '0) ? DMA_BEAT_BYTES : req_desc_i.byte_count;
            beat_bytes_q      <= '0;
            state_q           <= KDR_ISSUE_DESC;
          end
        end

        KDR_ISSUE_DESC: begin
          if (rd_desc_valid_o && rd_desc_ready_i) begin
            state_q <= KDR_WAIT_DATA;
          end
        end

        KDR_WAIT_DATA: begin
          if (rd_data_valid_i && rd_data_ready_o) begin
            beat_data_q       <= rd_data_i;
            beat_bytes_q      <= bytes_for_beat(bytes_remaining_q);
            beats_seen_q      <= beats_seen_q + 1'b1;
            bytes_remaining_q <= bytes_remaining_q - bytes_for_beat(bytes_remaining_q);
            state_q           <= KDR_STREAM;
          end
        end

        KDR_STREAM: begin
          if (kv_valid_o && kv_ready_i) begin
            if (beats_seen_q == beats_total_q) begin
              done_pulse_o <= 1'b1;
              state_q      <= KDR_DONE;
            end else begin
              state_q <= KDR_WAIT_DATA;
            end
          end
        end

        KDR_DONE: begin
          state_q <= KDR_IDLE;
        end

        default: begin
          state_q <= KDR_IDLE;
        end
      endcase
    end
  end

endmodule
