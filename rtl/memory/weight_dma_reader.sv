import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module weight_dma_reader (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  req_valid_i,
  output logic                  req_ready_o,
  input  logic [HBM_ADDR_W-1:0] base_addr_i,
  input  logic [31:0]           byte_count_i,
  input  logic [LAYER_ID_W-1:0] layer_id_i,
  input  tensor_id_e            tensor_id_i,
  input  logic [TILE_ID_W-1:0]  output_tile_id_i,
  input  logic [TILE_ID_W-1:0]  input_tile_id_i,
  output logic                  busy_o,
  output logic                  done_pulse_o,
  output logic                  error_valid_o,
  output error_code_e           error_code_o,
  output logic                  rd_desc_valid_o,
  input  logic                  rd_desc_ready_i,
  output dma_desc_t             rd_desc_o,
  input  logic                  rd_data_valid_i,
  input  logic [DMA_BEAT_W-1:0] rd_data_i,
  output logic                  rd_data_ready_o,
  output logic                  wt_valid_o,
  input  logic                  wt_ready_i,
  output wt_bus_t               wt_o
);

  typedef enum logic [2:0] {
    WDR_IDLE       = 3'd0,
    WDR_ISSUE_DESC = 3'd1,
    WDR_WAIT_DATA  = 3'd2,
    WDR_STREAM     = 3'd3,
    WDR_DONE       = 3'd4
  } wdr_state_e;

  wdr_state_e             state_q;
  dma_desc_t              req_desc_q;
  logic [DMA_BEAT_W-1:0]  beat_data_q;
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

  function automatic logic tensor_supported(
    input tensor_id_e tensor_id
  );
    begin
      unique case (tensor_id)
        TENSOR_WQ,
        TENSOR_WK,
        TENSOR_WV,
        TENSOR_WO,
        TENSOR_WGATE,
        TENSOR_WUP,
        TENSOR_WDOWN: tensor_supported = 1'b1;
        default:     tensor_supported = 1'b0;
      endcase
    end
  endfunction

  function automatic block_id_e block_id_from_tensor(
    input tensor_id_e tensor_id
  );
    begin
      unique case (tensor_id)
        TENSOR_WQ:    block_id_from_tensor = BLOCK_Q;
        TENSOR_WK:    block_id_from_tensor = BLOCK_K;
        TENSOR_WV:    block_id_from_tensor = BLOCK_V;
        TENSOR_WO:    block_id_from_tensor = BLOCK_O;
        TENSOR_WGATE: block_id_from_tensor = BLOCK_GATE;
        TENSOR_WUP:   block_id_from_tensor = BLOCK_UP;
        TENSOR_WDOWN: block_id_from_tensor = BLOCK_DOWN;
        default:      block_id_from_tensor = BLOCK_NONE;
      endcase
    end
  endfunction

  function automatic gemm_mode_e gemm_mode_from_tensor(
    input tensor_id_e tensor_id
  );
    begin
      unique case (tensor_id)
        TENSOR_WQ:    gemm_mode_from_tensor = GEMM_Q;
        TENSOR_WK:    gemm_mode_from_tensor = GEMM_K;
        TENSOR_WV:    gemm_mode_from_tensor = GEMM_V;
        TENSOR_WO:    gemm_mode_from_tensor = GEMM_O;
        TENSOR_WGATE: gemm_mode_from_tensor = GEMM_GATE;
        TENSOR_WUP:   gemm_mode_from_tensor = GEMM_UP;
        TENSOR_WDOWN: gemm_mode_from_tensor = GEMM_DOWN;
        default:      gemm_mode_from_tensor = GEMM_NONE;
      endcase
    end
  endfunction

  assign busy_o          = (state_q != WDR_IDLE);
  assign req_ready_o     = (state_q == WDR_IDLE);
  assign rd_desc_valid_o = (state_q == WDR_ISSUE_DESC);
  assign rd_data_ready_o = (state_q == WDR_WAIT_DATA);
  assign wt_valid_o      = (state_q == WDR_STREAM);
  assign rd_desc_o       = req_desc_q;

  always @* begin
    wt_o = '0;
    wt_o.tag.layer_id   = req_desc_q.layer_id;
    wt_o.tag.block_id   = block_id_from_tensor(req_desc_q.tensor_id);
    wt_o.tag.gemm_mode  = gemm_mode_from_tensor(req_desc_q.tensor_id);
    wt_o.tag.tile_id    = req_desc_q.tile_id;
    wt_o.tag.kv_head_id = req_desc_q.kv_head_id;
    wt_o.tag.elem_count = beat_bytes_q;
    wt_o.tag.is_partial = (beat_bytes_q != WEIGHT_VECTOR_ELEMS);
    wt_o.tag.is_last    = (beats_seen_q == beats_total_q);
    wt_o.data[DMA_BEAT_BYTES-1:0] = beat_data_q;
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o  <= 1'b0;
    error_valid_o <= 1'b0;
    error_code_o  <= ERROR_NONE;

    if (!ap_rst_n) begin
      state_q           <= WDR_IDLE;
      req_desc_q        <= '0;
      beat_data_q       <= '0;
      beats_total_q     <= '0;
      beats_seen_q      <= '0;
      bytes_remaining_q <= '0;
      beat_bytes_q      <= '0;
    end else begin
      unique case (state_q)
        WDR_IDLE: begin
          if (req_valid_i) begin
            if (!tensor_supported(tensor_id_i)) begin
              error_valid_o <= 1'b1;
              error_code_o  <= ERROR_BAD_DESCRIPTOR;
              done_pulse_o  <= 1'b1;
              state_q       <= WDR_DONE;
            end else begin
              req_desc_q.region         <= REGION_LAYER_WEIGHTS;
              req_desc_q.tensor_id      <= tensor_id_i;
              req_desc_q.write_not_read <= 1'b0;
              req_desc_q.pseudo_channel <= PC_ID_W'(output_tile_id_i[3:0]);
              req_desc_q.addr           <= base_addr_i;
              req_desc_q.burst_len      <= beats_from_bytes(byte_count_i);
              req_desc_q.byte_count     <= effective_byte_count(byte_count_i);
              req_desc_q.layer_id       <= layer_id_i;
              req_desc_q.kv_head_id     <= '0;
              req_desc_q.tile_id        <= output_tile_id_i ^ input_tile_id_i;
              beats_total_q             <= beats_from_bytes(byte_count_i);
              beats_seen_q              <= '0;
              bytes_remaining_q         <= effective_byte_count(byte_count_i);
              beat_bytes_q              <= '0;
              state_q                   <= WDR_ISSUE_DESC;
            end
          end
        end

        WDR_ISSUE_DESC: begin
          if (rd_desc_valid_o && rd_desc_ready_i) begin
            state_q <= WDR_WAIT_DATA;
          end
        end

        WDR_WAIT_DATA: begin
          if (rd_data_valid_i && rd_data_ready_o) begin
            beat_data_q       <= rd_data_i;
            beat_bytes_q      <= bytes_for_beat(bytes_remaining_q);
            beats_seen_q      <= beats_seen_q + 1'b1;
            bytes_remaining_q <= bytes_remaining_q - bytes_for_beat(bytes_remaining_q);
            state_q           <= WDR_STREAM;
          end
        end

        WDR_STREAM: begin
          if (wt_valid_o && wt_ready_i) begin
            if (beats_seen_q == beats_total_q) begin
              done_pulse_o <= 1'b1;
              state_q      <= WDR_DONE;
            end else begin
              state_q <= WDR_WAIT_DATA;
            end
          end
        end

        WDR_DONE: begin
          state_q <= WDR_IDLE;
        end

        default: begin
          state_q <= WDR_IDLE;
        end
      endcase
    end
  end

endmodule
