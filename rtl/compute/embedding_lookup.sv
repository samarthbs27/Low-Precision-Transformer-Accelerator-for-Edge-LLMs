import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module embedding_lookup (
  input  logic                               ap_clk,
  input  logic                               ap_rst_n,
  input  logic [HBM_ADDR_W-1:0]              embedding_base_addr_i,
  input  logic                               token_valid_i,
  output logic                               token_ready_o,
  input  token_bus_t                         token_i,
  output logic                               req_valid_o,
  input  logic                               req_ready_i,
  output logic [HBM_ADDR_W-1:0]              req_base_addr_o,
  output logic [31:0]                        req_byte_count_o,
  output tensor_id_e                         req_tensor_id_o,
  output logic [LAYER_ID_W-1:0]              req_layer_id_o,
  output logic [TILE_ID_W-1:0]               req_tile_id_o,
  input  logic                               embed_row_valid_i,
  output logic                               embed_row_ready_o,
  input  logic [DMA_BEAT_W-1:0]              embed_row_i,
  input  logic                               embed_row_last_i,
  output logic                               row_valid_o,
  input  logic                               row_ready_i,
  output logic [(D_MODEL * 16)-1:0]          row_fp16_o,
  output token_bus_t                         row_meta_o,
  output logic                               busy_o,
  output logic                               done_pulse_o
);

  localparam int unsigned EMBED_ELEM_W    = 16;
  localparam int unsigned EMBED_ROW_W     = D_MODEL * EMBED_ELEM_W;
  localparam int unsigned EMBED_ROW_BYTES = D_MODEL * (EMBED_ELEM_W / 8);
  localparam int unsigned EMBED_ROW_BEATS = EMBED_ROW_BYTES / DMA_BEAT_BYTES;

  typedef enum logic [1:0] {
    EL_IDLE      = 2'd0,
    EL_REQ       = 2'd1,
    EL_RECV_ROW  = 2'd2,
    EL_OUT       = 2'd3
  } el_state_e;

  el_state_e                    state_q;
  token_bus_t                   token_q;
  logic [(D_MODEL * 16)-1:0]    row_fp16_q;
  logic [$clog2(EMBED_ROW_BEATS + 1)-1:0] beat_idx_q;

  function automatic logic [HBM_ADDR_W-1:0] row_addr_from_token(
    input logic [HBM_ADDR_W-1:0] base_addr,
    input logic [TOKEN_W-1:0]    token_id
  );
    logic [HBM_ADDR_W-1:0] offset_bytes;
    begin
      offset_bytes = HBM_ADDR_W'(token_id) * HBM_ADDR_W'(EMBED_ROW_BYTES);
      row_addr_from_token = base_addr + offset_bytes;
    end
  endfunction

  assign token_ready_o     = (state_q == EL_IDLE);
  assign req_valid_o       = (state_q == EL_REQ);
  assign req_base_addr_o   = row_addr_from_token(embedding_base_addr_i, token_q.token_id);
  assign req_byte_count_o  = EMBED_ROW_BYTES;
  assign req_tensor_id_o   = TENSOR_EMBED;
  assign req_layer_id_o    = '0;
  assign req_tile_id_o     = token_q.tag.tile_id;
  assign embed_row_ready_o = (state_q == EL_RECV_ROW);
  assign row_valid_o       = (state_q == EL_OUT);
  assign row_fp16_o        = row_fp16_q;
  assign row_meta_o        = token_q;
  assign busy_o            = (state_q != EL_IDLE);

`ifndef SYNTHESIS
  always_comb begin
    if ((state_q == EL_RECV_ROW) && embed_row_valid_i && embed_row_ready_o &&
        embed_row_last_i && (beat_idx_q != EMBED_ROW_BEATS - 1)) begin
      $error("embedding_lookup saw early embed_row_last");
    end
  end
`endif

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q    <= EL_IDLE;
      token_q    <= '0;
      row_fp16_q <= '0;
      beat_idx_q <= '0;
    end else begin
      unique case (state_q)
        EL_IDLE: begin
          if (token_valid_i && token_ready_o) begin
            token_q    <= token_i;
            row_fp16_q <= '0;
            beat_idx_q <= '0;
            state_q    <= EL_REQ;
          end
        end

        EL_REQ: begin
          if (req_valid_o && req_ready_i) begin
            state_q <= EL_RECV_ROW;
          end
        end

        EL_RECV_ROW: begin
          if (embed_row_valid_i && embed_row_ready_o) begin
            row_fp16_q[(beat_idx_q * DMA_BEAT_W) +: DMA_BEAT_W] <= embed_row_i;
            if (embed_row_last_i) begin
              state_q <= EL_OUT;
            end
            beat_idx_q <= beat_idx_q + 1'b1;
          end
        end

        EL_OUT: begin
          if (row_valid_o && row_ready_i) begin
            done_pulse_o <= 1'b1;
            state_q      <= EL_IDLE;
          end
        end

        default: begin
          state_q <= EL_IDLE;
        end
      endcase
    end
  end

endmodule
