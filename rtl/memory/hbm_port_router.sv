import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module hbm_port_router (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  host_cmd_rd_desc_valid_i,
  output logic                  host_cmd_rd_desc_ready_o,
  input  dma_desc_t             host_cmd_rd_desc_i,
  output logic                  host_cmd_rd_data_valid_o,
  input  logic                  host_cmd_rd_data_ready_i,
  output logic [DMA_BEAT_W-1:0] host_cmd_rd_data_o,
  input  logic                  prompt_rd_desc_valid_i,
  output logic                  prompt_rd_desc_ready_o,
  input  dma_desc_t             prompt_rd_desc_i,
  output logic                  prompt_rd_data_valid_o,
  input  logic                  prompt_rd_data_ready_i,
  output logic [DMA_BEAT_W-1:0] prompt_rd_data_o,
  input  logic                  weight_rd_desc_valid_i,
  output logic                  weight_rd_desc_ready_o,
  input  dma_desc_t             weight_rd_desc_i,
  output logic                  weight_rd_data_valid_o,
  input  logic                  weight_rd_data_ready_i,
  output logic [DMA_BEAT_W-1:0] weight_rd_data_o,
  input  logic                  embed_lm_rd_desc_valid_i,
  output logic                  embed_lm_rd_desc_ready_o,
  input  dma_desc_t             embed_lm_rd_desc_i,
  output logic                  embed_lm_rd_data_valid_o,
  input  logic                  embed_lm_rd_data_ready_i,
  output logic [DMA_BEAT_W-1:0] embed_lm_rd_data_o,
  input  logic                  kv_rd_desc_valid_i,
  output logic                  kv_rd_desc_ready_o,
  input  dma_desc_t             kv_rd_desc_i,
  output logic                  kv_rd_data_valid_o,
  input  logic                  kv_rd_data_ready_i,
  output logic [DMA_BEAT_W-1:0] kv_rd_data_o,
  input  logic                  host_status_wr_desc_valid_i,
  output logic                  host_status_wr_desc_ready_o,
  input  dma_desc_t             host_status_wr_desc_i,
  input  logic                  host_status_wr_data_valid_i,
  output logic                  host_status_wr_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] host_status_wr_data_i,
  input  logic                  gen_token_wr_desc_valid_i,
  output logic                  gen_token_wr_desc_ready_o,
  input  dma_desc_t             gen_token_wr_desc_i,
  input  logic                  gen_token_wr_data_valid_i,
  output logic                  gen_token_wr_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] gen_token_wr_data_i,
  input  logic                  kv_wr_desc_valid_i,
  output logic                  kv_wr_desc_ready_o,
  input  dma_desc_t             kv_wr_desc_i,
  input  logic                  kv_wr_data_valid_i,
  output logic                  kv_wr_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] kv_wr_data_i,
  input  logic                  debug_wr_desc_valid_i,
  output logic                  debug_wr_desc_ready_o,
  input  dma_desc_t             debug_wr_desc_i,
  input  logic                  debug_wr_data_valid_i,
  output logic                  debug_wr_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] debug_wr_data_i,
  output logic                  shell_rd_desc_valid_o,
  input  logic                  shell_rd_desc_ready_i,
  output dma_desc_t             shell_rd_desc_o,
  input  logic                  shell_rd_data_valid_i,
  output logic                  shell_rd_data_ready_o,
  input  logic [DMA_BEAT_W-1:0] shell_rd_data_i,
  output logic                  shell_wr_desc_valid_o,
  input  logic                  shell_wr_desc_ready_i,
  output dma_desc_t             shell_wr_desc_o,
  output logic                  shell_wr_data_valid_o,
  input  logic                  shell_wr_data_ready_i,
  output logic [DMA_BEAT_W-1:0] shell_wr_data_o
);

  typedef enum logic [2:0] {
    RD_ARB_NONE     = 3'd0,
    RD_ARB_HOST_CMD = 3'd1,
    RD_ARB_PROMPT   = 3'd2,
    RD_ARB_WEIGHT   = 3'd3,
    RD_ARB_EMBED_LM = 3'd4,
    RD_ARB_KV       = 3'd5
  } rd_arb_e;

  typedef enum logic [2:0] {
    WR_ARB_NONE        = 3'd0,
    WR_ARB_HOST_STATUS = 3'd1,
    WR_ARB_GEN_TOKEN   = 3'd2,
    WR_ARB_KV          = 3'd3,
    WR_ARB_DEBUG       = 3'd4
  } wr_arb_e;

  rd_arb_e                rd_sel;
  wr_arb_e                wr_sel;
  rd_arb_e                rd_active_client_q;
  logic                   rd_active_q;
  logic [15:0]            rd_beats_remaining_q;
  dma_desc_t              rd_selected_desc;
  dma_desc_t              wr_selected_desc;
  logic [DMA_BEAT_W-1:0]  wr_selected_data;
  logic                   active_rd_client_ready;

  function automatic logic [15:0] beats_from_byte_count(
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
        beats_from_byte_count = 16'd1;
      end else if (beats_32 > 16'hFFFF) begin
        beats_from_byte_count = 16'hFFFF;
      end else begin
        beats_from_byte_count = beats_32[15:0];
      end
    end
  endfunction

  always_comb begin
    rd_sel           = RD_ARB_NONE;
    rd_selected_desc = '0;

    if (host_cmd_rd_desc_valid_i) begin
      rd_sel           = RD_ARB_HOST_CMD;
      rd_selected_desc = host_cmd_rd_desc_i;
    end else if (prompt_rd_desc_valid_i) begin
      rd_sel           = RD_ARB_PROMPT;
      rd_selected_desc = prompt_rd_desc_i;
    end else if (weight_rd_desc_valid_i) begin
      rd_sel           = RD_ARB_WEIGHT;
      rd_selected_desc = weight_rd_desc_i;
    end else if (embed_lm_rd_desc_valid_i) begin
      rd_sel           = RD_ARB_EMBED_LM;
      rd_selected_desc = embed_lm_rd_desc_i;
    end else if (kv_rd_desc_valid_i) begin
      rd_sel           = RD_ARB_KV;
      rd_selected_desc = kv_rd_desc_i;
    end

    wr_sel           = WR_ARB_NONE;
    wr_selected_desc = '0;
    wr_selected_data = '0;

    if (host_status_wr_desc_valid_i && host_status_wr_data_valid_i) begin
      wr_sel           = WR_ARB_HOST_STATUS;
      wr_selected_desc = host_status_wr_desc_i;
      wr_selected_data = host_status_wr_data_i;
    end else if (gen_token_wr_desc_valid_i && gen_token_wr_data_valid_i) begin
      wr_sel           = WR_ARB_GEN_TOKEN;
      wr_selected_desc = gen_token_wr_desc_i;
      wr_selected_data = gen_token_wr_data_i;
    end else if (kv_wr_desc_valid_i && kv_wr_data_valid_i) begin
      wr_sel           = WR_ARB_KV;
      wr_selected_desc = kv_wr_desc_i;
      wr_selected_data = kv_wr_data_i;
    end else if (debug_wr_desc_valid_i && debug_wr_data_valid_i) begin
      wr_sel           = WR_ARB_DEBUG;
      wr_selected_desc = debug_wr_desc_i;
      wr_selected_data = debug_wr_data_i;
    end
  end

  assign shell_rd_desc_valid_o = !rd_active_q && (rd_sel != RD_ARB_NONE);
  assign shell_rd_desc_o       = rd_selected_desc;

  assign host_cmd_rd_desc_ready_o = !rd_active_q && shell_rd_desc_ready_i && (rd_sel == RD_ARB_HOST_CMD);
  assign prompt_rd_desc_ready_o   = !rd_active_q && shell_rd_desc_ready_i && (rd_sel == RD_ARB_PROMPT);
  assign weight_rd_desc_ready_o   = !rd_active_q && shell_rd_desc_ready_i && (rd_sel == RD_ARB_WEIGHT);
  assign embed_lm_rd_desc_ready_o = !rd_active_q && shell_rd_desc_ready_i && (rd_sel == RD_ARB_EMBED_LM);
  assign kv_rd_desc_ready_o       = !rd_active_q && shell_rd_desc_ready_i && (rd_sel == RD_ARB_KV);

  always_comb begin
    active_rd_client_ready = 1'b0;

    unique case (rd_active_client_q)
      RD_ARB_HOST_CMD: active_rd_client_ready = host_cmd_rd_data_ready_i;
      RD_ARB_PROMPT:   active_rd_client_ready = prompt_rd_data_ready_i;
      RD_ARB_WEIGHT:   active_rd_client_ready = weight_rd_data_ready_i;
      RD_ARB_EMBED_LM: active_rd_client_ready = embed_lm_rd_data_ready_i;
      RD_ARB_KV:       active_rd_client_ready = kv_rd_data_ready_i;
      default:         active_rd_client_ready = 1'b0;
    endcase
  end

  assign shell_rd_data_ready_o   = rd_active_q && active_rd_client_ready;
  assign host_cmd_rd_data_valid_o = rd_active_q && (rd_active_client_q == RD_ARB_HOST_CMD) && shell_rd_data_valid_i;
  assign prompt_rd_data_valid_o   = rd_active_q && (rd_active_client_q == RD_ARB_PROMPT) && shell_rd_data_valid_i;
  assign weight_rd_data_valid_o   = rd_active_q && (rd_active_client_q == RD_ARB_WEIGHT) && shell_rd_data_valid_i;
  assign embed_lm_rd_data_valid_o = rd_active_q && (rd_active_client_q == RD_ARB_EMBED_LM) && shell_rd_data_valid_i;
  assign kv_rd_data_valid_o       = rd_active_q && (rd_active_client_q == RD_ARB_KV) && shell_rd_data_valid_i;
  assign host_cmd_rd_data_o       = shell_rd_data_i;
  assign prompt_rd_data_o         = shell_rd_data_i;
  assign weight_rd_data_o         = shell_rd_data_i;
  assign embed_lm_rd_data_o       = shell_rd_data_i;
  assign kv_rd_data_o             = shell_rd_data_i;

  assign shell_wr_desc_valid_o = (wr_sel != WR_ARB_NONE);
  assign shell_wr_desc_o       = wr_selected_desc;
  assign shell_wr_data_valid_o = (wr_sel != WR_ARB_NONE);
  assign shell_wr_data_o       = wr_selected_data;

  assign host_status_wr_desc_ready_o = (wr_sel == WR_ARB_HOST_STATUS) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign host_status_wr_data_ready_o = (wr_sel == WR_ARB_HOST_STATUS) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign gen_token_wr_desc_ready_o   = (wr_sel == WR_ARB_GEN_TOKEN) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign gen_token_wr_data_ready_o   = (wr_sel == WR_ARB_GEN_TOKEN) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign kv_wr_desc_ready_o          = (wr_sel == WR_ARB_KV) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign kv_wr_data_ready_o          = (wr_sel == WR_ARB_KV) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign debug_wr_desc_ready_o       = (wr_sel == WR_ARB_DEBUG) && shell_wr_desc_ready_i && shell_wr_data_ready_i;
  assign debug_wr_data_ready_o       = (wr_sel == WR_ARB_DEBUG) && shell_wr_desc_ready_i && shell_wr_data_ready_i;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      rd_active_q          <= 1'b0;
      rd_active_client_q   <= RD_ARB_NONE;
      rd_beats_remaining_q <= '0;
    end else begin
      if (!rd_active_q && shell_rd_desc_valid_o && shell_rd_desc_ready_i) begin
        rd_active_q          <= 1'b1;
        rd_active_client_q   <= rd_sel;
        rd_beats_remaining_q <= beats_from_byte_count(rd_selected_desc.byte_count);
      end

      if (rd_active_q && shell_rd_data_valid_i && active_rd_client_ready) begin
        if (rd_beats_remaining_q <= 16'd1) begin
          rd_active_q          <= 1'b0;
          rd_active_client_q   <= RD_ARB_NONE;
          rd_beats_remaining_q <= '0;
        end else begin
          rd_beats_remaining_q <= rd_beats_remaining_q - 1'b1;
        end
      end
    end
  end

endmodule
