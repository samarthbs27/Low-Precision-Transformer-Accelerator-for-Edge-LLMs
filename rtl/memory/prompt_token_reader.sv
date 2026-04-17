import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module prompt_token_reader (
  input  logic                  ap_clk,
  input  logic                  ap_rst_n,
  input  logic                  start_i,
  input  logic [HBM_ADDR_W-1:0] prompt_tokens_base_addr_i,
  input  logic [COUNT_W-1:0]    prompt_token_count_i,
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
  output logic                  token_valid_o,
  input  logic                  token_ready_i,
  output token_bus_t            token_o
);

  typedef enum logic [2:0] {
    PTR_IDLE      = 3'd0,
    PTR_REQ       = 3'd1,
    PTR_WAIT_DATA = 3'd2,
    PTR_STREAM    = 3'd3,
    PTR_DONE      = 3'd4
  } ptr_state_e;

  ptr_state_e             state_q;
  logic [HBM_ADDR_W-1:0]  curr_addr_q;
  logic [COUNT_W-1:0]     tokens_remaining_q;
  logic [COUNT_W-1:0]     emitted_count_q;
  logic [DMA_BEAT_W-1:0]  beat_data_q;
  logic [3:0]             beat_token_count_q;
  logic [3:0]             beat_word_idx_q;
  logic [TOKEN_W-1:0]     curr_token_word;
  logic [3:0]             beat_token_count_now;

  assign busy_o        = (state_q != PTR_IDLE);
  assign error_valid_o = 1'b0;
  assign error_code_o  = ERROR_NONE;

  assign rd_desc_o.region         = REGION_HOST_IO;
  assign rd_desc_o.tensor_id      = TENSOR_NONE;
  assign rd_desc_o.write_not_read = 1'b0;
  assign rd_desc_o.pseudo_channel = HOST_IO_PC_ID;
  assign rd_desc_o.addr           = curr_addr_q;
  assign rd_desc_o.burst_len      = 16'd1;
  assign rd_desc_o.byte_count     = DMA_BEAT_BYTES;
  assign rd_desc_o.layer_id       = '0;
  assign rd_desc_o.kv_head_id     = '0;
  assign rd_desc_o.tile_id        = TILE_ID_W'(emitted_count_q);

  assign beat_token_count_now = (tokens_remaining_q > TOKENS_PER_DMA_BEAT) ?
                                TOKENS_PER_DMA_BEAT[3:0] :
                                tokens_remaining_q[3:0];
  always @* begin
    curr_token_word = '0;
    unique case (beat_word_idx_q)
      4'd0: curr_token_word = beat_data_q[(0*TOKEN_W) +: TOKEN_W];
      4'd1: curr_token_word = beat_data_q[(1*TOKEN_W) +: TOKEN_W];
      4'd2: curr_token_word = beat_data_q[(2*TOKEN_W) +: TOKEN_W];
      4'd3: curr_token_word = beat_data_q[(3*TOKEN_W) +: TOKEN_W];
      4'd4: curr_token_word = beat_data_q[(4*TOKEN_W) +: TOKEN_W];
      4'd5: curr_token_word = beat_data_q[(5*TOKEN_W) +: TOKEN_W];
      4'd6: curr_token_word = beat_data_q[(6*TOKEN_W) +: TOKEN_W];
      4'd7: curr_token_word = beat_data_q[(7*TOKEN_W) +: TOKEN_W];
      default: curr_token_word = '0;
    endcase

    rd_desc_valid_o = (state_q == PTR_REQ);
    rd_data_ready_o = (state_q == PTR_WAIT_DATA);
    token_valid_o   = (state_q == PTR_STREAM);
    token_o         = '0;

    token_o.token_id           = curr_token_word;
    token_o.token_count        = emitted_count_q + 1'b1;
    token_o.tag.layer_id       = '0;
    token_o.tag.block_id       = BLOCK_EMBED;
    token_o.tag.gemm_mode      = GEMM_NONE;
    token_o.tag.tile_id        = TILE_ID_W'(emitted_count_q);
    token_o.tag.token_base     = POS_W'(emitted_count_q);
    token_o.tag.seq_count      = tokens_remaining_q;
    token_o.tag.q_head_id      = '0;
    token_o.tag.kv_head_id     = '0;
    token_o.tag.elem_count     = 16'd1;
    token_o.tag.is_last        = (tokens_remaining_q == 1);
    token_o.tag.is_partial     = (beat_token_count_q != TOKENS_PER_DMA_BEAT);
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q            <= PTR_IDLE;
      curr_addr_q        <= '0;
      tokens_remaining_q <= '0;
      emitted_count_q    <= '0;
      beat_data_q        <= '0;
      beat_token_count_q <= '0;
      beat_word_idx_q    <= '0;
    end else begin
      unique case (state_q)
        PTR_IDLE: begin
          if (start_i) begin
            curr_addr_q        <= prompt_tokens_base_addr_i;
            tokens_remaining_q <= prompt_token_count_i;
            emitted_count_q    <= '0;
            beat_word_idx_q    <= '0;
            beat_token_count_q <= '0;

            if (prompt_token_count_i == '0) begin
              done_pulse_o <= 1'b1;
              state_q      <= PTR_DONE;
            end else begin
              state_q <= PTR_REQ;
            end
          end
        end

        PTR_REQ: begin
          if (rd_desc_valid_o && rd_desc_ready_i) begin
            state_q <= PTR_WAIT_DATA;
          end
        end

        PTR_WAIT_DATA: begin
          if (rd_data_valid_i) begin
            beat_data_q        <= rd_data_i;
            beat_token_count_q <= beat_token_count_now;
            beat_word_idx_q    <= '0;
            state_q            <= PTR_STREAM;
          end
        end

        PTR_STREAM: begin
          if (token_valid_o && token_ready_i) begin
            emitted_count_q    <= emitted_count_q + 1'b1;
            tokens_remaining_q <= tokens_remaining_q - 1'b1;

            if (tokens_remaining_q == 1) begin
              done_pulse_o <= 1'b1;
              state_q      <= PTR_DONE;
            end else if (beat_word_idx_q + 1'b1 >= beat_token_count_q) begin
              curr_addr_q     <= curr_addr_q + DMA_BEAT_BYTES;
              beat_word_idx_q <= '0;
              state_q         <= PTR_REQ;
            end else begin
              beat_word_idx_q <= beat_word_idx_q + 1'b1;
            end
          end
        end

        PTR_DONE: begin
          state_q <= PTR_IDLE;
        end

        default: begin
          state_q <= PTR_IDLE;
        end
      endcase
    end
  end

endmodule
