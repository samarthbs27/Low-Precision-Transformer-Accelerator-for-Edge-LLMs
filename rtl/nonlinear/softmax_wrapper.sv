import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module softmax_wrapper (
  input  logic                 ap_clk,
  input  logic                 ap_rst_n,
  input  logic                 score_valid_i,
  output logic                 score_ready_o,
  input  acc_bus_t             score_i,
  input  logic [SCALE_W-1:0]   score_scale_i,
  output logic                 prob_scale_valid_o,
  input  logic                 prob_scale_ready_i,
  output scale_bus_t           prob_scale_o,
  output logic                 prob_valid_o,
  input  logic                 prob_ready_i,
  output act_bus_t             prob_o,
  output logic                 busy_o,
  output logic                 done_pulse_o
);

  localparam int unsigned CHUNK_ELEMS = N_TILE;
  localparam int unsigned CHUNK_W = CHUNK_ELEMS * SCALE_W;
  localparam logic [SCALE_W-1:0] PROB_SCALE_Q16 = SCALE_W'(32'd516);

  typedef enum logic [2:0] {
    SW_IDLE      = 3'd0,
    SW_SEND_CORE = 3'd1,
    SW_RECV_CORE = 3'd2,
    SW_OUT_SCALE = 3'd3,
    SW_OUT_ACT   = 3'd4
  } softmax_state_e;

  softmax_state_e             state_q;
  acc_bus_t                   score_q;
  act_bus_t                   prob_q;
  logic signed [(ACC_VECTOR_ELEMS * ACC_W)-1:0] score_data_q;
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] prob_data_q;
  logic [SCALE_W-1:0]         score_scale_q;
  logic [COUNT_W-1:0]         row_count_q;
  logic [3:0]                 send_chunk_idx_q;
  logic [3:0]                 recv_chunk_idx_q;
  logic                       core_start_q;
  logic                       core_busy_w;
  logic                       core_done_w;
  logic                       core_in_valid_w;
  logic                       core_in_ready_w;
  logic signed [CHUNK_W-1:0]  core_in_chunk_w;
  logic                       core_out_valid_w;
  logic                       core_out_ready_w;
  logic signed [CHUNK_W-1:0]  core_out_chunk_w;
  scale_bus_t                 prob_scale_bus_d;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] prob_data_unpack_w;

  function automatic logic [COUNT_W-1:0] row_count_from_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    logic [ELEM_COUNT_W-1:0] effective_count;
    begin
      effective_count = (elem_count == '0) ? ELEM_COUNT_W'(SCORE_CHUNK_ELEMS) : elem_count;
      row_count_from_elem_count = COUNT_W'((effective_count + SCORE_K_TILE - 1) / SCORE_K_TILE);
    end
  endfunction

  function automatic logic signed [31:0] dequantize_score_lane(
    input logic signed [ACC_W-1:0] score_val,
    input logic [SCALE_W-1:0]      score_scale
  );
    longint signed product;
    begin
      product = $signed(score_val) * $signed({1'b0, score_scale});
      if (product > 64'sd2147483647) begin
        dequantize_score_lane = 32'sh7fff_ffff;
      end else if (product < -64'sd2147483648) begin
        dequantize_score_lane = 32'sh8000_0000;
      end else begin
        dequantize_score_lane = product[31:0];
      end
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] quantize_probability_lane(
    input logic signed [31:0] prob_q16
  );
    longint signed product;
    longint signed rounded;
    logic [15:0] remainder_bits;
    begin
      product = prob_q16 * 32'sd127;
      rounded = product >>> 16;
      remainder_bits = product[15:0];
      if (remainder_bits > 16'h8000) begin
        rounded = rounded + 1;
      end else if ((remainder_bits == 16'h8000) && rounded[0]) begin
        rounded = rounded + 1;
      end

      if (rounded < 0) begin
        quantize_probability_lane = '0;
      end else if (rounded > 127) begin
        quantize_probability_lane = 8'sd127;
      end else begin
        quantize_probability_lane = ACT_W'(rounded);
      end
    end
  endfunction

  assign score_ready_o       = (state_q == SW_IDLE);
  assign prob_scale_valid_o  = (state_q == SW_OUT_SCALE);
  assign prob_valid_o        = (state_q == SW_OUT_ACT);
  assign prob_o              = prob_q;
  assign prob_scale_o        = prob_scale_bus_d;
  assign busy_o              = (state_q != SW_IDLE);
  assign core_in_valid_w     = (state_q == SW_SEND_CORE);
  assign core_out_ready_w    = (state_q == SW_RECV_CORE);

  always_comb begin
    prob_scale_bus_d = '0;
    prob_q = '0;
    prob_q.tag = score_q.tag;
    prob_q.tag.block_id = BLOCK_SOFTMAX;
    prob_q.tag.gemm_mode = GEMM_WEIGHTED_SUM;
    prob_q.data = prob_data_unpack_w;

    prob_scale_bus_d = '0;
    prob_scale_bus_d.tag = score_q.tag;
    prob_scale_bus_d.tag.block_id = BLOCK_SOFTMAX;
    prob_scale_bus_d.tag.gemm_mode = GEMM_WEIGHTED_SUM;
    prob_scale_bus_d.tag.elem_count = ELEM_COUNT_W'(SCALE_VECTOR_ELEMS);
    prob_scale_bus_d.tag.is_partial = 1'b0;
    prob_scale_bus_d.tag.is_last = 1'b1;
    prob_scale_bus_d.data = {SCALE_VECTOR_ELEMS{PROB_SCALE_Q16}};
  end

  always_comb begin
    core_in_chunk_w = '0;
    if (state_q == SW_SEND_CORE) begin
      for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
        core_in_chunk_w[(elem * SCALE_W) +: SCALE_W] =
          dequantize_score_lane(
            score_data_q[(((send_chunk_idx_q * CHUNK_ELEMS) + elem) * ACC_W) +: ACC_W],
            score_scale_q
          );
      end
    end
  end

  softmax_core_hls_ip u_softmax_core_hls_ip (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .start_i(core_start_q),
    .row_count_i(row_count_q),
    .key_col_count_i(COUNT_W'(SCORE_K_TILE)),
    .in_valid_i(core_in_valid_w),
    .in_ready_o(core_in_ready_w),
    .in_chunk_i(core_in_chunk_w),
    .out_valid_o(core_out_valid_w),
    .out_ready_i(core_out_ready_w),
    .out_chunk_o(core_out_chunk_w),
    .busy_o(core_busy_w),
    .done_pulse_o(core_done_w)
  );

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;
    core_start_q <= 1'b0;

    if (!ap_rst_n) begin
      state_q          <= SW_IDLE;
      score_q          <= '0;
      score_data_q     <= '0;
      prob_data_q      <= '0;
      score_scale_q    <= '0;
      row_count_q      <= '0;
      send_chunk_idx_q <= '0;
      recv_chunk_idx_q <= '0;
    end else begin
      unique case (state_q)
        SW_IDLE: begin
          if (score_valid_i && score_ready_o) begin
            score_q          <= score_i;
            score_data_q     <= score_i.data;
            score_scale_q    <= score_scale_i;
            row_count_q      <= row_count_from_elem_count(score_i.tag.elem_count);
            send_chunk_idx_q <= '0;
            recv_chunk_idx_q <= '0;
            prob_data_q      <= '0;
            core_start_q     <= 1'b1;
            state_q          <= SW_SEND_CORE;
          end
        end

        SW_SEND_CORE: begin
          if (core_in_valid_w && core_in_ready_w) begin
            if (send_chunk_idx_q == ((SCORE_CHUNK_ELEMS / CHUNK_ELEMS) - 1)) begin
              state_q <= SW_RECV_CORE;
            end
            send_chunk_idx_q <= send_chunk_idx_q + 1'b1;
          end
        end

        SW_RECV_CORE: begin
          if (core_out_valid_w && core_out_ready_w) begin
            for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
              prob_data_q[(((recv_chunk_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W] <=
                quantize_probability_lane(core_out_chunk_w[(elem * SCALE_W) +: SCALE_W]);
            end

            if (recv_chunk_idx_q == ((SCORE_CHUNK_ELEMS / CHUNK_ELEMS) - 1)) begin
              state_q <= SW_OUT_SCALE;
            end
            recv_chunk_idx_q <= recv_chunk_idx_q + 1'b1;
          end
        end

        SW_OUT_SCALE: begin
          if (prob_scale_valid_o && prob_scale_ready_i) begin
            state_q <= SW_OUT_ACT;
          end
        end

        SW_OUT_ACT: begin
          if (prob_valid_o && prob_ready_i) begin
            done_pulse_o <= 1'b1;
            state_q      <= SW_IDLE;
          end
        end

        default: begin
          state_q <= SW_IDLE;
        end
      endcase
    end
  end

  generate
    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_prob_unpack
      assign prob_data_unpack_w[lane] = prob_data_q[(lane * ACT_W) +: ACT_W];
    end
  endgenerate

endmodule
