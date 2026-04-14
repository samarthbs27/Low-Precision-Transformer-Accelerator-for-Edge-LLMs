import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module silu_wrapper (
  input  logic                 ap_clk,
  input  logic                 ap_rst_n,
  input  logic                 gate_valid_i,
  output logic                 gate_ready_o,
  input  act_bus_t             gate_i,
  input  logic [SCALE_W-1:0]   input_scale_i,
  input  logic [SCALE_W-1:0]   output_scale_i,
  output logic                 scale_valid_o,
  input  logic                 scale_ready_i,
  output scale_bus_t           scale_o,
  output logic                 silu_valid_o,
  input  logic                 silu_ready_i,
  output act_bus_t             silu_o,
  output logic                 busy_o,
  output logic                 done_pulse_o
);

  localparam int unsigned CHUNK_ELEMS = N_TILE;
  localparam int unsigned CHUNK_W = CHUNK_ELEMS * SCALE_W;

  typedef enum logic [2:0] {
    SI_IDLE      = 3'd0,
    SI_SEND_CORE = 3'd1,
    SI_RECV_CORE = 3'd2,
    SI_OUT_SCALE = 3'd3,
    SI_OUT_ACT   = 3'd4
  } silu_state_e;

  silu_state_e                state_q;
  act_bus_t                   gate_q;
  act_bus_t                   silu_q;
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] gate_data_q;
  logic signed [(ACT_VECTOR_ELEMS * ACT_W)-1:0] silu_data_q;
  logic [SCALE_W-1:0]         input_scale_q;
  logic [SCALE_W-1:0]         output_scale_q;
  logic [ELEM_COUNT_W-1:0]    elem_count_q;
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
  scale_bus_t                 scale_bus_d;
  wire signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] silu_data_unpack_w;

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    begin
      effective_elem_count = (elem_count == '0) ? ELEM_COUNT_W'(ACT_VECTOR_ELEMS) : elem_count;
    end
  endfunction

  function automatic logic signed [31:0] dequantize_act_lane(
    input logic signed [ACT_W-1:0] act_val,
    input logic [SCALE_W-1:0]      scale_val
  );
    longint signed product;
    begin
      product = $signed(act_val) * $signed({1'b0, scale_val});
      if (product > 64'sd2147483647) begin
        dequantize_act_lane = 32'sh7fff_ffff;
      end else if (product < -64'sd2147483648) begin
        dequantize_act_lane = 32'sh8000_0000;
      end else begin
        dequantize_act_lane = product[31:0];
      end
    end
  endfunction

  function automatic logic signed [ACT_W-1:0] quantize_fixed_lane(
    input logic signed [31:0] value_q16,
    input logic [SCALE_W-1:0] scale_q16
  );
    longint signed numerator_abs;
    longint unsigned denominator;
    longint unsigned quotient_mag;
    longint unsigned rounded_mag;
    longint unsigned remainder_mag;
    longint signed rounded_signed;
    begin
      denominator = (scale_q16 == '0) ? 1 : scale_q16;
      if (value_q16 < 0) begin
        numerator_abs = -value_q16;
      end else begin
        numerator_abs = value_q16;
      end
      quotient_mag = numerator_abs / denominator;
      remainder_mag = numerator_abs % denominator;
      rounded_mag = quotient_mag;

      if ((remainder_mag << 1) > denominator) begin
        rounded_mag = quotient_mag + 1;
      end else if (((remainder_mag << 1) == denominator) && quotient_mag[0]) begin
        rounded_mag = quotient_mag + 1;
      end

      rounded_signed = (value_q16 < 0) ? -rounded_mag : rounded_mag;
      if (rounded_signed > 127) begin
        quantize_fixed_lane = 8'sd127;
      end else if (rounded_signed < -127) begin
        quantize_fixed_lane = -8'sd127;
      end else begin
        quantize_fixed_lane = ACT_W'(rounded_signed);
      end
    end
  endfunction

  assign gate_ready_o      = (state_q == SI_IDLE);
  assign scale_valid_o     = (state_q == SI_OUT_SCALE);
  assign silu_valid_o      = (state_q == SI_OUT_ACT);
  assign silu_o            = silu_q;
  assign scale_o           = scale_bus_d;
  assign busy_o            = (state_q != SI_IDLE);
  assign core_in_valid_w   = (state_q == SI_SEND_CORE);
  assign core_out_ready_w  = (state_q == SI_RECV_CORE);

  always_comb begin
    silu_q = '0;
    silu_q.tag = gate_q.tag;
    silu_q.tag.block_id = BLOCK_SILU;
    silu_q.tag.gemm_mode = GEMM_GATE;
    silu_q.data = silu_data_unpack_w;

    scale_bus_d = '0;
    scale_bus_d.tag = gate_q.tag;
    scale_bus_d.tag.block_id = BLOCK_SILU;
    scale_bus_d.tag.gemm_mode = GEMM_GATE;
    scale_bus_d.tag.elem_count = ELEM_COUNT_W'(SCALE_VECTOR_ELEMS);
    scale_bus_d.tag.is_partial = 1'b0;
    scale_bus_d.tag.is_last = 1'b1;
    scale_bus_d.data = {SCALE_VECTOR_ELEMS{output_scale_q}};
  end

  always_comb begin
    core_in_chunk_w = '0;
    if (state_q == SI_SEND_CORE) begin
      for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
        core_in_chunk_w[(elem * SCALE_W) +: SCALE_W] =
          dequantize_act_lane(
            gate_data_q[(((send_chunk_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W],
            input_scale_q
          );
      end
    end
  end

  silu_core_hls_ip u_silu_core_hls_ip (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .start_i(core_start_q),
    .elem_count_i(elem_count_q),
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
      state_q          <= SI_IDLE;
      gate_q           <= '0;
      gate_data_q      <= '0;
      silu_data_q      <= '0;
      input_scale_q    <= '0;
      output_scale_q   <= '0;
      elem_count_q     <= '0;
      send_chunk_idx_q <= '0;
      recv_chunk_idx_q <= '0;
    end else begin
      unique case (state_q)
        SI_IDLE: begin
          if (gate_valid_i && gate_ready_o) begin
            gate_q           <= gate_i;
            gate_data_q      <= gate_i.data;
            silu_data_q      <= '0;
            input_scale_q    <= input_scale_i;
            output_scale_q   <= output_scale_i;
            elem_count_q     <= effective_elem_count(gate_i.tag.elem_count);
            send_chunk_idx_q <= '0;
            recv_chunk_idx_q <= '0;
            core_start_q     <= 1'b1;
            state_q          <= SI_SEND_CORE;
          end
        end

        SI_SEND_CORE: begin
          if (core_in_valid_w && core_in_ready_w) begin
            if (send_chunk_idx_q == (((elem_count_q + CHUNK_ELEMS - 1) / CHUNK_ELEMS) - 1)) begin
              state_q <= SI_RECV_CORE;
            end
            send_chunk_idx_q <= send_chunk_idx_q + 1'b1;
          end
        end

        SI_RECV_CORE: begin
          if (core_out_valid_w && core_out_ready_w) begin
            for (int elem = 0; elem < CHUNK_ELEMS; elem++) begin
              if (((recv_chunk_idx_q * CHUNK_ELEMS) + elem) < elem_count_q) begin
                silu_data_q[(((recv_chunk_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W] <=
                  quantize_fixed_lane(core_out_chunk_w[(elem * SCALE_W) +: SCALE_W], output_scale_q);
              end else begin
                silu_data_q[(((recv_chunk_idx_q * CHUNK_ELEMS) + elem) * ACT_W) +: ACT_W] <= '0;
              end
            end

            if (recv_chunk_idx_q == (((elem_count_q + CHUNK_ELEMS - 1) / CHUNK_ELEMS) - 1)) begin
              state_q <= SI_OUT_SCALE;
            end
            recv_chunk_idx_q <= recv_chunk_idx_q + 1'b1;
          end
        end

        SI_OUT_SCALE: begin
          if (scale_valid_o && scale_ready_i) begin
            state_q <= SI_OUT_ACT;
          end
        end

        SI_OUT_ACT: begin
          if (silu_valid_o && silu_ready_i) begin
            done_pulse_o <= 1'b1;
            state_q      <= SI_IDLE;
          end
        end

        default: begin
          state_q <= SI_IDLE;
        end
      endcase
    end
  end

  generate
    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_silu_unpack
      assign silu_data_unpack_w[lane] = silu_data_q[(lane * ACT_W) +: ACT_W];
    end
  endgenerate

endmodule
