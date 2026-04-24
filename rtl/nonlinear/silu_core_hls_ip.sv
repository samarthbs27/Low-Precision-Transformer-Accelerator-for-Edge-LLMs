import tinyllama_pkg::*;

module silu_core_hls_ip (
  input  logic                               ap_clk,
  input  logic                               ap_rst_n,
  input  logic                               start_i,
  input  logic [ELEM_COUNT_W-1:0]            elem_count_i,
  input  logic                               in_valid_i,
  output logic                               in_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] in_chunk_i,
  output logic                               out_valid_o,
  input  logic                               out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_o,
  output logic                               busy_o,
  output logic                               done_pulse_o
);

  localparam int unsigned MAX_CHUNKS = ACT_VECTOR_ELEMS / N_TILE;

  typedef enum logic [1:0] {
    CORE_IDLE  = 2'd0,
    CORE_INPUT = 2'd1,
    CORE_OUT   = 2'd2
  } core_state_e;

  core_state_e state_q;
  logic [ELEM_COUNT_W-1:0] elem_count_q;
  logic [4:0]              chunk_count_q;
  logic [4:0]              in_idx_q;
  logic [4:0]              out_idx_q;
  logic signed [(N_TILE * SCALE_W)-1:0] out_chunks_q [0:MAX_CHUNKS-1];

  function automatic real q16_to_real(
    input logic signed [31:0] value_q16
  );
    begin
      q16_to_real = $itor($signed(value_q16)) / 65536.0;
    end
  endfunction

  function automatic logic signed [31:0] q16_from_real(
    input real value_fp
  );
    real scaled_fp;
    real abs_scaled_fp;
    real frac_fp;
    longint signed floor_mag;
    longint signed rounded_val;
    begin
      scaled_fp = value_fp * 65536.0;
      if (scaled_fp >= 0.0) begin
        floor_mag = $rtoi(scaled_fp);
        frac_fp = scaled_fp - $itor(floor_mag);
        rounded_val = floor_mag;
        if (frac_fp > 0.5000000001) begin
          rounded_val = floor_mag + 1;
        end else if ((frac_fp >= 0.4999999999) &&
                     (frac_fp <= 0.5000000001) &&
                     floor_mag[0]) begin
          rounded_val = floor_mag + 1;
        end
      end else begin
        abs_scaled_fp = -scaled_fp;
        floor_mag = $rtoi(abs_scaled_fp);
        frac_fp = abs_scaled_fp - $itor(floor_mag);
        rounded_val = -floor_mag;
        if (frac_fp > 0.5000000001) begin
          rounded_val = -(floor_mag + 1);
        end else if ((frac_fp >= 0.4999999999) &&
                     (frac_fp <= 0.5000000001) &&
                     floor_mag[0]) begin
          rounded_val = -(floor_mag + 1);
        end
      end

      if (rounded_val > 64'sd2147483647) begin
        q16_from_real = 32'sh7fff_ffff;
      end else if (rounded_val < -64'sd2147483648) begin
        q16_from_real = 32'sh8000_0000;
      end else begin
        q16_from_real = rounded_val[31:0];
      end
    end
  endfunction

  function automatic real fixed_sigmoid(
    input real value_fp
  );
    begin
      fixed_sigmoid = 1.0 / (1.0 + $exp(-value_fp));
    end
  endfunction

  function automatic logic [ELEM_COUNT_W-1:0] effective_elem_count(
    input logic [ELEM_COUNT_W-1:0] elem_count
  );
    begin
      effective_elem_count = (elem_count == '0) ? ELEM_COUNT_W'(ACT_VECTOR_ELEMS) : elem_count;
    end
  endfunction

  assign in_ready_o = (state_q == CORE_INPUT);
  assign out_valid_o = (state_q == CORE_OUT);
  assign busy_o = (state_q != CORE_IDLE);
  assign out_chunk_o = out_chunks_q[out_idx_q];

  always_ff @(posedge ap_clk) begin
    logic signed [31:0] lane_q16;
    logic signed [31:0] silu_q16;
    real lane_fp;

    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q <= CORE_IDLE;
      elem_count_q <= '0;
      chunk_count_q <= '0;
      in_idx_q <= '0;
      out_idx_q <= '0;
      for (int chunk_idx = 0; chunk_idx < MAX_CHUNKS; chunk_idx++) begin
        out_chunks_q[chunk_idx] <= '0;
      end
    end else begin
      unique case (state_q)
        CORE_IDLE: begin
          if (start_i) begin
            elem_count_q <= effective_elem_count(elem_count_i);
            chunk_count_q <= (effective_elem_count(elem_count_i) + N_TILE - 1) / N_TILE;
            in_idx_q <= '0;
            out_idx_q <= '0;
            state_q <= CORE_INPUT;
          end
        end

        CORE_INPUT: begin
          if (in_valid_i && in_ready_o) begin
            for (int elem = 0; elem < N_TILE; elem++) begin
              if (((in_idx_q * N_TILE) + elem) < elem_count_q) begin
                lane_q16 = in_chunk_i[(elem * SCALE_W) +: SCALE_W];
                lane_fp = q16_to_real(lane_q16);
                silu_q16 = q16_from_real(lane_fp * fixed_sigmoid(lane_fp));
                out_chunks_q[in_idx_q][(elem * SCALE_W) +: SCALE_W] <= silu_q16;
              end else begin
                out_chunks_q[in_idx_q][(elem * SCALE_W) +: SCALE_W] <= '0;
              end
            end

            if (in_idx_q == (chunk_count_q - 1'b1)) begin
              out_idx_q <= '0;
              state_q <= CORE_OUT;
            end
            in_idx_q <= in_idx_q + 1'b1;
          end
        end

        CORE_OUT: begin
          if (out_valid_o && out_ready_i) begin
            if (out_idx_q == (chunk_count_q - 1'b1)) begin
              done_pulse_o <= 1'b1;
              state_q <= CORE_IDLE;
            end
            out_idx_q <= out_idx_q + 1'b1;
          end
        end

        default: begin
          state_q <= CORE_IDLE;
        end
      endcase
    end
  end

endmodule
