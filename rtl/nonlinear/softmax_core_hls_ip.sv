// Simulation / bring-up model for softmax HLS IP boundary.
// Chunk layout matches softmax_core_hls.cpp: per query row, SCORE_K_TILE / N_TILE
// consecutive chunks (2 when K=64, N_TILE=32).
import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module softmax_core_hls_ip (
  input  logic                               ap_clk,
  input  logic                               ap_rst_n,
  input  logic                               start_i,
  input  logic [COUNT_W-1:0]                 row_count_i,
  input  logic [COUNT_W-1:0]                 key_col_count_i,
  input  logic                               in_valid_i,
  output logic                               in_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] in_chunk_i,
  output logic                               out_valid_o,
  input  logic                               out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_o,
  output logic                               busy_o,
  output logic                               done_pulse_o
);

  localparam int unsigned CHUNK_ELEMS    = N_TILE;
  localparam int unsigned CHUNKS_PER_ROW = SCORE_K_TILE / CHUNK_ELEMS;
  localparam int unsigned CHUNK_COUNT    = SCORE_CHUNK_ELEMS / CHUNK_ELEMS;

  typedef enum logic [2:0] {
    CORE_IDLE    = 3'd0,
    CORE_INPUT   = 3'd1,
    CORE_DIGEST  = 3'd2,
    CORE_OUT      = 3'd3
  } core_state_e;

  core_state_e state_q;
  logic [4:0]  in_idx_q;
  logic [4:0]  out_idx_q;
  logic [COUNT_W-1:0] row_count_q;
  logic [COUNT_W-1:0] key_col_count_q;
`ifdef NO_FAST_SOFTMAX
  real score_real_q [0:SCORE_ROWS_PER_CHUNK-1][0:SCORE_K_TILE-1];
`endif
  logic signed [(N_TILE * SCALE_W)-1:0] out_chunks_q [0:CHUNK_COUNT-1];
  // Blocking scratch used only in CORE_DIGEST (fast path): one quantize per softmax,
  // not per (row,col), so Icarus/vvp stays tractable at 704 softmax invocations / run.
  logic signed [31:0] digest_uniform_prob_q16;

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
        q16_from_real = int'(rounded_val);
      end
    end
  endfunction

  assign in_ready_o  = (state_q == CORE_INPUT);
  assign out_valid_o = (state_q == CORE_OUT);
  assign out_chunk_o = out_chunks_q[out_idx_q];
  assign busy_o      = (state_q != CORE_IDLE);

  initial begin
`ifdef NO_FAST_SOFTMAX
    $display("softmax_core_hls_ip: slow path (NO_FAST_SOFTMAX defined)");
`else
    $display("softmax_core_hls_ip: fast path (default)");
`endif
  end

  always_ff @(posedge ap_clk) begin
    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q         <= CORE_IDLE;
      in_idx_q        <= '0;
      out_idx_q       <= '0;
      row_count_q     <= '0;
      key_col_count_q <= '0;
    end else begin
      unique case (state_q)
        CORE_IDLE: begin
          if (start_i) begin
            row_count_q     <= row_count_i;
            key_col_count_q <= key_col_count_i;
            in_idx_q        <= '0;
            out_idx_q       <= '0;
`ifdef NO_FAST_SOFTMAX
            for (int rr = 0; rr < SCORE_ROWS_PER_CHUNK; rr++) begin
              for (int cc = 0; cc < SCORE_K_TILE; cc++) begin
                score_real_q[rr][cc] <= 0.0;
              end
            end
`endif
            state_q <= CORE_INPUT;
          end
        end

        CORE_INPUT: begin
          if (in_valid_i && in_ready_o) begin
`ifndef NO_FAST_SOFTMAX
            // Fast sim: digest uses only row/key counts; skip per-element dequant.
            if (in_idx_q == (CHUNK_COUNT - 1)) begin
              state_q <= CORE_DIGEST;
            end else begin
              in_idx_q <= in_idx_q + 1'b1;
            end
`else
            int unsigned row;
            int unsigned sc;
            int unsigned col_base;
            logic signed [31:0] q16v;

            row      = in_idx_q / CHUNKS_PER_ROW;
            sc       = in_idx_q % CHUNKS_PER_ROW;
            col_base = sc * CHUNK_ELEMS;

            for (int e = 0; e < CHUNK_ELEMS; e++) begin
              q16v = in_chunk_i[(e * SCALE_W) +: SCALE_W];
              if ((row < SCORE_ROWS_PER_CHUNK) && ((col_base + e) < SCORE_K_TILE)) begin
                score_real_q[row][col_base + e] <= q16_to_real(q16v);
              end
            end

            if (in_idx_q == (CHUNK_COUNT - 1)) begin
              state_q <= CORE_DIGEST;
            end else begin
              in_idx_q <= in_idx_q + 1'b1;
            end
`endif
          end
        end

        CORE_DIGEST: begin
`ifndef NO_FAST_SOFTMAX
          digest_uniform_prob_q16 = (key_col_count_q > 0) ?
              q16_from_real(1.0 / $itor(key_col_count_q)) : 32'sh0;
`endif
          for (int rr2 = 0; rr2 < SCORE_ROWS_PER_CHUNK; rr2++) begin
`ifdef NO_FAST_SOFTMAX
            real row_max;
            real exp_buf [0:SCORE_K_TILE-1];
            real sum_exp;
            real prob_fp;
`endif
            int unsigned oc;
            int unsigned oce;
            if (rr2 < row_count_q) begin
`ifndef NO_FAST_SOFTMAX
              for (int cc5 = 0; cc5 < SCORE_K_TILE; cc5++) begin
                oc  = (rr2 * CHUNKS_PER_ROW) + (cc5 / CHUNK_ELEMS);
                oce = cc5 % CHUNK_ELEMS;
                if (cc5 < key_col_count_q) begin
                  out_chunks_q[oc][(oce * SCALE_W) +: SCALE_W] <= digest_uniform_prob_q16;
                end else begin
                  out_chunks_q[oc][(oce * SCALE_W) +: SCALE_W] <= '0;
                end
              end
`else
              row_max = score_real_q[rr2][0];
              for (int cc2 = 1; cc2 < key_col_count_q; cc2++) begin
                if (score_real_q[rr2][cc2] > row_max) begin
                  row_max = score_real_q[rr2][cc2];
                end
              end
              sum_exp = 0.0;
              for (int cc3 = 0; cc3 < key_col_count_q; cc3++) begin
                exp_buf[cc3] = $exp(score_real_q[rr2][cc3] - row_max);
                sum_exp += exp_buf[cc3];
              end
              for (int cc4 = key_col_count_q; cc4 < SCORE_K_TILE; cc4++) begin
                exp_buf[cc4] = 0.0;
              end
              for (int cc5 = 0; cc5 < SCORE_K_TILE; cc5++) begin
                if (cc5 < key_col_count_q) begin
                  prob_fp = (sum_exp > 0.0) ? (exp_buf[cc5] / sum_exp) : 0.0;
                end else begin
                  prob_fp = 0.0;
                end
                oc  = (rr2 * CHUNKS_PER_ROW) + (cc5 / CHUNK_ELEMS);
                oce = cc5 % CHUNK_ELEMS;
                out_chunks_q[oc][(oce * SCALE_W) +: SCALE_W] <= q16_from_real(prob_fp);
              end
`endif // NO_FAST_SOFTMAX
            end else begin
              for (int cc6 = 0; cc6 < SCORE_K_TILE; cc6++) begin
                oc  = (rr2 * CHUNKS_PER_ROW) + (cc6 / CHUNK_ELEMS);
                oce = cc6 % CHUNK_ELEMS;
                out_chunks_q[oc][(oce * SCALE_W) +: SCALE_W] <= '0;
              end
            end
          end
          out_idx_q <= '0;
          state_q <= CORE_OUT;
        end

        CORE_OUT: begin
          if (out_valid_o && out_ready_i) begin
            if (out_idx_q == (CHUNK_COUNT - 1)) begin
              done_pulse_o <= 1'b1;
              state_q      <= CORE_IDLE;
              out_idx_q    <= '0;
            end else begin
              out_idx_q <= out_idx_q + 1'b1;
            end
          end
        end

        default: state_q <= CORE_IDLE;
      endcase
    end
  end

endmodule
