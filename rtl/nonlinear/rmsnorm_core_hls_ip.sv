import tinyllama_pkg::*;

module rmsnorm_core_hls_ip (
  input  logic                               ap_clk,
  input  logic                               ap_rst_n,
  input  logic                               start_i,
  input  logic [COUNT_W-1:0]                 row_count_i,
  input  logic [15:0]                        feature_count_i,
  input  logic [31:0]                        epsilon_q16_i,
  input  logic                               act_valid_i,
  output logic                               act_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] act_chunk_i,
  input  logic                               gamma_valid_i,
  output logic                               gamma_ready_o,
  input  logic signed [(N_TILE * SCALE_W)-1:0] gamma_chunk_i,
  output logic                               out_valid_o,
  input  logic                               out_ready_i,
  output logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_o,
  output logic                               busy_o,
  output logic                               done_pulse_o
);

  localparam int unsigned FEATURE_CHUNKS = D_MODEL / N_TILE;

  typedef enum logic [2:0] {
    CORE_IDLE    = 3'd0,
    CORE_GAMMA   = 3'd1,
    CORE_ACT     = 3'd2,
    CORE_COMPUTE = 3'd3,
    CORE_OUT     = 3'd4
  } core_state_e;

  core_state_e state_q;

  logic [COUNT_W-1:0] row_count_q;
  logic [15:0]        feature_count_q;
  logic [31:0]        epsilon_q16_q;
  logic [6:0]         gamma_idx_q;
  logic [6:0]         feature_idx_q;
  logic [COUNT_W-1:0] row_idx_q;

  logic signed [SCALE_W-1:0] gamma_chunks_q [0:FEATURE_CHUNKS-1][0:N_TILE-1];
  logic signed [SCALE_W-1:0] act_chunks_q   [0:FEATURE_CHUNKS-1][0:M_TILE-1][0:N_TILE-1];
  logic signed [SCALE_W-1:0] out_chunks_q   [0:FEATURE_CHUNKS-1][0:M_TILE-1][0:N_TILE-1];
  logic signed [(N_TILE * SCALE_W)-1:0] out_chunk_d;

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

  assign gamma_ready_o = (state_q == CORE_GAMMA);
  assign act_ready_o   = (state_q == CORE_ACT);
  assign out_valid_o   = (state_q == CORE_OUT);
  assign out_chunk_o   = out_chunk_d;
  assign busy_o        = (state_q != CORE_IDLE);

  always_comb begin
    out_chunk_d = '0;
    if ((feature_idx_q < FEATURE_CHUNKS) && (row_idx_q < M_TILE)) begin
      for (int elem = 0; elem < N_TILE; elem++) begin
        out_chunk_d[(elem * SCALE_W) +: SCALE_W] =
          out_chunks_q[feature_idx_q][row_idx_q][elem];
      end
    end
  end

  always_ff @(posedge ap_clk) begin
    real epsilon_fp;
    real sumsq_fp [0:M_TILE-1];
    real inv_rms_fp [0:M_TILE-1];
    real sample_fp;
    real gamma_fp;
    real mean_sq_fp;

    done_pulse_o <= 1'b0;

    if (!ap_rst_n) begin
      state_q         <= CORE_IDLE;
      row_count_q     <= '0;
      feature_count_q <= '0;
      epsilon_q16_q   <= '0;
      gamma_idx_q     <= '0;
      feature_idx_q   <= '0;
      row_idx_q       <= '0;
      for (int feature_idx = 0; feature_idx < FEATURE_CHUNKS; feature_idx++) begin
        for (int lane = 0; lane < N_TILE; lane++) begin
          gamma_chunks_q[feature_idx][lane] <= '0;
        end
        for (int row = 0; row < M_TILE; row++) begin
          for (int lane = 0; lane < N_TILE; lane++) begin
            act_chunks_q[feature_idx][row][lane] <= '0;
            out_chunks_q[feature_idx][row][lane] <= '0;
          end
        end
      end
    end else begin
      unique case (state_q)
        CORE_IDLE: begin
          if (start_i) begin
            row_count_q     <= (row_count_i == '0) ? COUNT_W'(1) : row_count_i;
            feature_count_q <= (feature_count_i == '0) ? 16'(D_MODEL) : feature_count_i;
            epsilon_q16_q   <= epsilon_q16_i;
            gamma_idx_q     <= '0;
            feature_idx_q   <= '0;
            row_idx_q       <= '0;
            state_q         <= CORE_GAMMA;
          end
        end

        CORE_GAMMA: begin
          if (gamma_valid_i && gamma_ready_o) begin
            for (int lane = 0; lane < N_TILE; lane++) begin
              gamma_chunks_q[gamma_idx_q][lane] <=
                gamma_chunk_i[(lane * SCALE_W) +: SCALE_W];
            end
            if (gamma_idx_q == (FEATURE_CHUNKS - 1)) begin
              feature_idx_q <= '0;
              row_idx_q <= '0;
              state_q <= CORE_ACT;
            end else begin
              gamma_idx_q <= gamma_idx_q + 1'b1;
            end
          end
        end

        CORE_ACT: begin
          if (act_valid_i && act_ready_o) begin
            for (int lane = 0; lane < N_TILE; lane++) begin
              act_chunks_q[feature_idx_q][row_idx_q][lane] <=
                act_chunk_i[(lane * SCALE_W) +: SCALE_W];
            end

            if (row_idx_q == (row_count_q - 1'b1)) begin
              row_idx_q <= '0;
              if (feature_idx_q == (FEATURE_CHUNKS - 1)) begin
                state_q <= CORE_COMPUTE;
              end else begin
                feature_idx_q <= feature_idx_q + 1'b1;
              end
            end else begin
              row_idx_q <= row_idx_q + 1'b1;
            end
          end
        end

        CORE_COMPUTE: begin
          epsilon_fp = q16_to_real($signed(epsilon_q16_q));
          for (int row = 0; row < M_TILE; row++) begin
            sumsq_fp[row] = 0.0;
            inv_rms_fp[row] = 0.0;
          end

          for (int feature_idx = 0; feature_idx < FEATURE_CHUNKS; feature_idx++) begin
            for (int row = 0; row < row_count_q; row++) begin
              for (int lane = 0; lane < N_TILE; lane++) begin
                sample_fp = q16_to_real(act_chunks_q[feature_idx][row][lane]);
                sumsq_fp[row] = sumsq_fp[row] + (sample_fp * sample_fp);
              end
            end
          end

          for (int row = 0; row < row_count_q; row++) begin
            mean_sq_fp = sumsq_fp[row] / $itor(feature_count_q);
            inv_rms_fp[row] = 1.0 / $sqrt(mean_sq_fp + epsilon_fp);
          end

          for (int feature_idx = 0; feature_idx < FEATURE_CHUNKS; feature_idx++) begin
            for (int row = 0; row < row_count_q; row++) begin
              for (int lane = 0; lane < N_TILE; lane++) begin
                sample_fp = q16_to_real(act_chunks_q[feature_idx][row][lane]);
                gamma_fp = q16_to_real(gamma_chunks_q[feature_idx][lane]);
                out_chunks_q[feature_idx][row][lane] <=
                  q16_from_real(sample_fp * inv_rms_fp[row] * gamma_fp);
              end
            end
            for (int row = row_count_q; row < M_TILE; row++) begin
              for (int lane = 0; lane < N_TILE; lane++) begin
                out_chunks_q[feature_idx][row][lane] <= '0;
              end
            end
          end

          feature_idx_q <= '0;
          row_idx_q <= '0;
          state_q <= CORE_OUT;
        end

        CORE_OUT: begin
          if (out_valid_o && out_ready_i) begin
            if (row_idx_q == (row_count_q - 1'b1)) begin
              row_idx_q <= '0;
              if (feature_idx_q == (FEATURE_CHUNKS - 1)) begin
                done_pulse_o <= 1'b1;
                state_q <= CORE_IDLE;
              end else begin
                feature_idx_q <= feature_idx_q + 1'b1;
              end
            end else begin
              row_idx_q <= row_idx_q + 1'b1;
            end
          end
        end

        default: begin
          state_q <= CORE_IDLE;
        end
      endcase
    end
  end

endmodule
