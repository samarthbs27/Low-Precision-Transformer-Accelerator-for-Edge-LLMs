import tinyllama_pkg::*;

module rope_lut_rom #(
  parameter string COS_MEMH = "rtl/compute/rope_cos_rom.memh",
  parameter string SIN_MEMH = "rtl/compute/rope_sin_rom.memh"
) (
  input  logic [POS_W-1:0] token_base_i,
  input  logic [COUNT_W-1:0] token_count_i,
  output logic signed [ACT_VECTOR_ELEMS-1:0][SCALE_W-1:0] cos_o,
  output logic signed [ACT_VECTOR_ELEMS-1:0][SCALE_W-1:0] sin_o
);

  localparam int unsigned TABLE_HEAD_DIM = HEAD_DIM / 2;
  localparam int unsigned ROM_DEPTH = MAX_POS * TABLE_HEAD_DIM;
  localparam logic signed [SCALE_W-1:0] COS_IDENTITY = 32'sh0001_0000;
  localparam logic signed [SCALE_W-1:0] SIN_ZERO     = '0;

  logic signed [SCALE_W-1:0] cos_rom [0:ROM_DEPTH-1];
  logic signed [SCALE_W-1:0] sin_rom [0:ROM_DEPTH-1];
  wire [ROPE_CHUNK_TOKENS-1:0] token_active_w;
  wire [ROPE_CHUNK_TOKENS-1:0][POS_W-1:0] position_idx_w;
  wire [ROPE_CHUNK_TOKENS-1:0] position_in_range_w;
  wire signed [ROPE_CHUNK_TOKENS-1:0][TABLE_HEAD_DIM-1:0][SCALE_W-1:0] cos_table_w;
  wire signed [ROPE_CHUNK_TOKENS-1:0][TABLE_HEAD_DIM-1:0][SCALE_W-1:0] sin_table_w;

  initial begin
    $readmemh(COS_MEMH, cos_rom);
    $readmemh(SIN_MEMH, sin_rom);
  end

  generate
    for (genvar token = 0; token < ROPE_CHUNK_TOKENS; token++) begin : g_rope_token
      assign token_active_w[token] = (token < token_count_i);
      assign position_idx_w[token] = token_base_i + POS_W'(token);
      assign position_in_range_w[token] = (position_idx_w[token] < MAX_POS);

      for (genvar table_dim = 0; table_dim < TABLE_HEAD_DIM; table_dim++) begin : g_rope_dim
        wire [31:0] rom_index_w;

        assign rom_index_w = (position_idx_w[token] * TABLE_HEAD_DIM) + table_dim;
        assign cos_table_w[token][table_dim] =
          (token_active_w[token] && position_in_range_w[token]) ?
            cos_rom[rom_index_w] :
            COS_IDENTITY;
        assign sin_table_w[token][table_dim] =
          (token_active_w[token] && position_in_range_w[token]) ?
            sin_rom[rom_index_w] :
            SIN_ZERO;
      end
    end

    for (genvar lane = 0; lane < ACT_VECTOR_ELEMS; lane++) begin : g_rope_rom_lane
      localparam int unsigned TOKEN_LOCAL = lane / HEAD_DIM;
      localparam int unsigned DIM_LOCAL   = lane % HEAD_DIM;
      localparam int unsigned TABLE_DIM   =
        (DIM_LOCAL < TABLE_HEAD_DIM) ? DIM_LOCAL : (DIM_LOCAL - TABLE_HEAD_DIM);

      assign cos_o[lane] = cos_table_w[TOKEN_LOCAL][TABLE_DIM];
      assign sin_o[lane] = sin_table_w[TOKEN_LOCAL][TABLE_DIM];
    end
  endgenerate

endmodule
