import tinyllama_pkg::*;
import tinyllama_bus_pkg::*;

module causal_mask_unit (
  input  runtime_mode_e          runtime_mode_i,
  input  logic [POS_W-1:0]       query_pos_base_i,
  input  logic [POS_W-1:0]       key_pos_base_i,
  input  logic [COUNT_W-1:0]     query_row_count_i,
  input  logic [COUNT_W-1:0]     key_col_count_i,
  input  acc_bus_t               score_i,
  output acc_bus_t               masked_o
);

  acc_bus_t masked_bus_d;
  wire signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] masked_data_w;

  assign masked_o = masked_bus_d;

  always_comb begin
    masked_bus_d = '0;
    masked_bus_d.tag = score_i.tag;
    masked_bus_d.tag.block_id = BLOCK_CAUSAL_MASK;
    masked_bus_d.tag.gemm_mode = GEMM_SCORE;
    masked_bus_d.tag.token_base = query_pos_base_i;
    masked_bus_d.tag.seq_count = query_row_count_i;
    masked_bus_d.tag.elem_count = ELEM_COUNT_W'(query_row_count_i * SCORE_K_TILE);
    masked_bus_d.tag.is_partial =
      (query_row_count_i != SCORE_ROWS_PER_CHUNK) || (key_col_count_i != SCORE_K_TILE);
    masked_bus_d.data = masked_data_w;
  end

  generate
    for (genvar lane = 0; lane < ACC_VECTOR_ELEMS; lane++) begin : g_mask_lane
      localparam int unsigned ROW_LOCAL = lane / SCORE_K_TILE;
      localparam int unsigned COL_LOCAL = lane % SCORE_K_TILE;
      wire row_active_w;
      wire col_active_w;
      wire [POS_W-1:0] query_pos_w;
      wire [POS_W-1:0] key_pos_w;
      wire mode_supported_w;
      wire allow_score_w;

      assign row_active_w = (ROW_LOCAL < query_row_count_i);
      assign col_active_w = (COL_LOCAL < key_col_count_i);
      assign query_pos_w = query_pos_base_i + POS_W'(ROW_LOCAL);
      assign key_pos_w = key_pos_base_i + POS_W'(COL_LOCAL);
      assign mode_supported_w =
        (runtime_mode_i == MODE_PREFILL) || (runtime_mode_i == MODE_DECODE);
      assign allow_score_w = mode_supported_w && row_active_w && col_active_w &&
                             (key_pos_w <= query_pos_w);

      assign masked_data_w[lane] =
        !row_active_w ? '0 :
        (allow_score_w ? score_i.data[lane] : MASK_NEG_INF);
    end
  endgenerate

endmodule
