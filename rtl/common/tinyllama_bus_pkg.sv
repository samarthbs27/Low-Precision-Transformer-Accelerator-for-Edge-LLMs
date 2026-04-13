package tinyllama_bus_pkg;

  import tinyllama_pkg::*;

  typedef struct packed {
    logic [LAYER_ID_W-1:0]   layer_id;
    block_id_e               block_id;
    gemm_mode_e              gemm_mode;
    logic [TILE_ID_W-1:0]    tile_id;
    logic [POS_W-1:0]        token_base;
    logic [COUNT_W-1:0]      seq_count;
    logic [Q_HEAD_ID_W-1:0]  q_head_id;
    logic [KV_HEAD_ID_W-1:0] kv_head_id;
    logic [ELEM_COUNT_W-1:0] elem_count;
    logic                    is_last;
    logic                    is_partial;
  } tile_tag_t;

  typedef struct packed {
    logic [TOKEN_W-1:0] token_id;
    logic [COUNT_W-1:0] token_count;
    tile_tag_t          tag;
  } token_bus_t;

  typedef struct packed {
    logic signed [ACT_VECTOR_ELEMS-1:0][ACT_W-1:0] data;
    tile_tag_t                                      tag;
  } act_bus_t;

  typedef struct packed {
    logic signed [WEIGHT_VECTOR_ELEMS-1:0][WEIGHT_W-1:0] data;
    tile_tag_t                                           tag;
  } wt_bus_t;

  typedef struct packed {
    logic signed [ACC_VECTOR_ELEMS-1:0][ACC_W-1:0] data;
    tile_tag_t                                     tag;
  } acc_bus_t;

  typedef struct packed {
    logic [SCALE_VECTOR_ELEMS-1:0][SCALE_W-1:0] data;
    tile_tag_t                                  tag;
  } scale_bus_t;

  typedef struct packed {
    logic [DEBUG_BUS_W-1:0] data;
    tile_tag_t             tag;
  } dbg_bus_t;

  typedef struct packed {
    hbm_region_e               region;
    tensor_id_e                tensor_id;
    logic                      write_not_read;
    logic [PC_ID_W-1:0]        pseudo_channel;
    logic [HBM_ADDR_W-1:0]     addr;
    logic [15:0]               burst_len;
    logic [31:0]               byte_count;
    logic [LAYER_ID_W-1:0]     layer_id;
    logic [KV_HEAD_ID_W-1:0]   kv_head_id;
    logic [TILE_ID_W-1:0]      tile_id;
  } dma_desc_t;

  typedef struct packed {
    logic [TOKEN_W-1:0]      token_id;
    logic [HBM_ADDR_W-1:0]   write_addr;
    logic [COUNT_W-1:0]      token_index;
  } token_write_desc_t;

endpackage
