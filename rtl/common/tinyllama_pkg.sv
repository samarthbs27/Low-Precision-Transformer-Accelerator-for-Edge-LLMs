package tinyllama_pkg;

  // Core TinyLlama architectural constants.
  localparam int unsigned N_LAYERS      = 22;
  localparam int unsigned D_MODEL       = 2048;
  localparam int unsigned D_FF          = 5632;
  localparam int unsigned N_Q_HEADS     = 32;
  localparam int unsigned N_KV_HEADS    = 4;
  localparam int unsigned KV_GROUPS     = 8;
  localparam int unsigned HEAD_DIM      = 64;
  localparam int unsigned VOCAB_SIZE    = 32000;
  localparam int unsigned MAX_POS       = 2048;
  localparam int unsigned SEQ_TILE      = 64;

  // Datapath and quantization widths.
  localparam int unsigned ACT_W         = 8;
  localparam int unsigned WEIGHT_W      = 8;
  localparam int unsigned ACC_W         = 32;
  localparam int unsigned TOKEN_W       = 32;
  localparam int unsigned SCALE_W       = 32;
  localparam int unsigned PROB_W        = 8;
  localparam int unsigned AXIL_DATA_W   = 32;
  localparam int unsigned AXIL_STRB_W   = AXIL_DATA_W / 8;
  localparam int unsigned AXIL_ADDR_W   = 12;
  localparam int unsigned REG_WORD_ADDR_W = AXIL_ADDR_W - 2;

  // Memory and tiling parameters.
  localparam int unsigned GEMM_LANES       = 512;
  localparam int unsigned HBM_PC_COUNT     = 32;
  localparam int unsigned HBM_ADDR_W       = 64;
  localparam int unsigned DMA_BEAT_W       = 256;
  localparam int unsigned DMA_BEAT_BYTES   = DMA_BEAT_W / 8;
  localparam int unsigned TOKENS_PER_DMA_BEAT = DMA_BEAT_W / TOKEN_W;
  localparam int unsigned STREAM_FIFO_DEPTH = 4;
  localparam int unsigned SKID_BUFFER_DEPTH = 2;
  localparam int unsigned DESC_FIFO_DEPTH   = 8;
  localparam int unsigned DEBUG_FIFO_DEPTH  = 32;
  localparam int unsigned TILE_BUFFER_BANKS = 16;
  localparam int unsigned BANK_SLICE_INT8   = 32;
  localparam int unsigned BANK_SLICE_INT32  = 8;
  localparam int unsigned M_TILE            = 16;
  localparam int unsigned N_TILE            = 32;
  localparam int unsigned K_TILE            = 64;
  localparam int unsigned SCORE_Q_TILE      = 16;
  localparam int unsigned SCORE_K_TILE      = 64;
  localparam int unsigned VOCAB_TILE        = 128;
  localparam int unsigned HEAD_GROUP_PAR    = 1;

  // Common vector widths.
  localparam int unsigned ACT_VECTOR_ELEMS   = GEMM_LANES;
  localparam int unsigned WEIGHT_VECTOR_ELEMS = GEMM_LANES;
  localparam int unsigned ACC_VECTOR_ELEMS   = GEMM_LANES;
  localparam int unsigned SCORE_TILE_ELEMS   = SCORE_Q_TILE * SCORE_K_TILE;
  localparam int unsigned LMHEAD_TILE_ELEMS  = VOCAB_TILE;
  localparam int unsigned SCALE_VECTOR_ELEMS = TILE_BUFFER_BANKS;
  localparam int unsigned DEBUG_BUS_W        = DMA_BEAT_W;

  // Shared ID widths.
  localparam int unsigned LAYER_ID_W    = (N_LAYERS > 1) ? $clog2(N_LAYERS) : 1;
  localparam int unsigned BLOCK_ID_W    = 6;
  localparam int unsigned GEMM_MODE_W   = 4;
  localparam int unsigned TILE_ID_W     = 16;
  localparam int unsigned POS_W         = (MAX_POS > 1) ? $clog2(MAX_POS) : 1;
  localparam int unsigned COUNT_W       = $clog2(MAX_POS + 1);
  localparam int unsigned Q_HEAD_ID_W   = (N_Q_HEADS > 1) ? $clog2(N_Q_HEADS) : 1;
  localparam int unsigned KV_HEAD_ID_W  = (N_KV_HEADS > 1) ? $clog2(N_KV_HEADS) : 1;
  localparam int unsigned BANK_ID_W     = (TILE_BUFFER_BANKS > 1) ? $clog2(TILE_BUFFER_BANKS) : 1;
  localparam int unsigned PC_ID_W       = (HBM_PC_COUNT > 1) ? $clog2(HBM_PC_COUNT) : 1;
  localparam int unsigned ELEM_COUNT_W  = 16;
  localparam int unsigned ERROR_CODE_W  = 4;
  localparam int unsigned STOP_REASON_W = 3;
  localparam int unsigned TENSOR_ID_W   = 5;
  localparam int unsigned REGION_ID_W   = 3;
  localparam logic [PC_ID_W-1:0] HOST_IO_PC_ID = PC_ID_W'(30);

  // Host command/status block layout in PC30.
  localparam int unsigned HOST_BLOCK_WORDS = DMA_BEAT_W / AXIL_DATA_W;
  localparam int unsigned HOST_BLOCK_BYTES = DMA_BEAT_BYTES;
  localparam int unsigned HOST_CMD_WORD_PROMPT_BASE_LO = 0;
  localparam int unsigned HOST_CMD_WORD_PROMPT_BASE_HI = 1;
  localparam int unsigned HOST_CMD_WORD_GEN_BASE_LO    = 2;
  localparam int unsigned HOST_CMD_WORD_GEN_BASE_HI    = 3;
  localparam int unsigned HOST_CMD_WORD_GEN_CAPACITY   = 4;
  localparam int unsigned HOST_STATUS_WORD_STATUS      = 0;
  localparam int unsigned HOST_STATUS_WORD_GEN_COUNT   = 1;
  localparam int unsigned HOST_STATUS_WORD_LAST_TOKEN  = 2;
  localparam int unsigned HOST_STATUS_WORD_CUR_LAYER   = 3;
  localparam int unsigned HOST_STATUS_WORD_CUR_BLOCK   = 4;
  localparam int unsigned HOST_STATUS_WORD_VERSION     = 7;

  // AXI-Lite register map word indices.
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_CONTROL               = 'd0;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_STATUS                = 'd1;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_CMD_BASE_LO           = 'd2;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_CMD_BASE_HI           = 'd3;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_STATUS_BASE_LO        = 'd4;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_STATUS_BASE_HI        = 'd5;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_DEBUG_BASE_LO         = 'd6;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_DEBUG_BASE_HI         = 'd7;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_PROMPT_TOKEN_COUNT    = 'd8;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_MAX_NEW_TOKENS        = 'd9;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_EOS_TOKEN_ID          = 'd10;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_DEBUG_CFG             = 'd11;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_GENERATED_TOKEN_COUNT = 'd12;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_LAST_TOKEN_ID         = 'd13;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_CURRENT_LAYER         = 'd14;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_CURRENT_BLOCK         = 'd15;
  localparam logic [REG_WORD_ADDR_W-1:0] REGW_VERSION               = 'd16;

  // AXI-Lite control register bit definitions.
  localparam int unsigned CTRL_START_BIT       = 0;
  localparam int unsigned CTRL_MODE_BIT        = 1;
  localparam int unsigned CTRL_ABORT_REQ_BIT   = 2;

  localparam int unsigned STATUS_BUSY_BIT      = 0;
  localparam int unsigned STATUS_DONE_BIT      = 1;
  localparam int unsigned STATUS_ERROR_BIT     = 2;
  localparam int unsigned STATUS_STOP_VALID_BIT = 3;
  localparam int unsigned STATUS_STOP_REASON_LSB = 4;
  localparam int unsigned STATUS_STOP_REASON_MSB = STATUS_STOP_REASON_LSB + STOP_REASON_W - 1;
  localparam int unsigned STATUS_ERROR_CODE_LSB = 8;
  localparam int unsigned STATUS_ERROR_CODE_MSB = STATUS_ERROR_CODE_LSB + ERROR_CODE_W - 1;

  localparam int unsigned DEBUG_CFG_ENABLE_BIT = 0;
  localparam int unsigned DEBUG_CFG_LAYER_LSB  = 4;
  localparam int unsigned DEBUG_CFG_LAYER_MSB  = DEBUG_CFG_LAYER_LSB + LAYER_ID_W - 1;
  localparam int unsigned DEBUG_CFG_STEP_LSB   = 12;
  localparam int unsigned DEBUG_CFG_STEP_W     = 8;
  localparam int unsigned DEBUG_CFG_STEP_MSB   = DEBUG_CFG_STEP_LSB + DEBUG_CFG_STEP_W - 1;

  localparam logic [31:0] RTL_VERSION_WORD     = 32'h0001_0000;

  typedef enum logic {
    MODE_PREFILL = 1'b0,
    MODE_DECODE  = 1'b1
  } runtime_mode_e;

  typedef enum logic [GEMM_MODE_W-1:0] {
    GEMM_NONE         = 4'd0,
    GEMM_Q            = 4'd1,
    GEMM_K            = 4'd2,
    GEMM_V            = 4'd3,
    GEMM_SCORE        = 4'd4,
    GEMM_WEIGHTED_SUM = 4'd5,
    GEMM_O            = 4'd6,
    GEMM_GATE         = 4'd7,
    GEMM_UP           = 4'd8,
    GEMM_DOWN         = 4'd9,
    GEMM_LM_HEAD      = 4'd10
  } gemm_mode_e;

  typedef enum logic [BLOCK_ID_W-1:0] {
    BLOCK_NONE           = 6'd0,
    BLOCK_EMBED          = 6'd1,
    BLOCK_RMSNORM1       = 6'd2,
    BLOCK_Q              = 6'd3,
    BLOCK_K              = 6'd4,
    BLOCK_V              = 6'd5,
    BLOCK_ROPE           = 6'd6,
    BLOCK_KV_CACHE_WRITE = 6'd7,
    BLOCK_SCORE          = 6'd8,
    BLOCK_CAUSAL_MASK    = 6'd9,
    BLOCK_SOFTMAX        = 6'd10,
    BLOCK_WEIGHTED_SUM   = 6'd11,
    BLOCK_O              = 6'd12,
    BLOCK_RESIDUAL1      = 6'd13,
    BLOCK_REQUANTIZE     = 6'd14,
    BLOCK_RMSNORM2       = 6'd15,
    BLOCK_GATE           = 6'd16,
    BLOCK_UP             = 6'd17,
    BLOCK_SILU           = 6'd18,
    BLOCK_GLU_MUL        = 6'd19,
    BLOCK_DOWN           = 6'd20,
    BLOCK_RESIDUAL2      = 6'd21,
    BLOCK_FINAL_RMSNORM  = 6'd22,
    BLOCK_LM_HEAD        = 6'd23,
    BLOCK_ARGMAX         = 6'd24,
    BLOCK_DEBUG          = 6'd25
  } block_id_e;

  typedef enum logic [TENSOR_ID_W-1:0] {
    TENSOR_NONE            = 5'd0,
    TENSOR_WQ              = 5'd1,
    TENSOR_WK              = 5'd2,
    TENSOR_WV              = 5'd3,
    TENSOR_WO              = 5'd4,
    TENSOR_WGATE           = 5'd5,
    TENSOR_WUP             = 5'd6,
    TENSOR_WDOWN           = 5'd7,
    TENSOR_RMSNORM1_GAMMA  = 5'd8,
    TENSOR_RMSNORM2_GAMMA  = 5'd9,
    TENSOR_FINAL_RMS_GAMMA = 5'd10,
    TENSOR_EMBED           = 5'd11,
    TENSOR_LM_HEAD         = 5'd12,
    TENSOR_SCALE_META      = 5'd13
  } tensor_id_e;

  typedef enum logic [REGION_ID_W-1:0] {
    REGION_LAYER_WEIGHTS = 3'd0,
    REGION_EMBED_META    = 3'd1,
    REGION_LM_HEAD       = 3'd2,
    REGION_K_CACHE       = 3'd3,
    REGION_V_CACHE       = 3'd4,
    REGION_HOST_IO       = 3'd5,
    REGION_DEBUG         = 3'd6
  } hbm_region_e;

  typedef enum logic [STOP_REASON_W-1:0] {
    STOP_REASON_NONE         = 3'd0,
    STOP_REASON_EOS          = 3'd1,
    STOP_REASON_MAX_TOKENS   = 3'd2,
    STOP_REASON_HOST_ABORT   = 3'd3,
    STOP_REASON_INTERNAL_ERR = 3'd4
  } stop_reason_e;

  typedef enum logic [ERROR_CODE_W-1:0] {
    ERROR_NONE               = 4'd0,
    ERROR_BAD_DESCRIPTOR     = 4'd1,
    ERROR_HBM_READ           = 4'd2,
    ERROR_HBM_WRITE          = 4'd3,
    ERROR_SCALE_METADATA     = 4'd4,
    ERROR_DEBUG_OVERFLOW     = 4'd5,
    ERROR_UNSUPPORTED_MODE   = 4'd6,
    ERROR_INTERNAL_ASSERT    = 4'd7
  } error_code_e;

endpackage
