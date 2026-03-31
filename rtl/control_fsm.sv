// ============================================================
// control_fsm.sv
// Dataflow, Tiling & Control — Samarth
//
// Drives the 8-lane MAC array through tiled matrix-vector
// multiplication:  y = W * x  (one linear layer)
//
// FSM states: IDLE → LOAD → COMPUTE → WRITE → (LOAD | DONE)
//
// Parameters
//   N        : output dimension (rows of W)
//   K        : input dimension  (cols of W, length of x)
//   T        : tile size = number of MAC lanes
//   LOAD_LAT : BRAM read latency in cycles
// ============================================================

module control_fsm #(
    parameter int N        = 64,
    parameter int K        = 64,
    parameter int T        = 8,
    parameter int LOAD_LAT = 2
)(
    input  logic clk,
    input  logic rst,    // synchronous active-high reset
    input  logic start,  // pulse high to begin computation

    // ── Input Vector BRAM ──────────────────────────────────
    output logic [$clog2(K)-1:0]     x_rd_addr,
    output logic                     x_bram_en,

    // ── Weight Tile BRAMs (8 banks, shared address) ────────
    output logic [$clog2(K)-1:0]     w_rd_addr,  // same for all 8 banks
    output logic                     w_bram_en,

    // ── MAC Array control (→ Rijul) ────────────────────────
    output logic                     mac_valid,  // HIGH only in COMPUTE
    output logic                     clear_acc,  // HIGH for exactly 1 cycle before COMPUTE

    // ── Output Buffer control ──────────────────────────────
    output logic [$clog2(N/T)-1:0]  wr_addr,    // = tile_idx
    output logic                     wr_en,      // HIGH for exactly 1 cycle in WRITE

    // ── Host interface (→ Om) ──────────────────────────────
    output logic                     done
);

    // ── Derived widths ──────────────────────────────────────
    localparam int K_W    = $clog2(K);          // width for k_idx counter
    localparam int TILE_W = $clog2(N/T);        // width for tile_idx counter
    localparam int LAT_W  = $clog2(LOAD_LAT+1); // width for latency counter (safe for any LOAD_LAT)

    // ── State type ──────────────────────────────────────────
    typedef enum logic [2:0] {
        IDLE    = 3'd0,
        LOAD    = 3'd1,
        COMPUTE = 3'd2,
        WRITE   = 3'd3,
        DONE    = 3'd4
    } state_t;

    state_t state, next_state;

    // ── Counters ────────────────────────────────────────────
    logic [K_W-1:0]    k_idx;
    logic [TILE_W-1:0] tile_idx;
    logic [LAT_W-1:0]  lat_cnt;

    // ── State register ──────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) state <= IDLE;
        else     state <= next_state;
    end

    // ── Next-state logic ────────────────────────────────────
    always_comb begin
        next_state = state;
        unique case (state)
            IDLE:    if (start)        next_state = LOAD;
            LOAD:    if (lat_cnt == 0) next_state = COMPUTE;
            COMPUTE: if (k_idx == K_W'(K-1)) next_state = WRITE;
            WRITE:   next_state = (tile_idx == TILE_W'(N/T-1)) ? DONE : LOAD;
            DONE:    next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // ── Datapath & output logic ─────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            k_idx     <= '0;
            tile_idx  <= '0;
            lat_cnt   <= LAT_W'(LOAD_LAT - 1);
            x_rd_addr <= '0;
            w_rd_addr <= '0;
            x_bram_en <= 1'b0;
            w_bram_en <= 1'b0;
            mac_valid <= 1'b0;
            clear_acc <= 1'b0;
            wr_en     <= 1'b0;
            wr_addr   <= '0;
            done      <= 1'b0;
        end else begin
            // default: de-assert all one-cycle pulses each clock
            mac_valid <= 1'b0;
            clear_acc <= 1'b0;
            wr_en     <= 1'b0;
            done      <= 1'b0;
            x_bram_en <= 1'b0;
            w_bram_en <= 1'b0;

            unique case (state)

                // ── IDLE ──────────────────────────────────
                IDLE: begin
                    k_idx    <= '0;
                    tile_idx <= '0;
                    if (start) begin
                        clear_acc <= 1'b1;               // 1-cycle pulse, resets accumulators
                        lat_cnt   <= LAT_W'(LOAD_LAT-1); // prime latency counter
                    end
                end

                // ── LOAD ──────────────────────────────────
                // Present BRAM addresses; wait for read latency
                // mac_valid stays LOW — data not yet valid
                LOAD: begin
                    x_bram_en <= 1'b1;
                    w_bram_en <= 1'b1;
                    x_rd_addr <= k_idx;
                    w_rd_addr <= k_idx;
                    if (lat_cnt != '0)
                        lat_cnt <= lat_cnt - 1'b1;
                end

                // ── COMPUTE ───────────────────────────────
                // Data valid on BRAM outputs; assert mac_valid every cycle
                COMPUTE: begin
                    x_bram_en <= 1'b1;
                    w_bram_en <= 1'b1;
                    mac_valid <= 1'b1;
                    x_rd_addr <= k_idx;
                    w_rd_addr <= k_idx;
                    k_idx     <= (k_idx == K_W'(K-1)) ? '0 : k_idx + 1'b1;
                end

                // ── WRITE ─────────────────────────────────
                // K loop done; write 8 accumulators to output buffer (1 cycle)
                WRITE: begin
                    wr_en   <= 1'b1;
                    wr_addr <= tile_idx;
                    if (tile_idx == TILE_W'(N/T-1)) begin
                        // last tile — DONE next, leave tile_idx as-is
                    end else begin
                        tile_idx  <= tile_idx + 1'b1;
                        clear_acc <= 1'b1;               // 1-cycle pulse before next COMPUTE
                        lat_cnt   <= LAT_W'(LOAD_LAT-1);
                    end
                end

                // ── DONE ──────────────────────────────────
                DONE: begin
                    done     <= 1'b1;
                    tile_idx <= '0;
                    k_idx    <= '0;
                end

                default: ;
            endcase
        end
    end

endmodule
