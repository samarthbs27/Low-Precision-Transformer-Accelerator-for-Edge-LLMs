// ============================================================
// top.sv
// Top-Level Integration: Control FSM + MAC Array
// Samarth — Dataflow, Tiling & Control
//
// Wires control_fsm → mac_array with behavioral BRAM stubs.
// In the real U55C design these stubs become Xilinx BRAM macros
// pre-loaded by the host via PCIe (Om's domain).
//
// BRAM reads are combinational: the FSM's registered x_rd_addr /
// w_rd_addr output already provides the required 1-cycle pipeline
// stage between k_idx and the data seen by mac_array.
//
// Parameters
//   N        : output dimension (rows of W)      default 64
//   K        : input dimension  (cols of W / x)  default 64
//   T        : tile size = MAC lanes              default 8
//   LOAD_LAT : BRAM read latency in cycles        default 2
// ============================================================

module top #(
    parameter int N        = 64,
    parameter int K        = 64,
    parameter int T        = 8,
    parameter int LOAD_LAT = 2
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    // ── Host write port: load input vector x ──────────────
    input  logic                    x_wr_en,
    input  logic [$clog2(K)-1:0]   x_wr_addr,
    input  logic signed [7:0]       x_wr_data,

    // ── Host write port: load weight matrix W ──────────────
    // Addressed as W[row][col], row in [0..N-1], col in [0..K-1]
    input  logic                    w_wr_en,
    input  logic [$clog2(N)-1:0]   w_wr_row,
    input  logic [$clog2(K)-1:0]   w_wr_col,
    input  logic signed [7:0]       w_wr_data,

    // ── Output: full result vector (packed for Icarus compat) ─
    output logic [N-1:0][31:0] output_data,
    output logic               valid_out,
    output logic               done
);

    // ── Derived widths ──────────────────────────────────────
    localparam int K_W    = $clog2(K);
    localparam int TILE_W = $clog2(N/T);

    // ── BRAM stubs ──────────────────────────────────────────
    logic signed [7:0] x_mem [K-1:0];
    logic signed [7:0] w_mem [N-1:0][K-1:0];

    // ── Host write ports (synchronous) ─────────────────────
    always_ff @(posedge clk) begin
        if (x_wr_en)
            x_mem[x_wr_addr] <= x_wr_data;
        if (w_wr_en)
            w_mem[w_wr_row][w_wr_col] <= w_wr_data;
    end

    // ── FSM control signals ─────────────────────────────────
    logic [K_W-1:0]    x_rd_addr, w_rd_addr;
    logic              x_bram_en, w_bram_en;
    logic              mac_valid, clear_acc;
    logic [TILE_W-1:0] wr_addr, tile_idx_out;
    logic              wr_en;

    // ── Control FSM ────────────────────────────────────────
    control_fsm #(.N(N), .K(K), .T(T), .LOAD_LAT(LOAD_LAT)) u_fsm (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .x_rd_addr    (x_rd_addr),
        .x_bram_en    (x_bram_en),
        .w_rd_addr    (w_rd_addr),
        .w_bram_en    (w_bram_en),
        .mac_valid    (mac_valid),
        .clear_acc    (clear_acc),
        .wr_addr      (wr_addr),
        .wr_en        (wr_en),
        .tile_idx_out (tile_idx_out),
        .done         (done)
    );

    // ── Combinational BRAM reads ────────────────────────────
    // FSM's registered x_rd_addr/w_rd_addr already provides the
    // 1-cycle pipeline delay between k_idx and data reaching mac_array.
    // A second register stage here would lose the last k sample.
    logic signed [7:0] x_rd_data;
    logic signed [7:0] w_rd_data [T-1:0];

    assign x_rd_data = x_mem[x_rd_addr];

    genvar j;
    generate
        for (j = 0; j < T; j++) begin : w_read
            assign w_rd_data[j] = w_mem[tile_idx_out * T + j][w_rd_addr];
        end
    endgenerate

    // ── MAC Array ───────────────────────────────────────────
    mac_array #(.LANES(T), .K(K), .N(N)) u_mac (
        .clk         (clk),
        .rst         (rst),
        .mac_valid   (mac_valid),
        .clear_acc   (clear_acc),
        .wr_en       (wr_en),
        .wr_addr     (wr_addr),
        .x_in        (x_rd_data),
        .w_in        (w_rd_data),
        .output_data (output_data),
        .valid_out   (valid_out)
    );

endmodule
