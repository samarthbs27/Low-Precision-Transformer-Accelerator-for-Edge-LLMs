// ============================================================
// mac_array.sv
// 8-Lane Parallel MAC Array with Accumulators
// Rijul — MAC Unit & Parallel Compute Core
//
// Interfaces directly with Samarth's control_fsm.sv signals:
//   mac_valid  — HIGH only during COMPUTE (accumulate when HIGH)
//   clear_acc  — pulse HIGH for 1 cycle to zero all accumulators
//   wr_en      — pulse HIGH for 1 cycle in WRITE state
//   wr_addr    — tile index (0..N/T-1) for output buffer write
//
// Dataflow:
//   x_in     : single INT8 value broadcast to all 8 lanes from Input BRAM
//   w_in     : 8 × INT8 weights, one per lane from Weight BRAM banks
//   Each lane: acc[j] += w_in[j] * x_in   (when mac_valid = HIGH)
//   After K steps: acc[0..7] are written to output buffer
//
// Parameters (must match control_fsm):
//   LANES  : number of MAC lanes (= T = 8)
//   K      : input/weight dimension (= 64)
//   N      : output dimension (= 64)
// ============================================================

module mac_array #(
    parameter int LANES = 8,
    parameter int K     = 64,
    parameter int N     = 64
)(
    input  logic        clk,
    input  logic        rst,          // synchronous active-high reset

    // ── Control signals from control_fsm ──────────────────
    input  logic        mac_valid,    // HIGH during COMPUTE state only
    input  logic        clear_acc,    // 1-cycle pulse: zero all accumulators
    input  logic        wr_en,        // 1-cycle pulse: write tile results out
    input  logic [$clog2(N/LANES)-1:0] wr_addr, // tile_idx → output slot

    // ── Data inputs ───────────────────────────────────────
    input  logic signed [7:0]  x_in,            // broadcast input x[k]
    input  logic signed [7:0]  w_in [LANES-1:0], // w_in[j] = W[tile*T+j, k]

    // ── Output buffer interface ───────────────────────────
    output logic signed [31:0] output_data [N-1:0], // full N-element result vector
    output logic               valid_out             // HIGH for 1 cycle when tile written
);

    // ── Accumulator registers: one INT32 per lane ─────────
    logic signed [31:0] acc [LANES-1:0];

    // ── MAC unit outputs (combinational) ──────────────────
    logic signed [31:0] mac_out [LANES-1:0];

    // ── Instantiate LANES mac_units ───────────────────────
    genvar j;
    generate
        for (j = 0; j < LANES; j++) begin : mac_lane
            mac_unit u_mac (
                .a       (w_in[j]),    // weight for lane j
                .b       (x_in),       // broadcast input
                .acc_in  (acc[j]),     // current accumulator
                .acc_out (mac_out[j])  // updated accumulator (combinational)
            );
        end
    endgenerate

    // ── Accumulator update + clear + write logic ──────────
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < LANES; i++)
                acc[i] <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0; // default: de-assert each cycle

            // Priority 1: clear — zero all accumulators before a new tile
            // FSM guarantees clear_acc and mac_valid are never both HIGH
            if (clear_acc) begin
                for (int i = 0; i < LANES; i++)
                    acc[i] <= '0;

            // Priority 2: accumulate — only when FSM is in COMPUTE state
            end else if (mac_valid) begin
                for (int i = 0; i < LANES; i++)
                    acc[i] <= mac_out[i];
            end

            // Priority 3: write tile result to output buffer
            // wr_en fires in WRITE state, after mac_valid has gone LOW
            if (wr_en) begin
                for (int i = 0; i < LANES; i++)
                    output_data[wr_addr * LANES + i] <= acc[i];
                valid_out <= 1'b1;
            end
        end
    end

endmodule
