// ============================================================
// mac_unit.sv
// Single-Lane INT8 MAC Unit
// Rijul — MAC Unit & Parallel Compute Core
//
// Operation: acc_out = acc_in + (a * b)
//   a       : INT8  signed weight   W[j,k]
//   b       : INT8  signed input    x[k]   (broadcast from FSM)
//   acc_in  : INT32 running accumulator
//   acc_out : INT32 updated accumulator
//
// Notes:
//   - Purely combinational — accumulator register lives in mac_array
//   - Signed × Signed multiply produces a 16-bit product, then
//     sign-extended to 32 bits before adding to acc_in
//   - No saturation: INT8×INT8 max product = 127×127 = 16129,
//     and 64 such terms sum to at most 1,032,256 — well within INT32
// ============================================================

module mac_unit (
    input  logic signed [7:0]  a,        // INT8 signed weight
    input  logic signed [7:0]  b,        // INT8 signed input (broadcast)
    input  logic signed [31:0] acc_in,   // INT32 running accumulator
    output logic signed [31:0] acc_out   // INT32 updated accumulator
);

    // Use 32-bit intermediate so Icarus doesn't need bit-select inside always_comb
    logic signed [31:0] product;

    always_comb begin
        product = $signed(a) * $signed(b);   // INT8 × INT8, result widened to 32-bit signed
        acc_out = acc_in + product;
    end

endmodule
