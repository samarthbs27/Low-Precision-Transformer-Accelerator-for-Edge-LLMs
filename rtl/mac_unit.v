// ============================================================
// mac_unit.v
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
    input  wire signed [7:0]  a,        // INT8 signed weight
    input  wire signed [7:0]  b,        // INT8 signed input (broadcast)
    input  wire signed [31:0] acc_in,   // INT32 running accumulator
    output wire signed [31:0] acc_out   // INT32 updated accumulator
);

    // 16-bit product, sign-extended implicitly to 32 bits by Verilog
    // signed arithmetic rules when operands are declared signed
    wire signed [15:0] product;
    assign product  = a * b;                            // INT8 × INT8 → INT16
    assign acc_out  = acc_in + {{16{product[15]}}, product}; // sign-extend → INT32, add

endmodule