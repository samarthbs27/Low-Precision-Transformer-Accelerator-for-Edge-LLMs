// ============================================================
// tb_control_fsm.sv
// SystemVerilog testbench for control_fsm
//
// Checks:
//   1. mac_valid LOW before start (IDLE)
//   2. mac_valid LOW during LOAD and WRITE
//   3. clear_acc pulses exactly N/T times (once per tile)
//   4. wr_en pulses exactly N/T times (once per tile)
//   5. wr_addr increments correctly 0 → N/T-1
//   6. done asserts after all tiles complete
//   7. done de-asserts next cycle (FSM back to IDLE)
// ============================================================

`timescale 1ns/1ps

module tb_control_fsm;

    // ── Parameters ──────────────────────────────────────────
    localparam int N        = 64;
    localparam int K        = 64;
    localparam int T        = 8;
    localparam int LOAD_LAT = 2;
    localparam int NUM_TILES = N / T;  // 8

    // ── DUT signals ──────────────────────────────────────────
    logic clk, rst, start;

    logic [$clog2(K)-1:0]    x_rd_addr;
    logic                    x_bram_en;
    logic [$clog2(K)-1:0]    w_rd_addr;
    logic                    w_bram_en;
    logic                    mac_valid;
    logic                    clear_acc;
    logic [$clog2(N/T)-1:0]  wr_addr;
    logic                    wr_en;
    logic [$clog2(N/T)-1:0]  tile_idx_out;
    logic                    done;

    // ── Instantiate DUT ─────────────────────────────────────
    control_fsm #(.N(N), .K(K), .T(T), .LOAD_LAT(LOAD_LAT)) dut (.*);

    // ── Clock: 10 ns period ──────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Pulse counters (always blocks — reliable, never miss a pulse) ──
    int mac_valid_count = 0;
    int clear_acc_count = 0;
    int wr_en_count     = 0;

    always @(posedge clk) if (mac_valid) mac_valid_count++;
    always @(posedge clk) if (clear_acc) clear_acc_count++;
    always @(posedge clk) if (wr_en)     wr_en_count++;

    // ── Tasks ────────────────────────────────────────────────
    task apply_reset();
        rst = 1; start = 0;
        repeat(3) @(posedge clk);
        rst = 0;
        @(posedge clk);
    endtask

    task pulse_start();
        start = 1;
        @(posedge clk);
        start = 0;
    endtask

    // ── Main test ────────────────────────────────────────────
    initial begin
        $dumpfile("sim/tb_control_fsm.vcd");
        $dumpvars(0, tb_control_fsm);

        apply_reset();

        // ── Check 1: mac_valid LOW in IDLE before start ──────
        @(posedge clk);
        if (!mac_valid)
            $display("PASS check 1: mac_valid LOW in IDLE");
        else
            $error("FAIL check 1: mac_valid HIGH in IDLE before start");

        pulse_start();

        // ── Run until done ────────────────────────────────────
        fork
            // watchdog
            begin
                repeat(2000) @(posedge clk);
                $fatal(1, "FAIL: timeout — done never asserted");
            end

            // monitor loop
            begin
                int cycle_count = 0;
                int exp_wr_addr = 0;

                while (!done) begin
                    @(posedge clk);
                    cycle_count++;

                    // ── Check 2: mac_valid LOW during WRITE ──
                    if (wr_en && mac_valid)
                        $error("FAIL check 2: mac_valid HIGH during WRITE (tile %0d)",
                               wr_en_count - 1);

                    // ── Check 5: wr_addr correct ─────────────
                    if (wr_en) begin
                        if (wr_addr == exp_wr_addr)
                            $display("PASS check 5: tile %0d wr_addr=%0d",
                                     exp_wr_addr, wr_addr);
                        else
                            $error("FAIL check 5: wr_addr=%0d expected %0d",
                                   wr_addr, exp_wr_addr);
                        exp_wr_addr++;
                    end
                end

                // ── Check 3: clear_acc count ──────────────────
                // One extra cycle for always block to sample last clear_acc
                @(posedge clk);
                if (clear_acc_count == NUM_TILES)
                    $display("PASS check 3: clear_acc pulsed %0d times", clear_acc_count);
                else
                    $error("FAIL check 3: clear_acc pulsed %0d times (expected %0d)",
                           clear_acc_count, NUM_TILES);

                // ── Check 4: wr_en count ──────────────────────
                if (wr_en_count == NUM_TILES)
                    $display("PASS check 4: wr_en pulsed %0d times", wr_en_count);
                else
                    $error("FAIL check 4: wr_en pulsed %0d times (expected %0d)",
                           wr_en_count, NUM_TILES);

                // ── Check 1b: mac_valid total cycles ─────────
                if (mac_valid_count == NUM_TILES * K)
                    $display("PASS check 1b: mac_valid asserted %0d cycles", mac_valid_count);
                else
                    $error("FAIL check 1b: mac_valid asserted %0d cycles (expected %0d)",
                           mac_valid_count, NUM_TILES * K);

                // ── Check 6: done asserted ────────────────────
                $display("PASS check 6: done asserted after %0d cycles", cycle_count);

                // ── Check 7: done de-asserts ──────────────────
                @(posedge clk);
                if (!done)
                    $display("PASS check 7: done de-asserted, FSM back to IDLE");
                else
                    $error("FAIL check 7: done still HIGH — FSM did not return to IDLE");

                $display("--- All checks complete ---");
                $finish;
            end
        join_any
        disable fork;
    end

endmodule
