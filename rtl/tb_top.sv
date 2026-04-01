// ============================================================
// tb_top.sv
// Integration Testbench — top.sv (FSM + MAC Array)
// Samarth — Dataflow, Tiling & Control
//
// Test plan:
//   CHECK_1 : all-ones test — x=1, W=1 → y[i] = K for all i
//   CHECK_2 : identity-diagonal test — W=I (diagonal), x=1..N → y=x
//   CHECK_3 : done de-asserts after one cycle (DONE → IDLE)
//   CHECK_4 : second start produces identical results (no stale state)
//
// No external files required — all vectors generated inline.
// ============================================================

`timescale 1ns/1ps

module tb_top;

    // ── Parameters ────────────────────────────────────────
    localparam int N        = 64;
    localparam int K        = 64;
    localparam int T        = 8;
    localparam int LOAD_LAT = 2;
    localparam int K_W      = $clog2(K);
    localparam int N_W      = $clog2(N);

    // ── DUT signals ───────────────────────────────────────
    logic        clk, rst, start;

    logic                  x_wr_en;
    logic [K_W-1:0]        x_wr_addr;
    logic signed [7:0]     x_wr_data;

    logic                  w_wr_en;
    logic [N_W-1:0]        w_wr_row;
    logic [K_W-1:0]        w_wr_col;
    logic signed [7:0]     w_wr_data;

    logic [N-1:0][31:0]    output_data;
    logic                  valid_out;
    logic                  done;

    // ── Instantiate DUT ───────────────────────────────────
    top #(.N(N), .K(K), .T(T), .LOAD_LAT(LOAD_LAT)) dut (
        .clk         (clk),
        .rst         (rst),
        .start       (start),
        .x_wr_en     (x_wr_en),
        .x_wr_addr   (x_wr_addr),
        .x_wr_data   (x_wr_data),
        .w_wr_en     (w_wr_en),
        .w_wr_row    (w_wr_row),
        .w_wr_col    (w_wr_col),
        .w_wr_data   (w_wr_data),
        .output_data (output_data),
        .valid_out   (valid_out),
        .done        (done)
    );

    // ── Clock: 10 ns period ───────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Error counter ─────────────────────────────────────
    int errors;

    // ── Reset + idle signals ──────────────────────────────
    task automatic do_reset();
        rst      = 1'b1;
        start    = 1'b0;
        x_wr_en  = 1'b0;
        w_wr_en  = 1'b0;
        x_wr_addr = '0;
        x_wr_data = '0;
        w_wr_row  = '0;
        w_wr_col  = '0;
        w_wr_data = '0;
        repeat(3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);
    endtask

    // Load x_mem with a single byte value broadcast to all K entries
    task automatic load_x_scalar(input logic signed [7:0] val);
        for (int k = 0; k < K; k++) begin
            @(negedge clk);
            x_wr_en   = 1'b1;
            x_wr_addr = K_W'(k);
            x_wr_data = val;
        end
        @(posedge clk); #1;
        x_wr_en = 1'b0;
    endtask

    // Load x_mem with x[k] = (k % 128) — keeps values in INT8 range
    task automatic load_x_ramp();
        for (int k = 0; k < K; k++) begin
            @(negedge clk);
            x_wr_en   = 1'b1;
            x_wr_addr = K_W'(k);
            x_wr_data = 8'(k % 128);
        end
        @(posedge clk); #1;
        x_wr_en = 1'b0;
    endtask

    // Load w_mem: every entry = 1 (all-ones weight matrix)
    task automatic load_w_ones();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++) begin
                @(negedge clk);
                w_wr_en  = 1'b1;
                w_wr_row = N_W'(i);
                w_wr_col = K_W'(j);
                w_wr_data = 8'sd1;
            end
        @(posedge clk); #1;
        w_wr_en = 1'b0;
    endtask

    // Load w_mem: identity — W[i][j] = 1 if i==j (and i<K), else 0
    // For N=K=64 this gives y[i] = x[i] (but x[i] repeats every 128)
    // To make expected = x_ramp[i % K] straightforward we use W=I and x=ramp
    task automatic load_w_identity();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++) begin
                @(negedge clk);
                w_wr_en  = 1'b1;
                w_wr_row = N_W'(i);
                w_wr_col = K_W'(j);
                w_wr_data = (i == j) ? 8'sd1 : 8'sd0;
            end
        @(posedge clk); #1;
        w_wr_en = 1'b0;
    endtask

    // Pulse start, wait for done, return
    task automatic run_and_wait();
        @(negedge clk);
        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;
        // Wait for done with watchdog
        fork
            begin : wait_done
                @(posedge done);
                disable watchdog;
            end
            begin : watchdog
                repeat(4000) @(posedge clk);
                $fatal(1, "TIMEOUT: done never asserted");
                disable wait_done;
            end
        join
        // Let done de-assert (it pulses for 1 cycle in DONE state)
        // #1 lets NBA assignments settle before caller reads done
        @(posedge clk); #1;
    endtask

    // ── Main test ─────────────────────────────────────────
    initial begin
        $dumpfile("sim/tb_top.vcd");
        $dumpvars(0, tb_top);

        errors = 0;

        // ── CHECK_1: all-ones test ────────────────────────
        // x = [1,1,...,1], W = all 1s  →  y[i] = sum_k(1*1) = K
        $display("========================================");
        $display("tb_top: integration testbench");
        $display("N=%0d  K=%0d  T=%0d  LOAD_LAT=%0d", N, K, T, LOAD_LAT);
        $display("========================================");
        $display("\n-- CHECK_1: all-ones (y should be K=%0d) --", K);

        do_reset();
        load_w_ones();
        load_x_scalar(8'sd1);

        run_and_wait();

        begin
            int c1_err = 0;
            for (int i = 0; i < N; i++) begin
                if (output_data[i] !== K) begin
                    $display("FAIL CHECK_1 y[%0d]: got %0d exp %0d",
                        i, output_data[i], K);
                    c1_err++;
                end
            end
            if (c1_err == 0)
                $display("PASS CHECK_1: all %0d outputs == %0d", N, K);
            else
                errors += c1_err;
        end

        // ── CHECK_2: identity test ────────────────────────
        // W = I (diagonal), x[k] = k   →  y[i] = x[i] = i
        $display("\n-- CHECK_2: identity matrix (y[i] = x[i]) --");

        do_reset();
        load_w_identity();
        load_x_ramp();

        run_and_wait();

        begin
            int c2_err = 0;
            for (int i = 0; i < N; i++) begin
                int exp_val;
                exp_val = i % 128;   // matches load_x_ramp
                if (output_data[i] !== exp_val) begin
                    $display("FAIL CHECK_2 y[%0d]: got %0d exp %0d",
                        i, output_data[i], exp_val);
                    c2_err++;
                end
            end
            if (c2_err == 0)
                $display("PASS CHECK_2: identity test — all y[i] == x[i]");
            else
                errors += c2_err;
        end

        // ── CHECK_3: done de-asserts after 1 cycle ────────
        // Already verified implicitly by run_and_wait (waits for posedge done then 1 more clock).
        // Re-verify that done is LOW after that extra clock.
        $display("\n-- CHECK_3: done de-asserts after 1 cycle --");
        // At this point we've already advanced 1 cycle past done.
        if (done !== 1'b0) begin
            $display("FAIL CHECK_3: done still HIGH after 1 cycle");
            errors++;
        end else
            $display("PASS CHECK_3: done de-asserted correctly");

        // ── CHECK_4: second run produces same results ──────
        $display("\n-- CHECK_4: second run reproducibility --");
        // W and x still loaded from CHECK_2; just re-start
        @(negedge clk); start = 1'b1;
        @(posedge clk); #1; start = 1'b0;
        fork
            begin : wd2_done
                @(posedge done); disable wd2;
            end
            begin : wd2
                repeat(4000) @(posedge clk);
                $fatal(1, "TIMEOUT CHECK_4");
                disable wd2_done;
            end
        join
        @(posedge clk);

        begin
            int c4_err = 0;
            for (int i = 0; i < N; i++) begin
                int exp_val;
                exp_val = i % 128;
                if (output_data[i] !== exp_val) begin
                    $display("FAIL CHECK_4 y[%0d]: got %0d exp %0d",
                        i, output_data[i], exp_val);
                    c4_err++;
                end
            end
            if (c4_err == 0)
                $display("PASS CHECK_4: second run matches");
            else
                errors += c4_err;
        end

        // ── Final report ──────────────────────────────────
        $display("\n========================================");
        if (errors == 0)
            $display("ALL CHECKS PASSED — 0 errors");
        else
            $display("FAILED — %0d error(s)", errors);
        $display("========================================");

        $finish;
    end

endmodule
