# Timing constraint for tinyllama_u55c_kernel_top out-of-context synthesis.
# Target: 100 MHz (10 ns period). Synthesis timing is pessimistic without
# placement; real closure target is evaluated post-P&R.
create_clock -name ap_clk -period 10.0 [get_ports ap_clk]
