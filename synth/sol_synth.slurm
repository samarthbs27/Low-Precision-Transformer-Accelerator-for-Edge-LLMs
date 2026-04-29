#!/bin/bash
#SBATCH --job-name=tinyllama_synth
#SBATCH --output=synth/logs/slurm_%j.out
#SBATCH --error=synth/logs/slurm_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # 8 threads for Vivado maxThreads=8
#SBATCH --mem=256G
#SBATCH --partition=public

# ── Load Vivado ───────────────────────────────────────────────────────────────
# Confirmed on SOL: xilinx/vivado-2022.1
module load xilinx/vivado-2022.1

# ── Run synthesis from project root ──────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p synth/out synth/logs

echo "Starting synthesis on $(hostname) at $(date)"
echo "Node memory: $(free -h | grep Mem)"
echo "Vivado: $(which vivado)"
vivado -version | head -3

vivado -mode batch \
       -source synth/run_synth_sol.tcl \
       -log    synth/logs/vivado_${SLURM_JOB_ID}.log \
       -nojournal

EXIT_CODE=$?
echo "Vivado exited with code $EXIT_CODE at $(date)"

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=== Utilization Summary ==="
  grep -E "LUT|FF|Register|BRAM|DSP|Slice|CLB" synth/out/utilization_synth.rpt | head -25
  echo ""
  echo "=== Timing Summary ==="
  grep -E "WNS|TNS|Worst Negative Slack|Design Timing" synth/out/timing_synth.rpt | head -10
else
  echo "FAIL: check synth/logs/vivado_${SLURM_JOB_ID}.log"
fi
