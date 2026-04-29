# run_synth.ps1 — Launch Vivado synthesis for tinyllama_u55c_kernel_top.
# Run from project root: .\synth\run_synth.ps1
# Requires Vivado 2025.x on PATH.

param(
  [string]$VivadoBin = "E:\AMDDesignTools\2025.2\Vivado\bin"
)
$VivadoExe = Join-Path $VivadoBin "vivado.bat"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TclScript   = Join-Path $PSScriptRoot "run_synth.tcl"
$OutDir      = Join-Path $PSScriptRoot "out"
$LogDir      = Join-Path $PSScriptRoot "logs"
$Timestamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog      = Join-Path $LogDir "synth_$Timestamp.log"

New-Item -ItemType Directory -Force -Path $OutDir  | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir  | Out-Null

# Remove read-only flag OneDrive may have set on synth/out.
attrib -R "$OutDir" 2>$null

Write-Host "Starting Vivado synthesis at $Timestamp"
Write-Host "  TCL:  $TclScript"
Write-Host "  Log:  $RunLog"
Write-Host "  Out:  $OutDir"
Write-Host ""

# ── Pause OneDrive sync ───────────────────────────────────────────────────────
# OneDrive holds open file handles on files it is syncing.  Vivado creates
# .Xil/Vivado-PID/realtime/ in the project root (anchored by the TCL "cd"
# that makes relative source paths work).  When OneDrive locks those files,
# Vivado's cleanup raises [Designutils 20-411] which aborts synth_design
# BEFORE technology mapping runs — yielding an RTL-only netlist (0 LUTs/FFs).
# Stopping OneDrive for the duration of synthesis prevents all file locking.
$OneDriveExe = "$env:LOCALAPPDATA\Microsoft\OneDrive\OneDrive.exe"
$OneDriveWasRunning = $null -ne (Get-Process -Name OneDrive -ErrorAction SilentlyContinue)
if ($OneDriveWasRunning) {
  Write-Host "Pausing OneDrive sync for synthesis (~15 min)..."
  Stop-Process -Name OneDrive -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  Write-Host "OneDrive paused."
  Write-Host ""
}

# ── Run Vivado ────────────────────────────────────────────────────────────────
# Launch from %TEMP% so any stray .Xil writes outside the project also land
# outside OneDrive.  The TCL script does "cd $PROJ_ROOT" for source reading.
$VivadoWorkDir = Join-Path $env:TEMP "vivado_synth_tinyllama"
New-Item -ItemType Directory -Force -Path $VivadoWorkDir | Out-Null

$ExitCode = 1
try {
  Push-Location $VivadoWorkDir
  try {
    & $VivadoExe -mode batch -source $TclScript -log $RunLog -nojournal
    $ExitCode = $LASTEXITCODE
  } finally {
    Pop-Location
  }
} finally {
  # ── Restart OneDrive ───────────────────────────────────────────────────────
  if ($OneDriveWasRunning -and (Test-Path $OneDriveExe)) {
    Write-Host ""
    Write-Host "Restarting OneDrive..."
    Start-Process $OneDriveExe
  }
}

Write-Host ""
if ($ExitCode -eq 0) {
  Write-Host "PASS: Vivado exited cleanly (exit code 0)"

  # Post-synth utilization summary (from temp dir or synth/out)
  $TmpOutDir = Join-Path $env:TEMP "vivado_synth_tinyllama_out"
  $UtilRpt = if (Test-Path (Join-Path $OutDir "utilization_synth.rpt")) {
    Join-Path $OutDir "utilization_synth.rpt"
  } elseif (Test-Path (Join-Path $TmpOutDir "utilization_synth.rpt")) {
    Join-Path $TmpOutDir "utilization_synth.rpt"
  } else { $null }

  if ($UtilRpt) {
    Write-Host ""
    Write-Host "--- Utilization summary (post-synth) ---"
    Select-String -Path $UtilRpt -Pattern "LUT|FF|Register|BRAM|DSP|Slice|CLB" |
      Select-Object -First 20 | ForEach-Object { Write-Host $_.Line }
  }

  $TimingRpt = if (Test-Path (Join-Path $OutDir "timing_synth.rpt")) {
    Join-Path $OutDir "timing_synth.rpt"
  } elseif (Test-Path (Join-Path $TmpOutDir "timing_synth.rpt")) {
    Join-Path $TmpOutDir "timing_synth.rpt"
  } else { $null }

  if ($TimingRpt) {
    Write-Host ""
    Write-Host "--- Timing summary (post-synth estimate) ---"
    Select-String -Path $TimingRpt -Pattern "Worst Negative Slack|Design Timing Summary|VIOLATED|MET|WNS|TNS" |
      Select-Object -First 10 | ForEach-Object { Write-Host $_.Line }
  }
} else {
  Write-Host "FAIL: Vivado exited with code $ExitCode - check $RunLog"
}
