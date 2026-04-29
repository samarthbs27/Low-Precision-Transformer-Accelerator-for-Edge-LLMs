param(
  [ValidateSet("xsim", "iverilog")]
  [string]$Simulator = "xsim",
  [string]$VivadoBin = "E:\\AMDDesignTools\\2025.2\\Vivado\\bin",
  [string]$FixtureBase = "sim/golden_traces/phase6/rtl/phase6_decode_embedding_quantizer_batch0",
  [switch]$FullSoftmax
)

$ErrorActionPreference = "Stop"

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory
  )

  Push-Location $WorkingDirectory
  try {
    $output = & $FilePath @ArgumentList 2>&1
    $exitCode = $LASTEXITCODE
  }
  finally {
    Pop-Location
  }

  $outputText = if ($null -ne $output) {
    ($output | Out-String)
  }
  else {
    ""
  }

  Set-Content -Path $LogPath -Value $outputText

  if ($outputText.Length -gt 0) {
    Write-Host $outputText.TrimEnd()
  }

  return $exitCode
}

function Resolve-RepoRelativeFileList {
  param(
    [Parameter(Mandatory = $true)][string]$SourceFileList,
    [Parameter(Mandatory = $true)][string]$ResolvedFileList,
    [Parameter(Mandatory = $true)][string]$RepoRoot
  )

  $sourceDir = Split-Path -Parent $SourceFileList
  Get-Content $SourceFileList |
    Where-Object { $_.Trim() -ne "" } |
    ForEach-Object {
      $fullPath = [System.IO.Path]::GetFullPath((Join-Path $sourceDir $_.Trim()))
      if ($fullPath.StartsWith($RepoRoot + [System.IO.Path]::DirectorySeparatorChar)) {
        $fullPath.Substring($RepoRoot.Length + 1)
      }
      else {
        $fullPath
      }
    } |
    Set-Content $ResolvedFileList
}

function Resolve-RepoPath {
  param(
    [Parameter(Mandatory = $true)][string]$PathValue,
    [Parameter(Mandatory = $true)][string]$RepoRoot
  )

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Stage-FixturesForXsim {
  param(
    [Parameter(Mandatory = $true)][string]$FixtureBaseResolved,
    [Parameter(Mandatory = $true)][string]$RunDirectory
  )

  $fixtureParent = Split-Path -Parent $FixtureBaseResolved
  $fixtureLeaf = Split-Path -Leaf $FixtureBaseResolved
  $fixtureStageDir = Join-Path $RunDirectory "sim/golden_traces/phase6/rtl"
  $fixtureGlob = Join-Path $fixtureParent ($fixtureLeaf + "*")

  New-Item -ItemType Directory -Force -Path $fixtureStageDir | Out-Null
  Copy-Item -Path $fixtureGlob -Destination $fixtureStageDir -Force
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRootRelative = "sim\\logs\\tb_runtime_decoder_datapath\\$stamp"
$logRoot = Join-Path $repoRoot $logRootRelative
$resolvedFileListRelative = "$logRootRelative\\decoder_filelist_resolved.txt"
$resolvedFileList = Join-Path $repoRoot $resolvedFileListRelative
$sourceFileList = Join-Path $scriptDir "decoder_filelist.txt"
$fixtureBaseResolved = Resolve-RepoPath -PathValue $FixtureBase -RepoRoot $repoRoot
$buildArgsLog = Join-Path $logRoot "build_args.txt"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Resolve-RepoRelativeFileList -SourceFileList $sourceFileList -ResolvedFileList $resolvedFileList -RepoRoot $repoRoot
Stage-FixturesForXsim -FixtureBaseResolved $fixtureBaseResolved -RunDirectory $logRoot

if ($Simulator -eq "iverilog") {
  $compileLog = Join-Path $logRoot "compile.log"
  $runLog = Join-Path $logRoot "run.log"
  $vvpPathRelative = "$logRootRelative\\tb_runtime_decoder_datapath.vvp"

  if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    throw "iverilog is not on PATH."
  }
  if (-not (Get-Command vvp -ErrorAction SilentlyContinue)) {
    throw "vvp is not on PATH."
  }

  Write-Host "Compiling tb_runtime_decoder_datapath with Icarus into $vvpPathRelative"
  $iverilogArgs = @("-g2012")
  if ($FullSoftmax) {
    $iverilogArgs += @("-DNO_FAST_SOFTMAX")
    Write-Host "Icarus build mode: full softmax (NO_FAST_SOFTMAX defined)"
  }
  else {
    Write-Host "Icarus build mode: fast softmax (default)"
  }
  $iverilogArgs += @("-f", $resolvedFileListRelative, "-o", $vvpPathRelative)
  Set-Content -Path $buildArgsLog -Value @(
    "simulator=iverilog"
    "mode=" + ($(if ($FullSoftmax) { "full_softmax" } else { "fast_softmax" }))
    "command=iverilog " + ($iverilogArgs -join " ")
    "quant_base=" + $fixtureBaseResolved
  )
  $compileExit = Invoke-NativeLogged `
    -FilePath "iverilog" `
    -ArgumentList $iverilogArgs `
    -LogPath $compileLog `
    -WorkingDirectory $repoRoot
  if ($compileExit -ne 0) {
    throw "iverilog failed. See $compileLog"
  }

  Write-Host "Running tb_runtime_decoder_datapath with Icarus from repo root"
  $runExit = Invoke-NativeLogged `
    -FilePath "vvp" `
    -ArgumentList @($vvpPathRelative, "+QUANT_BASE=$fixtureBaseResolved") `
    -LogPath $runLog `
    -WorkingDirectory $repoRoot
  if ($runExit -ne 0) {
    throw "vvp failed. See $runLog"
  }

  if (-not (Select-String -Path $runLog -Pattern "PASS: tb_runtime_decoder_datapath" -Quiet)) {
    throw "PASS string missing from $runLog"
  }

  Write-Host "PASS confirmed. Logs:"
  Write-Host "  $compileLog"
  Write-Host "  $runLog"
  return
}

$xvlogBat = Join-Path $VivadoBin "xvlog.bat"
$xelabBat = Join-Path $VivadoBin "xelab.bat"
$xsimBat = Join-Path $VivadoBin "xsim.bat"
$libraryName = "rddp_$stamp"
$snapshotName = "tb_runtime_decoder_datapath_$stamp"
$xvlogLog = Join-Path $logRoot "xvlog.log"
$xelabLog = Join-Path $logRoot "xelab.log"
$xsimLog = Join-Path $logRoot "xsim.log"
$snapshotDir = Join-Path $logRoot "xsim.dir\\$snapshotName"
$resolvedFiles = Get-Content $resolvedFileList | ForEach-Object {
  if ([System.IO.Path]::IsPathRooted($_)) {
    $_
  }
  else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $_))
  }
}

foreach ($toolPath in @($xvlogBat, $xelabBat, $xsimBat)) {
  if (-not (Test-Path $toolPath)) {
    throw "Missing Vivado simulator tool: $toolPath"
  }
}

Write-Host "Compiling tb_runtime_decoder_datapath with XSIM in $logRootRelative"
$xvlogArgs = @("--nolog", "--sv")
if ($FullSoftmax) {
  $xvlogArgs += @("--define", "NO_FAST_SOFTMAX")
  Write-Host "XSIM build mode: full softmax (NO_FAST_SOFTMAX defined)"
}
else {
  Write-Host "XSIM build mode: fast softmax (default)"
}
$xvlogArgs += @("-work", $libraryName)
$xvlogArgs += $resolvedFiles
Set-Content -Path $buildArgsLog -Value @(
  "simulator=xsim"
  "mode=" + ($(if ($FullSoftmax) { "full_softmax" } else { "fast_softmax" }))
  "command=xvlog " + ($xvlogArgs -join " ")
  "library=" + $libraryName
  "snapshot=" + $snapshotName
  "quant_base=" + $fixtureBaseResolved
)
$xvlogExit = Invoke-NativeLogged `
  -FilePath $xvlogBat `
  -ArgumentList $xvlogArgs `
  -LogPath $xvlogLog `
  -WorkingDirectory $logRoot
if ($xvlogExit -ne 0) {
  throw "xvlog failed. See $xvlogLog"
}

Write-Host "Elaborating tb_runtime_decoder_datapath snapshot $snapshotName"
$xelabExit = Invoke-NativeLogged `
  -FilePath $xelabBat `
  -ArgumentList @("--nolog", "--timescale", "1ns/1ps", "$libraryName.tb_runtime_decoder_datapath", "-s", $snapshotName) `
  -LogPath $xelabLog `
  -WorkingDirectory $logRoot
if (($xelabExit -ne 0) -and
    (-not (Test-Path $snapshotDir)) -and
    (-not (Select-String -Path $xelabLog -Pattern "Built simulation snapshot" -Quiet))) {
  throw "xelab failed. See $xelabLog"
}
if ($xelabExit -ne 0) {
  Write-Host "xelab returned a nonzero exit after building the snapshot. Continuing because the snapshot artifacts exist."
}

Write-Host "Running tb_runtime_decoder_datapath with XSIM in $logRootRelative"
$xsimExit = Invoke-NativeLogged `
  -FilePath $xsimBat `
  -ArgumentList @($snapshotName, "-nolog", "-xsimdir", "xsim.dir", "-runall") `
  -LogPath $xsimLog `
  -WorkingDirectory $logRoot
if ($xsimExit -ne 0) {
  throw "xsim failed. See $xsimLog"
}

if (-not (Select-String -Path $xsimLog -Pattern "PASS: tb_runtime_decoder_datapath" -Quiet)) {
  throw "PASS string missing from $xsimLog"
}

Write-Host "PASS confirmed. Logs:"
Write-Host "  $xvlogLog"
Write-Host "  $xelabLog"
Write-Host "  $xsimLog"
Write-Host "  $buildArgsLog"
