# Rebuild every table of the paper from the runs in ontology_packages\ and compare
# them with results\tables\. Run from anywhere:
#     powershell -ExecutionPolicy Bypass -File analysis\run_all.ps1
#     powershell -ExecutionPolicy Bypass -File analysis\run_all.ps1 -Expected <folder>
# Tables are written to output\tables\, console logs to output\logs\.

param([string]$Expected = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$logs = Join-Path $root "output\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

$steps = @(
    "analysis\build_headline_tables.py",
    "analysis\significance_tests.py --strict --substitute S4:spider-world-1=det_full_a",
    "analysis\api_comparison.py",
    "analysis\error_analysis.py",
    "analysis\difficulty_levels.py --ea",
    "analysis\ontology_stats.py",
    "analysis\schema_size.py",
    "analysis\prompt_length.py"
)
foreach ($step in $steps) {
    $name = [IO.Path]::GetFileNameWithoutExtension(($step -split " ")[0])
    Write-Host "=== python $step"
    cmd /c "python $step > `"$logs\$name.txt`" 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "python $step failed - see output\logs\$name.txt" }
}
Write-Host ""
if ($Expected) { python analysis\compare_tables.py --expected $Expected } else { python analysis\compare_tables.py }
