# Drain the worker, then stop the comfyui task and everything it started.
param([int]$DrainSeconds = 300)
$ErrorActionPreference = "Continue"
$release = if ($env:APP_FROM_DIR) { $env:APP_FROM_DIR } else { Join-Path $PSScriptRoot ".." }
. (Join-Path $release "scripts\worker\common.ps1")

$pattern = "ComfyUI\\main\.py|run-comfyui\.ps1"
function Get-WorkerProcesses {
    @(Get-CimInstance Win32_Process |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $pattern })
}

$drain = Join-Path $Data "worker\drain"
if (-not (Get-WorkerProcesses | Where-Object { $_.CommandLine -match "ComfyUI\\main\.py" })) {
    "drain: ComfyUI is not running"
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $drain) | Out-Null
    Set-Content -Path $drain -Value "deploy" -Encoding ASCII
    $deadline = (Get-Date).AddSeconds($DrainSeconds)
    while ((Test-Path $drain) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (Test-Path $drain) {
        Remove-Item $drain -Force -ErrorAction SilentlyContinue
        "drain: timed out after $DrainSeconds s"
    } else {
        "drain: worker left on its own"
    }
}

Stop-ScheduledTask -TaskName "comfyui" -ErrorAction SilentlyContinue
"task: comfyui stopped"

$procs = Get-WorkerProcesses
foreach ($p in $procs) { cmd /c "taskkill /PID $($p.ProcessId) /T /F >nul 2>&1" }
"processes: killed $($procs.Count)"

$deadline = (Get-Date).AddSeconds(30)
while ((Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) -and
       (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    "port 8188: still listening"
} else {
    "port 8188: closed"
}
exit 0
