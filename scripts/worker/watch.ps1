# Resident chimera worker for the GPU box. Registered as a logon task by
# register-watch.ps1; restarts the CLI whenever it exits.
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$log = Join-Path $repo ".local\_nogit\worker\watch.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
function Wait-ComfyUI {
    while ($true) {
        try {
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://127.0.0.1:8188/system_stats | Out-Null
            return
        } catch { Start-Sleep -Seconds 10 }
    }
}
while ($true) {
    Wait-ComfyUI
    "$(Get-Date -Format s) start" | Out-File -Append -Encoding utf8 $log
    & "$repo\.venv\Scripts\chagama.exe" work --interval 30 2>&1 |
        Out-File -Append -Encoding utf8 $log
    "$(Get-Date -Format s) exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
    Start-Sleep -Seconds 10
}
