# Succeed once ComfyUI answers and the worker reports the expected version.
$ErrorActionPreference = "Continue"
$release = if ($env:APP_TO_DIR) { $env:APP_TO_DIR } else { Join-Path $PSScriptRoot ".." }
. (Join-Path $release "scripts\worker\common.ps1")
$expected = $env:APP_TO_VERSION
if (-not $expected) {
    $expected = (Get-Content -Raw -Encoding UTF8 (Join-Path $release "release.json") | ConvertFrom-Json).version
}
$statusFile = Join-Path $Data "worker\status.json"
$log = Join-Path $Data "worker\launcher.log"

$deadline = (Get-Date).AddMinutes(9)
while ($true) {
    $problem = $null
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:8188/system_stats
        if ($r.StatusCode -ne 200) { $problem = "http: $($r.StatusCode)" }
    } catch {
        $problem = "http: $($_.Exception.Message)"
    }
    $status = $null
    if (-not $problem) {
        if (Test-Path $statusFile) {
            try { $status = Get-Content -Raw -Encoding UTF8 $statusFile | ConvertFrom-Json }
            catch { $problem = "status.json: unreadable" }
        } else {
            $problem = "status.json: missing"
        }
    }
    if (-not $problem) {
        if ($status.version -ne $expected) {
            $problem = "status.json: version $($status.version), expected $expected"
        } elseif (-not (Get-Process -Id $status.pid -ErrorAction SilentlyContinue)) {
            $problem = "status.json: pid $($status.pid) is not running"
        }
    }
    if (-not $problem) { "ok: worker $expected is up (pid $($status.pid))"; exit 0 }
    if ((Get-Date) -ge $deadline) { break }
    Start-Sleep -Seconds 5
}

"timed out waiting for $expected; last: $problem"
if (Test-Path $statusFile) { "status.json:"; Get-Content $statusFile }
if (Test-Path $log) { "launcher.log (tail):"; Get-Content $log -Tail 40 }
exit 1
