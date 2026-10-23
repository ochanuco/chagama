# Action of the comfyui logon task: link the node packs, converge ComfyUI to
# the release manifest when it changed, then run ComfyUI in the foreground.
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "common.ps1")
$release = Join-Path $PSScriptRoot "..\.."
$workerData = Join-Path $Data "worker"
New-Item -ItemType Directory -Force -Path $workerData | Out-Null
Start-Transcript -Path (Join-Path $workerData "launcher.log") -Append | Out-Null

function Stop-Launcher([int]$Code) {
    Stop-Transcript | Out-Null
    exit $Code
}

if (-not $EmbeddedPython -or -not (Test-Path $EmbeddedPython)) {
    "not found: ComfyUI portable root (see $Config\worker.json)"
    Stop-Launcher 1
}

# Node packs
foreach ($pack in Get-ChildItem (Join-Path $release "comfy_nodes") -Directory) {
    $target = Join-Path $AppRoot "current\comfy_nodes\$($pack.Name)"
    $link = Join-Path $ComfyDir "custom_nodes\$($pack.Name)"
    if (Test-Path $link) {
        $item = Get-Item $link
        if ($item.LinkType -ne "Junction") {
            "$link exists and is not a junction"
            Stop-Launcher 1
        }
        if (@($item.Target)[0] -ne $target) { cmd /c "rmdir `"$link`"" }
    }
    if (-not (Test-Path $link)) {
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    }
    "node pack: $($pack.Name) -> $target"
}

# Converge
$manifest = Join-Path $release "manifests\worker-nodes.toml"
$pipSteps = "websockets>=12;onnxruntime-if-missing;pymatting-if-missing"
$sha = [System.Security.Cryptography.SHA256]::Create()
$bytes = [System.IO.File]::ReadAllBytes($manifest) + [System.Text.Encoding]::UTF8.GetBytes($pipSteps)
$stamp = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "")
$stampFile = Join-Path $workerData "comfyui-applied.txt"
$applied = if (Test-Path $stampFile) { (Get-Content -Raw $stampFile).Trim() } else { "" }

if ($applied -eq $stamp) {
    "converge: up to date"
} else {
    "converge: manifest changed, syncing"
    $sync = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File")
    foreach ($script in "sync-comfyui.ps1", "sync-nodes.ps1") {
        & powershell @sync (Join-Path $PSScriptRoot $script) `
            -Manifest $manifest -ComfyRoot $ComfyDir -Python $EmbeddedPython | Out-Host
        if ($LASTEXITCODE) { "converge: $script failed"; Stop-Launcher $LASTEXITCODE }
    }
    & $EmbeddedPython -m pip install "websockets>=12" 2>&1 | Out-Host
    if ($LASTEXITCODE) { "converge: pip websockets failed"; Stop-Launcher $LASTEXITCODE }
    & $EmbeddedPython -c "import onnxruntime" 2>$null
    if ($LASTEXITCODE) {
        & $EmbeddedPython -m pip install onnxruntime 2>&1 | Out-Host
        if ($LASTEXITCODE) { "converge: pip onnxruntime failed"; Stop-Launcher $LASTEXITCODE }
    }
    & $EmbeddedPython -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('pymatting') is None)"
    if ($LASTEXITCODE) {
        & $EmbeddedPython -m pip install pymatting 2>&1 | Out-Host
        if ($LASTEXITCODE) { "converge: pip pymatting failed"; Stop-Launcher $LASTEXITCODE }
    }
    Set-Content -Path $stampFile -Value $stamp -Encoding ASCII
    "converge: done"
}

# Run
$env:CHAGAMA_WORKER = "1"
$env:CHAGAMA_DATA = $Data
$env:CHAGAMA_CONFIG = $Config
Set-Location $PortableRoot
& $EmbeddedPython -s ComfyUI\main.py --listen --windows-standalone-build --disable-auto-launch `
    --cache-ram 6 24 --fast fp16_accumulation autotune --use-ck-attention
$code = $LASTEXITCODE
"comfyui exited: $code"
Stop-Launcher $code
