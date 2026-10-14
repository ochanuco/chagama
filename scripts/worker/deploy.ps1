# Move the standing worker checkout to the commit <Ref> (a SHA on main), refresh
# the venv and restart the work task. Run by deploy.yml on the self-hosted runner.
# The checkout lands on the local branch <Branch>: the worker serves only the
# requests whose recipe_ref equals the branch it is checked out on.
param(
    [Parameter(Mandatory = $true)][string]$Checkout,
    [Parameter(Mandatory = $true)][string]$Ref,
    [string]$Branch = "production",
    [string]$ComfyRoot = $env:COMFYUI_PORTABLE_ROOT,
    [int]$DrainSeconds = 300
)
$ErrorActionPreference = "Stop"
Set-Location $Checkout
# Ask the worker to finish the request it is running before the checkout moves
# under it. It removes the file as its last act, so the wait ends on the
# worker's own word. The kill below stays for a worker already gone or wedged.
$drain = Join-Path $Checkout ".local\_nogit\worker\drain"
New-Item -ItemType Directory -Force -Path (Split-Path $drain) | Out-Null
Set-Content -Path $drain -Value "deploy" -Encoding ASCII
$deadline = (Get-Date).AddSeconds($DrainSeconds)
while ((Test-Path $drain) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
if (Test-Path $drain) {
    # A left-behind file would drain the next worker the moment it starts.
    Remove-Item $drain -Force -ErrorAction SilentlyContinue
    Write-Output "drain: timed out after $DrainSeconds s"
} else {
    Write-Output "drain: worker left on its own"
}

Stop-ScheduledTask -TaskName "chagama-watch" -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and
        $_.CommandLine -match "(chagama|comfy-recipes)\.exe.* work|[\\/]watch\.ps1" } |
    ForEach-Object { cmd /c "taskkill /PID $($_.ProcessId) /T /F >nul 2>&1" }

git fetch --quiet origin
git checkout --quiet -B $Branch $Ref
git log --oneline -1
$uv = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe |
    Select-Object -First 1 -ExpandProperty FullName
& $uv pip install --python .venv\Scripts\python.exe -q -e . pillow numpy opencv-python scipy pytest onnxruntime "websockets>=12"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
# Fetched here so the first upload after a deploy doesn't stall on a 445 MiB download.
& .venv\Scripts\python.exe -c "from chagama.infrastructure.imaging.safety import ensure_files; print('wd tagger:', ensure_files())"
if ($LASTEXITCODE) { "wd tagger: download failed, uploads will retry it" }
if ($ComfyRoot) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\worker\register-nodes.ps1 `
        -Checkout $Checkout -ComfyRoot $ComfyRoot
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $core = & powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\worker\sync-comfyui.ps1 `
        -Checkout $Checkout -ComfyRoot $ComfyRoot
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $core
    $sync = & powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\worker\sync-nodes.ps1 `
        -Checkout $Checkout -ComfyRoot $ComfyRoot
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $sync
    # The claim loop runs in ComfyUI's Python now, not the checkout's venv, so
    # what it imports beyond ComfyUI's own set has to be there. Only the gap is
    # installed: numpy, PIL, cv2 and scipy already come with the box, and
    # reinstalling them is how a working ComfyUI gets broken.
    $embedded = Join-Path (Split-Path $ComfyRoot -Parent) "python_embeded\python.exe"
    $pip = & $embedded -m pip install "websockets>=12"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    # The box may already carry onnxruntime or onnxruntime-gpu from a custom
    # node; a CPU wheel installed over the GPU one breaks both.
    & $embedded -c "import onnxruntime" 2>$null
    if ($LASTEXITCODE) {
        & $embedded -m pip install onnxruntime
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    # YukariMatting's foreground estimator; transformers already comes with the box.
    # pip and the hub client write warnings and progress to stderr, which
    # "Stop" turns into a terminating error, so exit codes decide here.
    $ErrorActionPreference = "Continue"
    & $embedded -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('pymatting') is None)"
    if ($LASTEXITCODE) {
        & $embedded -m pip install pymatting 2>&1 | Out-Host
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    # Fetched here so the first deliver after a deploy doesn't stall on the download.
    & $embedded -c "import sys; sys.path.insert(0, 'src'); from chagama.domain.yukari.delivery_style import MATTING_MODEL, MATTING_REVISION; from huggingface_hub import snapshot_download; print('matting model:', snapshot_download(MATTING_MODEL, revision=MATTING_REVISION))" 2>$null
    if ($LASTEXITCODE) { "matting model: download failed, the first deliver will retry it" }
    $ErrorActionPreference = "Stop"
    "embedded python: $(if ($pip -match '^Successfully installed') { 'installed' } else { 'ok' })"
    # Unconditional: the drain above ended the claim loop, and the loop is a
    # thread in this process, so nothing brings the worker back but this.
    & powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\worker\restart-comfyui.ps1
}

# Nothing to start: the claim loop lives in the ComfyUI process the restart
# above brought back. register-watch.ps1 stays for running it standalone.
Get-ScheduledTask -TaskName "comfyui" -ErrorAction SilentlyContinue |
    Select-Object TaskName, State
