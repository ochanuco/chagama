# Build the reference analysis env at data\reference-env\<key>, <key> being
# the hash of the release's reference-env requirements; an env already marked
# ready is left alone. ComfyUI's own Python is never installed into.
param([string]$Release = (Join-Path $PSScriptRoot "..\.."))
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "common.ps1")

$dir = Join-Path $Release "scripts\worker\reference-env"
$requirements = Join-Path $dir "requirements.txt"
$standalone = Join-Path $dir "requirements-standalone.txt"
$sha = [System.Security.Cryptography.SHA256]::Create()
$bytes = [System.IO.File]::ReadAllBytes($requirements) + [System.IO.File]::ReadAllBytes($standalone)
$key = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLower().Substring(0, 12)
$envDir = Join-Path (Join-Path $Data "reference-env") $key
$python = Join-Path $envDir "Scripts\python.exe"
$ready = Join-Path $envDir "ready"
if (Test-Path $ready) { "reference env: up to date ($envDir)"; exit 0 }

$uv = $env:UV
if (-not $uv) {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { $uv = $found.Source }
}
if (-not $uv) {
    $workerJson = Join-Path $Config "worker.json"
    if (Test-Path $workerJson) { $uv = (Get-Content -Raw -Encoding UTF8 $workerJson | ConvertFrom-Json).uv }
}
if (-not $uv -or -not (Test-Path $uv)) {
    "reference env: uv not found (PATH, `$env:UV, or `"uv`" in $Config\worker.json)"
    exit 1
}

$check = "import sys, numpy, PIL, safetensors, timm, torch, transformers, yaml; from transformers import AutoModel; " +
    "assert sys.prefix != sys.base_prefix, 'not running in the venv'; " +
    "assert tuple(int(p) for p in transformers.__version__.split('.')[:2]) >= (5, 14), transformers.__version__; " +
    "assert torch.cuda.is_available(), 'torch ' + torch.__version__ + ' has no CUDA'; " +
    "print('torch', torch.__version__, 'transformers', transformers.__version__, 'timm', timm.__version__)"

function Reset-Env {
    if (Test-Path $envDir) { Remove-Item -Recurse -Force $envDir }
}

function Build-OverComfy {
    if (-not $EmbeddedPython -or -not (Test-Path $EmbeddedPython)) {
        "reference env: ComfyUI's Python not found" | Out-Host
        return $false
    }
    & $uv venv --python $EmbeddedPython --system-site-packages $envDir 2>&1 | Out-Host
    if ($LASTEXITCODE) { return $false }
    # torch and torchvision stay ComfyUI's, with its CUDA build.
    & $uv pip install --python $python -r $requirements `
        --no-install-package torch --no-install-package torchvision 2>&1 | Out-Host
    if ($LASTEXITCODE) { return $false }
    & $python -c $check 2>&1 | Out-Host
    return ($LASTEXITCODE -eq 0)
}

function Build-Standalone {
    & $uv venv --python 3.12 $envDir 2>&1 | Out-Host
    if ($LASTEXITCODE) { return $false }
    & $uv pip install --python $python -r $requirements -r $standalone `
        --extra-index-url https://download.pytorch.org/whl/cu130 `
        --index-strategy unsafe-best-match 2>&1 | Out-Host
    if ($LASTEXITCODE) { return $false }
    & $python -c $check 2>&1 | Out-Host
    return ($LASTEXITCODE -eq 0)
}

Reset-Env
$mode = "over-comfyui"
if (-not (Build-OverComfy)) {
    "reference env: a venv over ComfyUI's Python did not pass, building a standalone one"
    Reset-Env
    $mode = "standalone"
    if (-not (Build-Standalone)) {
        Reset-Env
        "reference env: failed; reference requests fail until it is built"
        exit 1
    }
}
Set-Content -Path $ready -Value $mode -Encoding ASCII
"reference env: ready ($mode, $envDir)"
exit 0
