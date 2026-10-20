# Bring ComfyUI's models/ to the pinned worker-models manifest.
param([switch]$DryRun)
$ErrorActionPreference = "Continue"
$release = Join-Path $PSScriptRoot ".."
. (Join-Path $release "scripts\worker\common.ps1")
if (-not $EmbeddedPython -or -not (Test-Path $EmbeddedPython)) {
    "not found: ComfyUI portable root (see $Config\worker.json)"
    exit 1
}
$hf = Join-Path $Config "hf-token"
if (Test-Path $hf) { $env:HF_TOKEN = (Get-Content -Raw $hf).Trim() }
$civitai = Join-Path $Config "civitai-token"
if (Test-Path $civitai) { $env:CIVITAI_TOKEN = (Get-Content -Raw $civitai).Trim() }
$env:PYTHONUTF8 = "1"
$arguments = @(
    (Join-Path $release "scripts\worker_models.py"),
    "--manifest", (Join-Path $release "manifests\worker-models.toml"),
    "--comfy-root", $ComfyDir
)
if ($DryRun) { $arguments += "--dry-run" }
& $EmbeddedPython @arguments
exit $LASTEXITCODE
