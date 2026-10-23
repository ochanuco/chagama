# Stage a release next to the running one: fetch the pinned ComfyUI tag, node
# commits and models, and build the reference analysis env. The running
# ComfyUI is left alone.
$ErrorActionPreference = "Continue"
$release = if ($env:APP_TO_DIR) { $env:APP_TO_DIR } else { Join-Path $PSScriptRoot ".." }
$scripts = Join-Path $release "scripts\worker"
. (Join-Path $scripts "common.ps1")
if (-not $EmbeddedPython -or -not (Test-Path $EmbeddedPython)) {
    "not found: ComfyUI portable root (see $Config\worker.json)"
    exit 1
}
$manifest = Join-Path $release "manifests\worker-nodes.toml"

foreach ($script in "sync-comfyui.ps1", "sync-nodes.ps1") {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts $script) `
        -Manifest $manifest -ComfyRoot $ComfyDir -Python $EmbeddedPython -FetchOnly | Out-Host
    if ($LASTEXITCODE) { "$script failed"; exit $LASTEXITCODE }
}

$env:CHAGAMA_DATA = $Data
$env:CHAGAMA_CONFIG = $Config
$src = Join-Path $release "src"
$env:PREFETCH_SRC = $src
& $EmbeddedPython -c "import os, sys; sys.path.insert(0, os.environ['PREFETCH_SRC']); from chagama.infrastructure.imaging.safety import ensure_files; print('wd tagger:', ensure_files())" 2>&1 | Out-Host
if ($LASTEXITCODE) { "wd tagger: download failed, uploads will retry it" }
& $EmbeddedPython -c "import os, sys; sys.path.insert(0, os.environ['PREFETCH_SRC']); from chagama.domain.yukari.delivery_style import MATTING_MODEL, MATTING_REVISION; from huggingface_hub import snapshot_download; print('matting model:', snapshot_download(MATTING_MODEL, revision=MATTING_REVISION))" 2>&1 | Out-Host
if ($LASTEXITCODE) { "matting model: download failed, the first deliver will retry it" }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts "reference-env.ps1") `
    -Release $release | Out-Host
if ($LASTEXITCODE) { "reference env: not built, only reference requests fail" }
exit 0
