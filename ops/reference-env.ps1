# Build the reference analysis env for this release if it is not there yet.
$ErrorActionPreference = "Continue"
$release = Join-Path $PSScriptRoot ".."
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $release "scripts\worker\reference-env.ps1") `
    -Release $release | Out-Host
exit $LASTEXITCODE
