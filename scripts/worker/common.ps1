# Shared paths for the worker scripts; dot-source it.
$AppRoot = if ($env:APP_ROOT) { $env:APP_ROOT } else { "C:\apps\chagama" }
$Data = if ($env:APP_DATA) { $env:APP_DATA } else { Join-Path $AppRoot "data" }
$Config = if ($env:APP_CONFIG) { $env:APP_CONFIG } else { "C:\ProgramData\chagama" }
$PortableRoot = $null
$ComfyDir = $null
$EmbeddedPython = $null
$workerJson = Join-Path $Config "worker.json"
if (Test-Path $workerJson) {
    $PortableRoot = (Get-Content -Raw -Encoding UTF8 $workerJson | ConvertFrom-Json).comfyui_portable_root
    $ComfyDir = Join-Path $PortableRoot "ComfyUI"
    $EmbeddedPython = Join-Path $PortableRoot "python_embeded\python.exe"
}
