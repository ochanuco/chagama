# One-time move from the standing checkout to C:\apps\chagama. Run by hand,
# elevated, with the worker stopped.
param(
    [string]$Clone = "C:\Users\chanu\source\repos\comfyui-recipes",
    [string]$PortableRoot = "C:\Users\chanu\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable",
    [string]$HfTokenFile
)
$ErrorActionPreference = "Stop"
$data = "C:\apps\chagama\data"
$config = "C:\ProgramData\chagama"

$running = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "ComfyUI[\\/]main\.py" }
if ($running) { throw "ComfyUI is running (pid $($running.ProcessId -join ', ')); stop the worker first" }

New-Item -ItemType Directory -Force -Path $data, $config | Out-Null
"created: $data"
"created: $config"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
icacls $config /inheritance:r /grant:r "*$($identity.User.Value):(OI)(CI)M" "*S-1-5-32-544:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" | Out-Null
if ($LASTEXITCODE) { throw "icacls failed" }
"acl: $config (inheritance off; $($identity.Name) Modify; Administrators, SYSTEM FullControl)"

$nogit = Join-Path $Clone ".local\_nogit"
if (Test-Path $nogit) {
    foreach ($entry in Get-ChildItem $nogit -Force) {
        $dest = Join-Path $data $entry.Name
        if (Test-Path $dest) { "skipped (exists): $dest"; continue }
        Move-Item -Path $entry.FullName -Destination $dest
        "moved: $($entry.Name)"
    }
    $drain = Join-Path $data "worker\drain"
    if (Test-Path $drain) { Remove-Item $drain -Force; "removed: $drain" }
} else {
    "not found: $nogit"
}

foreach ($name in "chimera-token", "discord-webhook") {
    $src = Join-Path $Clone ".local\$name"
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $config $name) -Force
        "copied: $name"
    } else {
        "not found: $src"
    }
}
if ($HfTokenFile) {
    Copy-Item $HfTokenFile (Join-Path $config "hf-token") -Force
    "copied: hf-token"
}

$json = @{ comfyui_portable_root = $PortableRoot } | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $config "worker.json"), $json, (New-Object System.Text.UTF8Encoding($false)))
"wrote: $config\worker.json"

& (Join-Path $PSScriptRoot "register-comfyui.ps1")
"registered: task comfyui"
& (Join-Path $PSScriptRoot "register-prune.ps1")
"registered: task chagama-prune"
