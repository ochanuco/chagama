# Register the portable ComfyUI as a per-user logon task on the GPU worker.
#   register-comfyui.ps1 -PortableRoot C:\path\to\ComfyUI_windows_portable
param([Parameter(Mandatory = $true)][string]$PortableRoot)
# The claim loop runs inside this process (comfy_nodes/yukari_worker), and a
# scheduled task inherits the user environment rather than a shell's.
[Environment]::SetEnvironmentVariable("CHAGAMA_WORKER", "1", "User")
[Environment]::SetEnvironmentVariable("COMFYUI_RECIPES_WORKER", $null, "User")
$python = Join-Path $PortableRoot "python_embeded\python.exe"
if (-not (Test-Path $python)) { throw "not found: $python" }
$arguments = @(
    "-s", "ComfyUI\main.py",
    "--listen",
    "--windows-standalone-build",
    "--disable-auto-launch",
    "--cache-ram", "6", "24",
    "--fast", "fp16_accumulation", "autotune",
    "--use-ck-attention"
) -join " "
$action = New-ScheduledTaskAction -Execute $python `
    -Argument $arguments -WorkingDirectory $PortableRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "comfyui" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "comfyui" | Select-Object TaskName, State
