# Register the logon task that runs run-comfyui.ps1 from the current release.
# Run by hand, elevated.
$ErrorActionPreference = "Stop"
[Environment]::SetEnvironmentVariable("CHAGAMA_WORKER", $null, "User")
[Environment]::SetEnvironmentVariable("COMFYUI_RECIPES_WORKER", $null, "User")
$script = "C:\apps\chagama\current\scripts\worker\run-comfyui.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "comfyui" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "comfyui" | Select-Object TaskName, State
