# Register watch.ps1 as a per-user logon task on the GPU worker and start it.
$script = Join-Path $PSScriptRoot "watch.ps1"
# Leftover from before the rename: stop it and drop it so it cannot start a
# second worker at logon.
Stop-ScheduledTask -TaskName "comfyui-recipes-watch" -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "comfyui-recipes-watch" -Confirm:$false -ErrorAction SilentlyContinue
Stop-ScheduledTask -TaskName "chagama-watch" -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and
        $_.CommandLine -match "(chagama|comfy-recipes)\.exe.* work|[\\/]watch\.ps1" } |
    ForEach-Object { cmd /c "taskkill /PID $($_.ProcessId) /T /F >nul 2>&1" }
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
# Registering needs elevation; the runner has none, so an existing task is
# only restarted.
if (-not (Get-ScheduledTask -TaskName "chagama-watch" -ErrorAction SilentlyContinue)) {
    Register-ScheduledTask -TaskName "chagama-watch" -Action $action `
        -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
}
Start-ScheduledTask -TaskName "chagama-watch"
Get-ScheduledTask -TaskName "chagama-watch" | Select-Object TaskName, State
