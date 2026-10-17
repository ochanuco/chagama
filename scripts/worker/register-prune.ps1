# Register the daily task that runs prune-outputs.ps1 from the current release.
# Run by hand.
$ErrorActionPreference = "Stop"
$script = "C:\apps\chagama\current\scripts\worker\prune-outputs.ps1"
$log = "C:\apps\chagama\data\worker\prune.log"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"& '$script' *> '$log'`""
$trigger = New-ScheduledTaskTrigger -Daily -At 5am
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "chagama-prune" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "chagama-prune" | Select-Object TaskName, State
