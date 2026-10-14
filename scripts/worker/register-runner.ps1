# Register the GitHub Actions self-hosted runner as a per-user logon task.
#   register-runner.ps1 -Token <registration token> -Version <runner version>
param(
    [Parameter(Mandatory = $true)][string]$Token,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Root = "C:\actions-runner",
    [string]$Repo = "https://github.com/ochanuco/chagama"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $Root "config.cmd"))) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    $zip = Join-Path $Root "runner.zip"
    Invoke-WebRequest -UseBasicParsing -OutFile $zip `
        "https://github.com/actions/runner/releases/download/v$Version/actions-runner-win-x64-$Version.zip"
    Expand-Archive -Path $zip -DestinationPath $Root -Force
    Remove-Item $zip
}
Set-Location $Root
& .\config.cmd --unattended --replace --url $Repo --token $Token `
    --name $env:COMPUTERNAME --labels gpu-box --work _work
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$Root\run.cmd`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "actions-runner" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "actions-runner"
Get-ScheduledTask -TaskName "actions-runner" | Select-Object TaskName, State
