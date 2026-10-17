# Delete images older than -Days from ComfyUI's output and input and from the
# worker's output roots under the data directory. Run daily by the task that
# register-prune.ps1 registers.
param([int]$Days = 14)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
if (-not (Test-Path -LiteralPath $Data -PathType Container)) { throw "no data directory: $Data" }

$roots = @()
if ($ComfyDir) { $roots += (Join-Path $ComfyDir "output"), (Join-Path $ComfyDir "input") }
$roots += Get-ChildItem $Data -Directory |
    Where-Object { $_.Name -notin "models", "worker" } |
    ForEach-Object FullName

$cutoff = (Get-Date).AddDays(-$Days)
foreach ($root in $roots) {
    if (-not (Test-Path $root)) { continue }
    $old = @(Get-ChildItem $root -Recurse -File | Where-Object LastWriteTime -lt $cutoff)
    $bytes = ($old | Measure-Object Length -Sum).Sum
    $old | Remove-Item -Force
    Get-ChildItem $root -Recurse -Directory |
        Sort-Object { $_.FullName.Length } -Descending |
        Where-Object { -not (Get-ChildItem $_.FullName -Force) } |
        Remove-Item -Force
    "{0}: {1} files, {2:N2} GB" -f $root, $old.Count, ($bytes / 1GB)
}
