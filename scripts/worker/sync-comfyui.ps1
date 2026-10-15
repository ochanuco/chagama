# Bring the ComfyUI checkout to the tag pinned in a worker-nodes manifest.
#   sync-comfyui.ps1 -Manifest <file> -ComfyRoot <ComfyUI dir> -Python <python.exe> [-FetchOnly]
# -FetchOnly only makes the tag present locally. Otherwise the checkout moves
# to the tag and its requirements are installed.
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$ComfyRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [switch]$FetchOnly
)
$ErrorActionPreference = "Continue"
if (-not (Test-Path $Python)) { throw "not found: $Python" }
if (-not (Test-Path (Join-Path $ComfyRoot ".git"))) { throw "not a git checkout: $ComfyRoot" }

$reader = @'
import sys, tomllib
print(tomllib.loads(open(sys.argv[1], encoding="utf-8").read())["comfyui"]["tag"])
'@
$readerPath = Join-Path $env:TEMP "worker-comfyui-reader.py"
Set-Content -Path $readerPath -Value $reader -Encoding ASCII
$tag = & $Python $readerPath $Manifest
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($FetchOnly) {
    git -C $ComfyRoot rev-parse -q --verify "refs/tags/$tag" 2>&1 | Out-Null
    if ($LASTEXITCODE) {
        git -C $ComfyRoot fetch --quiet --tags origin 2>&1 | Out-Host
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    "ComfyUI $tag fetched"
    exit 0
}

$current = git -C $ComfyRoot describe --tags --exact-match 2>$null
$moved = $false
if ($current -ne $tag) {
    git -C $ComfyRoot fetch --quiet --tags origin 2>&1 | Out-Host
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    git -C $ComfyRoot checkout --quiet --detach --force $tag 2>&1 | Out-Host
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $Python -m pip install --quiet -r (Join-Path $ComfyRoot "requirements.txt") 2>&1 | Out-Host
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $moved = $true
}
"ComfyUI {0} {1}" -f $tag, $(if ($moved) { "updated" } else { "ok" })
"changed: $moved"
