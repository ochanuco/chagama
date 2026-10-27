# Bring ComfyUI's custom_nodes/ to the commits pinned in a worker-nodes manifest.
#   sync-nodes.ps1 -Manifest <file> -ComfyRoot <ComfyUI dir> -Python <python.exe> [-FetchOnly]
# -FetchOnly only fetches the pinned commits into nodes already cloned, with no
# clone, no checkout and no pip. A node with `subdir` is cloned, without its LFS
# files, under custom_nodes_src\ and custom_nodes\<name> is a junction to
# that folder of it.
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$ComfyRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [switch]$FetchOnly
)
$ErrorActionPreference = "Continue"
if (-not (Test-Path $Python)) { throw "not found: $Python" }
$nodesDir = Join-Path $ComfyRoot "custom_nodes"
if (-not (Test-Path $nodesDir)) { throw "not found: $nodesDir" }

$reader = @'
import sys, tomllib
for node in tomllib.loads(open(sys.argv[1], encoding="utf-8").read()).get("node", []):
    print("\t".join([node["name"], node["repo"], node["commit"],
                     str(bool(node.get("install_requirements", True))),
                     " ".join(node.get("pip", [])), node.get("subdir", "")]))
'@
$readerPath = Join-Path $env:TEMP "worker-nodes-reader.py"
Set-Content -Path $readerPath -Value $reader -Encoding ASCII
$entries = & $Python $readerPath $Manifest
if ($LASTEXITCODE) { exit $LASTEXITCODE }

$changed = $false
$sourcesDir = Join-Path $ComfyRoot "custom_nodes_src"
foreach ($line in $entries) {
    $name, $repo, $commit, $requirements, $pip, $subdir = $line -split "`t"
    $dir = if ($subdir) { Join-Path $sourcesDir $name } else { Join-Path $nodesDir $name }
    $moved = $false
    $present = Test-Path (Join-Path $dir ".git")
    if ($FetchOnly) {
        if (-not $present) { "{0} missing, cloned at start" -f $name; continue }
        git -C $dir cat-file -e "$commit^{commit}" 2>&1 | Out-Null
        if ($LASTEXITCODE) {
            git -C $dir fetch --quiet origin 2>&1 | Out-Host
            if ($LASTEXITCODE) { exit $LASTEXITCODE }
        }
        "{0} {1} fetched" -f $name, $commit.Substring(0, 7)
        continue
    }
    if (-not $present) {
        if ($subdir) { New-Item -ItemType Directory -Force $sourcesDir | Out-Null }
        $env:GIT_LFS_SKIP_SMUDGE = "1"
        git clone --quiet $repo $dir 2>&1 | Out-Host
        Remove-Item Env:GIT_LFS_SKIP_SMUDGE
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
        $moved = $true
    }
    $head = git -C $dir rev-parse HEAD
    if ($head -ne $commit) {
        git -C $dir fetch --quiet origin 2>&1 | Out-Host
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
        $env:GIT_LFS_SKIP_SMUDGE = "1"
        git -C $dir checkout --quiet --detach $commit 2>&1 | Out-Host
        $checkout = $LASTEXITCODE
        Remove-Item Env:GIT_LFS_SKIP_SMUDGE
        if ($checkout) { exit $checkout }
        $moved = $true
    }
    if ($subdir) {
        $target = Join-Path $dir $subdir
        if (-not (Test-Path $target)) { throw "not found: $target" }
        $link = Join-Path $nodesDir $name
        $item = Get-Item $link -ErrorAction SilentlyContinue
        if (-not $item -or $item.Target -ne $target) {
            if ($item) { cmd /c rmdir "$link" | Out-Null }
            New-Item -ItemType Junction -Path $link -Target $target | Out-Null
            $moved = $true
        }
        $dir = $target
    }
    if ($moved -and $requirements -eq "True" -and (Test-Path (Join-Path $dir "requirements.txt"))) {
        & $Python -m pip install --quiet -r (Join-Path $dir "requirements.txt") 2>&1 | Out-Host
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    if ($pip) {
        & $Python -m pip install --quiet ($pip -split " ") 2>&1 | Out-Host
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    "{0} {1} {2}" -f $name, $commit.Substring(0, 7), $(if ($moved) { "updated" } else { "ok" })
    $changed = $changed -or $moved
}
if (-not $FetchOnly) { "changed: $changed" }
