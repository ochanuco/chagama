# Resolve $env:REF to a commit on main, check it out and publish it as the
# step output `sha`. Expects a full clone with main checked out. REF is read
# from the environment, not from the command line, so it is never re-parsed
# by cmd; only main or a hex commit id is accepted.
$ErrorActionPreference = "Continue"
$ref = $env:REF
$sha = $null
if ($ref -eq "main" -or $ref -eq "refs/heads/main") {
    $sha = git rev-parse HEAD
} elseif ($ref -match '^[0-9a-fA-F]{7,40}$') {
    $sha = git rev-parse --verify --quiet "$ref^{commit}"
    if ($LASTEXITCODE) { $sha = $null }
    elseif (-not $sha) { $sha = $null }
    else {
        git merge-base --is-ancestor $sha HEAD
        if ($LASTEXITCODE) { $sha = $null }
    }
}
if (-not $sha) {
    Write-Output "::error::ref must be main or a commit SHA on main: $ref"
    exit 1
}
git checkout --quiet $sha
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Add-Content -Path $env:GITHUB_OUTPUT -Value "sha=$sha" -Encoding ascii
Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value "deploying ``$sha`` to production" -Encoding utf8
