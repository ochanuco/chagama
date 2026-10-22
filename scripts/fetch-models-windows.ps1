<#
.SYNOPSIS
    Put the legacy optional Civitai models on a Windows ComfyUI box.

.DESCRIPTION
    Fetches the legacy optional Civitai file: a reserve stage-1 model.
    Everything the recipe needs -- silvermoon-turbo, the Anima
    text encoder and VAE -- is fetched by
    manifests/worker-models.toml (scripts/worker_models.py).

    Every SHA256 below was taken from the mac's own copy. The script verifies,
    and refuses to leave a file in place that hashes wrong.

    Downloads resume, so a killed transfer can be restarted with the same
    command.

.PARAMETER ComfyRoot
    The ComfyUI directory -- the one holding main.py and models\. For a
    portable install this is ...\ComfyUI_windows_portable\ComfyUI

.PARAMETER CivitaiToken
    API token from https://civitai.com/user/account. Civitai requires one for
    downloads. Defaults to $env:CIVITAI_TOKEN. Without it the manual download
    instructions are printed instead.

.PARAMETER SkipLoras
    Skip the Civitai files.

.EXAMPLE
    .\fetch-models-windows.ps1 -ComfyRoot C:\ComfyUI_windows_portable\ComfyUI

.EXAMPLE
    .\fetch-models-windows.ps1 -ComfyRoot C:\ComfyUI_windows_portable\ComfyUI -CivitaiToken abc123
#>

#Requires -Version 5.1

param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyRoot,

    [string]$CivitaiToken = $env:CIVITAI_TOKEN,

    [switch]$SkipLoras
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# What to fetch
# ---------------------------------------------------------------------------

# The filenames on the left are legacy optional model files.
# Civitai serves them under their own longer names, so they get renamed on the
# way in -- keep the left column exactly as it is or the graph will not build.
$LoraFiles = @(
    @{
        Name       = 'fnMomentAnimaTurbo_v20.safetensors'
        Dir        = 'diffusion_models'
        VersionId  = 3078189
        Size       = 4182401760
        Sha256     = 'aca019d4aa9196c8ac363a693a90f201ef68fff2708d0c0a3f4968390464b98c'
        CivitaiUrl = 'https://civitai.com/api/download/models/3078189'
        Title      = 'Fn-Moment Anima-Turbo v2.0'
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Test-Existing {
    <# True when the file is already there and correct. #>
    param([string]$Path, [long]$Size, [string]$Sha256)

    if (-not (Test-Path -LiteralPath $Path)) { return $false }

    $actualSize = (Get-Item -LiteralPath $Path).Length
    if ($actualSize -ne $Size) {
        Write-Host "    size mismatch ($actualSize vs $Size), refetching" -ForegroundColor Yellow
        return $false
    }

    if ($Sha256) {
        Write-Host "    verifying..." -NoNewline
        $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $Sha256) {
            Write-Host " hash mismatch, refetching" -ForegroundColor Yellow
            return $false
        }
        Write-Host " ok" -ForegroundColor Green
    }
    return $true
}

function Get-RemoteFile {
    <# Resumable download via curl.exe, which ships with Windows 10 1803+. #>
    param([string]$Url, [string]$Destination)

    $dir = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    & curl.exe --location --fail --retry 3 --retry-delay 5 `
        --continue-at - --progress-bar `
        --output $Destination $Url

    if ($LASTEXITCODE -ne 0) {
        throw "download failed (curl exit $LASTEXITCODE): $Url"
    }
}

function Assert-Hash {
    param([string]$Path, [string]$Sha256, [long]$Size)

    $actualSize = (Get-Item -LiteralPath $Path).Length
    if ($actualSize -ne $Size) {
        throw "size wrong after download: $Path ($actualSize, expected $Size)"
    }
    if (-not $Sha256) { return }

    Write-Host "    verifying..." -NoNewline
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Sha256) {
        throw "SHA256 wrong after download: $Path`n  got      $actual`n  expected $Sha256"
    }
    Write-Host " ok" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe not found. It ships with Windows 10 1803 and later; on older builds install curl or fetch the files by hand."
}

if (-not (Test-Path -LiteralPath $ComfyRoot)) {
    throw "ComfyRoot does not exist: $ComfyRoot"
}

$mainPy = Join-Path $ComfyRoot 'main.py'
if (-not (Test-Path -LiteralPath $mainPy)) {
    Write-Host "warning: no main.py under $ComfyRoot -- is this the ComfyUI directory?" -ForegroundColor Yellow
}

$modelsDir = Join-Path $ComfyRoot 'models'
$lorasDir = Join-Path $modelsDir 'loras'

Write-Host "ComfyUI root : $ComfyRoot"
Write-Host "loras        : $lorasDir"

# ---------------------------------------------------------------------------
# The LoRAs
# ---------------------------------------------------------------------------

if ($SkipLoras) {
    Write-Step "skipping LoRAs (-SkipLoras)"
}
elseif (-not $CivitaiToken) {
    Write-Step "skipping LoRAs -- no Civitai token"
    Write-Host ""
    Write-Host "Civitai needs an API token for downloads. Either rerun with"
    Write-Host "  -CivitaiToken <token>       (get one at https://civitai.com/user/account)"
    Write-Host "or download these by hand in a browser and drop them under $modelsDir"
    Write-Host "with EXACTLY these paths:"
    Write-Host ""
    foreach ($lora in $LoraFiles) {
        $dir = if ($lora.Dir) { $lora.Dir } else { 'loras' }
        Write-Host "  $dir\$($lora.Name)" -ForegroundColor White
        Write-Host "    $($lora.Title)"
        Write-Host "    https://civitai.com/models?modelVersionId=$($lora.VersionId)"
        Write-Host "    sha256 $($lora.Sha256)"
    }
}
else {
    Write-Step "Fn-Moment (~4.2 GB) from Civitai"

    foreach ($lora in $LoraFiles) {
        $dir = if ($lora.Dir) { Join-Path $modelsDir $lora.Dir } else { $lorasDir }
        $target = Join-Path $dir $lora.Name
        Write-Host "  $($lora.Name)  <- $($lora.Title)"

        if (Test-Existing -Path $target -Size $lora.Size -Sha256 $lora.Sha256) {
            Write-Host "    already present" -ForegroundColor DarkGray
            continue
        }

        $separator = if ($lora.CivitaiUrl.Contains('?')) { '&' } else { '?' }
        $url = "$($lora.CivitaiUrl)${separator}token=$CivitaiToken"
        Get-RemoteFile -Url $url -Destination $target
        Assert-Hash -Path $target -Sha256 $lora.Sha256 -Size $lora.Size
    }

    Write-Host ""
    Write-Host "loras done" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Step "next"
Write-Host "Restart ComfyUI -- it reads the model folders once at startup."
Write-Host "Then from the mac:"
Write-Host ""
Write-Host "  COMFYUI_HOST=<this machine's IP> uv run chagama generate --request request.json"
Write-Host ""
