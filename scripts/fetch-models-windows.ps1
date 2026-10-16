<#
.SYNOPSIS
    Put Yukari's models on a Windows ComfyUI box.

.DESCRIPTION
    The recipe in scripts/yukari_recipe.py draws stage 1 on
    silvermoonmixAnimaEvolved_v2329BTurbo, a Civitai file, and redraws
    (opt-in) through the hassaku-il-v22 diffusers folder. The other Civitai
    files are optional: two legacy IL LoRAs and a reserve stage-1 model.

    Every SHA256 below was taken from the mac's own copy and matched against
    the Hugging Face mirror, so what lands here is bit-for-bit the model the
    settled renders were made with. The script verifies, and refuses to leave
    a file in place that hashes wrong.

    Downloads resume, so a killed transfer can be restarted with the same
    command.

.PARAMETER ComfyRoot
    The ComfyUI directory -- the one holding main.py and models\. For a
    portable install this is ...\ComfyUI_windows_portable\ComfyUI

.PARAMETER CivitaiToken
    API token from https://civitai.com/user/account. Civitai requires one for
    downloads. Defaults to $env:CIVITAI_TOKEN. Without it the LoRAs are
    skipped and the manual download instructions are printed instead.

.PARAMETER SkipLoras
    Fetch only the checkpoint. yukari_recipe.py runs fine on this alone.

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

$HfRepo = 'John6666/hassaku-xl-illustrious-v22-sdxl'

# Mirror of Civitai model 140272 version 1697082 (Hassaku XL Illustrious v2.2
# by Ikena), converted to diffusers by John6666. ComfyUI's CheckpointLoader
# cannot read this layout -- DiffusersLoader can, which is why the recipe uses
# it. Hashes are only listed for the weights; the small json/txt files are
# checked by size alone.
$CheckpointFiles = @(
    @{ Path = 'model_index.json';                        Size = 721;        Sha256 = $null }
    @{ Path = 'README.md';                               Size = 595;        Sha256 = $null }
    @{ Path = 'scheduler/scheduler_config.json';         Size = 537;        Sha256 = $null }
    @{ Path = 'text_encoder/config.json';                Size = 589;        Sha256 = $null }
    @{ Path = 'text_encoder/model.safetensors';          Size = 246144152;  Sha256 = 'd66ff60453e5cb75a23df75b0ece48a9c1a5f99a5f4dd3ecfa2d454f02a557da' }
    @{ Path = 'text_encoder_2/config.json';              Size = 599;        Sha256 = $null }
    @{ Path = 'text_encoder_2/model.safetensors';        Size = 1389382176; Sha256 = 'fd855a47ec8dbdcbfeb4569cec9b52a2df7c480bf8377e1f9058f4dbaef25f7e' }
    @{ Path = 'tokenizer/merges.txt';                    Size = 573514;     Sha256 = $null }
    @{ Path = 'tokenizer/special_tokens_map.json';       Size = 496;        Sha256 = $null }
    @{ Path = 'tokenizer/tokenizer_config.json';         Size = 765;        Sha256 = $null }
    @{ Path = 'tokenizer/vocab.json';                    Size = 1109372;    Sha256 = $null }
    @{ Path = 'tokenizer_2/merges.txt';                  Size = 573514;     Sha256 = $null }
    @{ Path = 'tokenizer_2/special_tokens_map.json';     Size = 484;        Sha256 = $null }
    @{ Path = 'tokenizer_2/tokenizer_config.json';       Size = 924;        Sha256 = $null }
    @{ Path = 'tokenizer_2/vocab.json';                  Size = 1109372;    Sha256 = $null }
    @{ Path = 'unet/config.json';                        Size = 1844;       Sha256 = $null }
    @{ Path = 'unet/diffusion_pytorch_model.safetensors'; Size = 5135149760; Sha256 = '2d4c3bc34da8f79f5adf998a44c2192644f31aed3ad3fc1229a8d0b847d662ec' }
    @{ Path = 'vae/config.json';                         Size = 855;        Sha256 = $null }
    @{ Path = 'vae/diffusion_pytorch_model.safetensors'; Size = 167335342;  Sha256 = '6353737672c94b96174cb590f711eac6edf2fcce5b6e91aa9d73c5adc589ee48' }
)

# The filenames on the left are legacy optional model files.
# Civitai serves them under their own longer names, so they get renamed on the
# way in -- keep the left column exactly as it is or the graph will not build.
$LoraFiles = @(
    @{
        Name       = 'perfect-eyes-ill.safetensors'
        VersionId  = 2066663
        Size       = 228457660
        Sha256     = '97c1a083ffe6b4d45c545196eabd01c754936b996ade0c9db6d072f3bd340c55'
        CivitaiUrl = 'https://civitai.com/api/download/models/2066663'
        Title      = 'Eyes for Illustrious Lora (Perfect anime eyes) V1'
    }
    @{
        Name       = 'detailed-perfection-ill.safetensors'
        VersionId  = 1506333
        Size       = 456485044
        Sha256     = '6584fdabcf861b492da0472173fa4f37283f3c0c660072e2a12f2eb385e53bab'
        CivitaiUrl = 'https://civitai.com/api/download/models/1506333'
        Title      = 'Detailed Perfection style - Detailed Illu v0.9'
    }
    @{
        Name       = 'fnMomentAnimaTurbo_v20.safetensors'
        Dir        = 'diffusion_models'
        VersionId  = 3078189
        Size       = 4182401760
        Sha256     = 'aca019d4aa9196c8ac363a693a90f201ef68fff2708d0c0a3f4968390464b98c'
        CivitaiUrl = 'https://civitai.com/api/download/models/3078189'
        Title      = 'Fn-Moment Anima-Turbo v2.0'
    }
    @{
        Name       = 'silvermoonmixAnimaEvolved_v2329BTurbo.safetensors'
        Dir        = 'diffusion_models'
        VersionId  = 3335132
        Size       = 5843204174
        Sha256     = '84abc9577777a9d6335256d0f4745b85f7a734ea054c9e6ee093f7ca9ab289f0'
        CivitaiUrl = 'https://civitai.com/api/download/models/3335132?fileId=3221561'
        Title      = 'SilvermoonMix-Anima-Evolved v2.3_2.9B_Turbo'
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
$diffusersDir = Join-Path $modelsDir 'diffusers\hassaku-il-v22'
$lorasDir = Join-Path $modelsDir 'loras'

Write-Host "ComfyUI root : $ComfyRoot"
Write-Host "checkpoint   : $diffusersDir"
Write-Host "loras        : $lorasDir"

# ---------------------------------------------------------------------------
# The checkpoint
# ---------------------------------------------------------------------------

Write-Step "hassaku-il-v22 (diffusers, ~6.9 GB) from $HfRepo"

$fetched = 0
$skipped = 0
foreach ($file in $CheckpointFiles) {
    $target = Join-Path $diffusersDir ($file.Path -replace '/', '\')
    Write-Host "  $($file.Path)"

    if (Test-Existing -Path $target -Size $file.Size -Sha256 $file.Sha256) {
        Write-Host "    already present" -ForegroundColor DarkGray
        $skipped++
        continue
    }

    $url = "https://huggingface.co/$HfRepo/resolve/main/$($file.Path)"
    Get-RemoteFile -Url $url -Destination $target
    Assert-Hash -Path $target -Sha256 $file.Sha256 -Size $file.Size
    $fetched++
}

Write-Host ""
Write-Host "checkpoint done: $fetched fetched, $skipped already in place" -ForegroundColor Green

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
    Write-Host ""
    Write-Host "silvermoonmixAnimaEvolved_v2329BTurbo is the Yukari recipe's"
    Write-Host "stage-1 model; the rest of this list is optional."
}
else {
    Write-Step "LoRAs, Fn-Moment and SilvermoonMix (~10.3 GB) from Civitai"

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
