[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "work\real_data\source")
)

$ErrorActionPreference = "Stop"

$sources = @(
    [pscustomobject]@{
        Name = "lbw.dta"
        Url = "https://www.stata-press.com/data/r19/lbw.dta"
        Sha256 = "00204ef3586836e56e49598cd9850148aea9058090a607e5bf20e12a6b0a58ee"
    },
    [pscustomobject]@{
        Name = "tvsfpors.dta"
        Url = "https://www.stata-press.com/data/r19/tvsfpors.dta"
        Sha256 = "50197a3e7b15809ed816b2846ca9dc1a4bc6aecac06ba75f4ae0312d7ceebfc8"
    },
    [pscustomobject]@{
        Name = "nlswork.dta"
        Url = "https://www.stata-press.com/data/r19/nlswork.dta"
        Sha256 = "b77bc182ac586205d769ad847e5e7cb0063c31be2c4bbef5f1ad16b74118c86f"
    }
)

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path

foreach ($source in $sources) {
    $destination = Join-Path $resolvedOutput $source.Name
    if (Test-Path -LiteralPath $destination) {
        $existingHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existingHash -eq $source.Sha256) {
            Write-Host "Verified cached $($source.Name)"
            continue
        }
    }

    $temporary = "$destination.download"
    try {
        Write-Host "Downloading $($source.Url)"
        Invoke-WebRequest -Uri $source.Url -OutFile $temporary -TimeoutSec 120
        $downloadedHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($downloadedHash -ne $source.Sha256) {
            throw "SHA-256 mismatch for $($source.Name): expected $($source.Sha256), got $downloadedHash"
        }
        Move-Item -LiteralPath $temporary -Destination $destination -Force
        Write-Host "Saved and verified $destination"
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

Write-Host "All Stata Press source datasets are present and hash-verified in $resolvedOutput"
