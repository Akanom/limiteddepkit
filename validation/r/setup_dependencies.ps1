param(
    [string]$Rscript
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"

if (-not $Rscript) {
    $onPath = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($onPath) {
        $Rscript = $onPath.Source
    } else {
        $rRoot = Join-Path $env:ProgramFiles "R"
        if (Test-Path -LiteralPath $rRoot) {
            $Rscript = Get-ChildItem -LiteralPath $rRoot -Directory |
                Sort-Object Name -Descending |
                ForEach-Object { Join-Path $_.FullName "bin\Rscript.exe" } |
                Where-Object { Test-Path -LiteralPath $_ } |
                Select-Object -First 1
        }
    }
}
if (-not $Rscript -or -not (Test-Path -LiteralPath $Rscript)) {
    throw "Rscript.exe was not found. Pass its full path with -Rscript."
}

$workDirectory = Join-Path $PSScriptRoot "work"
$downloadDirectory = Join-Path $workDirectory "downloads"
$libraryDirectory = Join-Path $workDirectory "library"
New-Item -ItemType Directory -Force -Path $downloadDirectory, $libraryDirectory | Out-Null

$packages = @(
    @{
        File = "MASS_7.3-65.zip"
        Url = "https://cran.r-project.org/bin/windows/contrib/4.5/MASS_7.3-65.zip"
        Sha256 = "46f1a3d0991c8387411b23cc9faf657a5abfc5e93438546f8b042073d9988c14"
    },
    @{
        File = "jsonlite_2.0.0.zip"
        Url = "https://cran.r-project.org/bin/windows/contrib/4.5/jsonlite_2.0.0.zip"
        Sha256 = "4b9418cff57f2357fbf5d24b1a618f082310cb9d5b63af051bd8dd7f570e188a"
    },
    @{
        File = "numDeriv_2016.8-1.1.zip"
        Url = "https://cran.r-project.org/bin/windows/contrib/4.5/numDeriv_2016.8-1.1.zip"
        Sha256 = "0df596925b695a2ba0bc327b71340921ba6550e8cbdc53e49024e41b50e2cdac"
    },
    @{
        File = "ucminf_1.2.3.zip"
        Url = "https://cran.r-project.org/bin/windows/contrib/4.5/ucminf_1.2.3.zip"
        Sha256 = "335437fae88c185ae31142e7828ba1855b45e50524a5ac0bca17175d53d673e0"
    },
    @{
        File = "ordinal_2025.12-29.zip"
        Url = "https://cran.r-project.org/bin/windows/contrib/4.5/ordinal_2025.12-29.zip"
        Sha256 = "b27a83300c6664abe0b568fab39c962c4651e62d3be95bdfb552a15550789e9b"
    },
    @{
        File = "VGAM_1.1-14.zip"
        Url = "https://cloud.r-project.org/bin/windows/contrib/4.5/VGAM_1.1-14.zip"
        Sha256 = "752dd0d4012731a0e7b37bdf4a443631850d8b0263100dae1a877afae3a61bed"
    }
)

$installExpression = 'args <- commandArgs(trailingOnly = TRUE); install.packages(args[[1]], repos = NULL, lib = args[[2]], quiet = TRUE)'

foreach ($package in $packages) {
    $archive = Join-Path $downloadDirectory $package.File
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri $package.Url -OutFile $archive -UseBasicParsing
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actualHash -ne $package.Sha256) {
        throw "SHA-256 mismatch for $($package.File): expected $($package.Sha256), got $actualHash"
    }
    & $Rscript --vanilla -e $installExpression $archive $libraryDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "R package installation failed for $($package.File)."
    }
}

$verifyExpression = 'args <- commandArgs(trailingOnly = TRUE); .libPaths(c(args[[1]], .Library)); expected <- c(MASS = "7.3.65", jsonlite = "2.0.0", numDeriv = "2016.8.1.1", ucminf = "1.2.3", ordinal = "2025.12.29", VGAM = "1.1.14", Matrix = "1.7.3", nlme = "3.1.168"); actual <- vapply(names(expected), function(package) as.character(packageVersion(package)), ""); if (!identical(unname(actual), unname(expected))) stop(paste("Unexpected package versions:", paste(names(actual), actual, collapse = ", "))); local_packages <- c("MASS", "jsonlite", "numDeriv", "ucminf", "ordinal", "VGAM"); local_root <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE); local_paths <- vapply(local_packages, function(package) dirname(normalizePath(find.package(package), winslash = "/", mustWork = TRUE)), ""); if (any(local_paths != local_root)) stop(paste("Pinned packages did not resolve from the project library:", paste(names(local_paths)[local_paths != local_root], collapse = ", "))); system_packages <- c("Matrix", "nlme"); system_root <- normalizePath(.Library, winslash = "/", mustWork = TRUE); system_paths <- vapply(system_packages, function(package) dirname(normalizePath(find.package(package), winslash = "/", mustWork = TRUE)), ""); if (any(system_paths != system_root)) stop(paste("Recommended packages did not resolve from R .Library:", paste(names(system_paths)[system_paths != system_root], collapse = ", "))); cat(paste(names(actual), actual, sep = "=", collapse = "\n"), "\n", sep = "")'
& $Rscript --vanilla -e $verifyExpression $libraryDirectory
if ($LASTEXITCODE -ne 0) {
    throw "R dependency verification failed."
}

Write-Output "Pinned R dependencies are installed in $libraryDirectory"
