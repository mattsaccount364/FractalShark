[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourcePatterns = @('*.cpp', '*.h', '*.cu', '*.cuh', '*.cc', '*.hh', '*.hpp')
$excludedSourcePatterns = @(
    'FractalSharkLib/LargeCoords*.h',
    'FractalSharkLib/QuadDouble/Original/*',
    'FractalSharkLib/WPngImage/*'
)
$repositoryRoot = $PSScriptRoot
$script:excludedSourcePathCount = 0

function Get-ClangFormatPath {
    $attemptedLocations = [System.Collections.Generic.List[string]]::new()

    foreach ($commandName in @('clang-format.exe', 'clang-format')) {
        $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue
        if ($command -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
            return $command.Source
        }

        $attemptedLocations.Add("PATH ($commandName)")
    }

    $vswhereCandidates = @(
        ${env:ProgramFiles(x86)}
        $env:ProgramFiles
    ) |
        Where-Object { $_ } |
        ForEach-Object { Join-Path $_ 'Microsoft Visual Studio\Installer\vswhere.exe' } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -Unique

    foreach ($vswherePath in $vswhereCandidates) {
        $visualStudioPath = (& $vswherePath -latest -products '*' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        if (-not $visualStudioPath) {
            $attemptedLocations.Add("$vswherePath (no Visual Studio VC tools installation)")
            continue
        }

        $clangFormatPath = Join-Path $visualStudioPath 'VC\Tools\Llvm\x64\bin\clang-format.exe'
        if (Test-Path -LiteralPath $clangFormatPath -PathType Leaf) {
            return $clangFormatPath
        }

        $attemptedLocations.Add($clangFormatPath)
    }

    $cudaRoots = [System.Collections.Generic.List[string]]::new()
    if ($env:CUDA_PATH) {
        $cudaRoots.Add($env:CUDA_PATH)
    }

    $cudaInstallRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $cudaInstallRoot -PathType Container) {
        Get-ChildItem -LiteralPath $cudaInstallRoot -Directory |
            Sort-Object -Property Name -Descending |
            ForEach-Object { $cudaRoots.Add($_.FullName) }
    }

    foreach ($cudaRoot in $cudaRoots | Select-Object -Unique) {
        $clangFormatPath = Join-Path $cudaRoot 'bin\clang-format.exe'
        if (Test-Path -LiteralPath $clangFormatPath -PathType Leaf) {
            return $clangFormatPath
        }

        $attemptedLocations.Add($clangFormatPath)
    }

    throw "Unable to find clang-format. Attempted: $($attemptedLocations -join '; ')"
}

function Get-TrackedSourcePaths {
    $relativePaths = & git -C $repositoryRoot ls-files -z -- $sourcePatterns
    if ($LASTEXITCODE -ne 0) {
        throw 'git ls-files failed while locating tracked C++/CUDA files.'
    }

    $sourcePaths = [System.Collections.Generic.List[string]]::new()
    $script:excludedSourcePathCount = 0

    foreach ($relativePath in $relativePaths -split [char]0) {
        if (-not $relativePath) {
            continue
        }

        $isExcluded = $false
        foreach ($excludedSourcePattern in $excludedSourcePatterns) {
            if ($relativePath -like $excludedSourcePattern) {
                $isExcluded = $true
                break
            }
        }

        if ($isExcluded) {
            $script:excludedSourcePathCount++
            continue
        }

        $sourcePaths.Add((Join-Path $repositoryRoot $relativePath))
    }

    return @($sourcePaths)
}

function Normalize-CrlfLineEndings {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $contents = [System.IO.File]::ReadAllBytes($Path)
    $normalizedContents = [System.Collections.Generic.List[byte]]::new($contents.Length)

    for ($index = 0; $index -lt $contents.Length; $index++) {
        if ($contents[$index] -eq 13) {
            if ($index + 1 -lt $contents.Length -and $contents[$index + 1] -eq 10) {
                $index++
            }

            $normalizedContents.Add(13)
            $normalizedContents.Add(10)
        } elseif ($contents[$index] -eq 10) {
            $normalizedContents.Add(13)
            $normalizedContents.Add(10)
        } else {
            $normalizedContents.Add($contents[$index])
        }
    }

    if ($contents.Length -ne $normalizedContents.Count) {
        [System.IO.File]::WriteAllBytes($Path, $normalizedContents.ToArray())
        return $true
    }

    for ($index = 0; $index -lt $contents.Length; $index++) {
        if ($contents[$index] -ne $normalizedContents[$index]) {
            [System.IO.File]::WriteAllBytes($Path, $normalizedContents.ToArray())
            return $true
        }
    }

    return $false
}

if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot '.clang-format') -PathType Leaf)) {
    throw ".clang-format was not found in $repositoryRoot."
}

$sourcePaths = Get-TrackedSourcePaths
if (-not $sourcePaths) {
    throw 'No tracked C++/CUDA files were found.'
}

$missingPaths = @($sourcePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missingPaths) {
    throw "Tracked C++/CUDA files are absent from the working tree: $($missingPaths -join '; ')"
}

$clangFormatPath = Get-ClangFormatPath
Write-Host "Using clang-format: $clangFormatPath"
Write-Host "Formatting $($sourcePaths.Count) tracked C++/CUDA files."
Write-Host "Excluded $excludedSourcePathCount upstream C++/CUDA files."

Push-Location $repositoryRoot
try {
    foreach ($sourcePath in $sourcePaths) {
        & $clangFormatPath -style=file -i -- $sourcePath
        if ($LASTEXITCODE -ne 0) {
            throw "clang-format failed for $sourcePath with exit code $LASTEXITCODE."
        }
    }
} finally {
    Pop-Location
}

$normalizedCount = @($sourcePaths | Where-Object { Normalize-CrlfLineEndings -Path $_ }).Count
Write-Host "Normalized line endings in $normalizedCount of $($sourcePaths.Count) tracked C++/CUDA files."
