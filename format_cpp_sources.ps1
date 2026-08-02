[CmdletBinding()]
param(
    [ValidateRange(1, 256)]
    [int]$ThrottleLimit = [Math]::Max(1, [Environment]::ProcessorCount)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourcePatterns = @('*.cpp', '*.h', '*.cu', '*.cuh', '*.cc', '*.hh', '*.hpp')
$lineEndingOnlyPatterns = @(
    '*.txt', '*.vcxproj', '*.props', '*.targets', '*.sln', '*.filters',
    '*.rc', '*.rc2', '*.manifest', '*.nvsettings'
)
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

function Get-TrackedLineEndingOnlyPaths {
    $relativePaths = & git -C $repositoryRoot ls-files -z -- $lineEndingOnlyPatterns
    if ($LASTEXITCODE -ne 0) {
        throw 'git ls-files failed while locating tracked metadata files.'
    }

    $paths = [System.Collections.Generic.List[string]]::new()
    foreach ($relativePath in $relativePaths -split [char]0) {
        if (-not $relativePath) {
            continue
        }

        $paths.Add((Join-Path $repositoryRoot $relativePath))
    }

    return @($paths)
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

function Format-SourcePaths {
    param(
        [Parameter(Mandatory)]
        [string]$ClangFormatPath,

        [Parameter(Mandatory)]
        [string[]]$SourcePaths,

        [Parameter(Mandatory)]
        [int]$MaximumConcurrency
    )

    $activeProcesses = [System.Collections.Generic.List[object]]::new()
    $failedPaths = [System.Collections.Generic.List[string]]::new()

    function Collect-CompletedFormats {
        param([switch]$WaitForOne)

        do {
            $completedFormats = @($activeProcesses | Where-Object { $_.Process.HasExited })
            if ($completedFormats) {
                foreach ($completedFormat in $completedFormats) {
                    if ($completedFormat.Process.ExitCode -ne 0) {
                        $failedPaths.Add("$($completedFormat.SourcePath) (exit code $($completedFormat.Process.ExitCode))")
                    }

                    $completedFormat.Process.Dispose()
                    [void]$activeProcesses.Remove($completedFormat)
                }

                return
            }

            if ($WaitForOne) {
                Start-Sleep -Milliseconds 25
            }
        } while ($WaitForOne)
    }

    foreach ($sourcePath in $SourcePaths) {
        while ($activeProcesses.Count -ge $MaximumConcurrency) {
            Collect-CompletedFormats -WaitForOne
        }

        $process = Start-Process -FilePath $ClangFormatPath `
            -ArgumentList "-style=file -i -- `"$sourcePath`"" `
            -WorkingDirectory $repositoryRoot `
            -NoNewWindow `
            -PassThru
        $activeProcesses.Add([pscustomobject]@{
                Process = $process
                SourcePath = $sourcePath
            })
    }

    while ($activeProcesses.Count -gt 0) {
        Collect-CompletedFormats -WaitForOne
    }

    if ($failedPaths) {
        throw "clang-format failed for: $($failedPaths -join '; ')"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot '.clang-format') -PathType Leaf)) {
    throw ".clang-format was not found in $repositoryRoot."
}

$sourcePaths = Get-TrackedSourcePaths
if (-not $sourcePaths) {
    throw 'No tracked C++/CUDA files were found.'
}

$lineEndingOnlyPaths = Get-TrackedLineEndingOnlyPaths
$lineEndingPaths = @($sourcePaths + $lineEndingOnlyPaths | Sort-Object -Unique)

$missingPaths = @($lineEndingPaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missingPaths) {
    throw "Tracked formatter files are absent from the working tree: $($missingPaths -join '; ')"
}

$clangFormatPath = Get-ClangFormatPath
Write-Host "Using clang-format: $clangFormatPath"
Write-Host "Formatting $($sourcePaths.Count) tracked C++/CUDA files."
Write-Host "Excluded $excludedSourcePathCount upstream C++/CUDA files."
Write-Host "Using up to $ThrottleLimit concurrent clang-format processes."

Push-Location $repositoryRoot
try {
    Format-SourcePaths -ClangFormatPath $clangFormatPath -SourcePaths $sourcePaths -MaximumConcurrency $ThrottleLimit
} finally {
    Pop-Location
}

$normalizedCount = 0
$unchangedLineEndingCount = 0
foreach ($lineEndingPath in $lineEndingPaths) {
    if (Normalize-CrlfLineEndings -Path $lineEndingPath) {
        $normalizedCount++
    } else {
        $unchangedLineEndingCount++
    }
}

Write-Host ''
Write-Host 'Formatter summary:'
Write-Host "  clang-format attempted: $($sourcePaths.Count)"
Write-Host "  clang-format exclusions: $excludedSourcePathCount"
Write-Host "  line-ending candidates: $($lineEndingPaths.Count)"
Write-Host "  line-ending files changed: $normalizedCount"
Write-Host "  line-ending files unchanged: $unchangedLineEndingCount"
