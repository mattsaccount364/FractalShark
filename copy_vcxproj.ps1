# ZipProjects.ps1

$ErrorActionPreference = "Stop"

# Output zip name
$zipName = "FractalProjects.zip"
$zipPath = Join-Path (Get-Location) $zipName

# Temporary staging directory
$stagingDir = Join-Path $env:TEMP ("FractalZip_" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $stagingDir | Out-Null

try {
    # Explicit file list
    $files = @(
        ".\FractalSaver\FractalSaver.vcxproj",
        ".\FractalShark\Fractals.vcxproj",
        ".\FractalSharkGpuLib\FractalSharkGpuLib.vcxproj",
        ".\FractalSharkLib\FractalSharkLib.vcxproj",
        ".\FractalTray\FractalTray.vcxproj",
        ".\HpSharkFloatLib\HpSharkFloatLib.vcxproj",
        ".\HpSharkFloatTest\HpSharkFloatTest.vcxproj",
        ".\HpSharkFloatTestLib\HpSharkFloatTestLib.vcxproj",
        ".\HpSharkInstantiate\HpSharkInstantiate.vcxproj",
        ".\build\*.props",
        ".\FractalShark\*.props",
        ".\.github\workflows\*.yml"
    )

    # Copy individual files
    foreach ($file in $files) {
        Copy-Item $file -Destination $stagingDir
    }

    # Copy all .props files
    Copy-Item ".\FractalShark\*.props" -Destination $stagingDir

    # Remove existing zip if present
    if (Test-Path $zipPath) {
        Remove-Item $zipPath
    }

    # Create zip
    Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath
}
finally {
    # Cleanup
    Remove-Item $stagingDir -Recurse -Force
}

Write-Host "Created $zipName"
