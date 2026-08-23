[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',
    [switch]$FullRebuild
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$solutionPath = Join-Path $repositoryRoot 'FractalShark\FractalShark.sln'
$vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsInstallationPath = (& $vswherePath -latest -products * -requires Microsoft.Component.MSBuild -property installationPath).Trim()
$msbuildPath = Join-Path $vsInstallationPath 'MSBuild\Current\Bin\amd64\MSBuild.exe'

if (-not (Test-Path -LiteralPath $msbuildPath -PathType Leaf)) {
    throw "Visual Studio 2026 MSBuild was not found at $msbuildPath."
}

$startInfo = [Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $msbuildPath
$startInfo.WorkingDirectory = $repositoryRoot
$startInfo.UseShellExecute = $false
$startInfo.ArgumentList.Add($solutionPath)
if ($FullRebuild) {
    $startInfo.ArgumentList.Add('/t:Rebuild')
}
$startInfo.ArgumentList.Add('/m')
$startInfo.ArgumentList.Add('/nr:false')
$startInfo.ArgumentList.Add('/v:m')
$startInfo.ArgumentList.Add("/p:Configuration=$Configuration")
$startInfo.ArgumentList.Add('/p:Platform=x64')

# The agent launcher can provide both PATH and Path. Keep that malformed environment
# out of the .NET Framework VC tool tasks while preserving all other variables.
$startInfo.Environment.Clear()
foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
    $startInfo.Environment[$entry.Key] = [string]$entry.Value
}
$null = $startInfo.Environment.Remove('PATH')
$null = $startInfo.Environment.Remove('Path')

$process = [Diagnostics.Process]::Start($startInfo)
$process.WaitForExit()
exit $process.ExitCode
