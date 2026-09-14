[CmdletBinding()]
param(
    [string]$CliPath = (Join-Path $PSScriptRoot '..\Release\FractalSharkCli.exe'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\cli-server-example-$PID"),
    [string]$Endpoint = "FractalSharkCli-example-$PID"
)

$ErrorActionPreference = 'Stop'

$resolvedCliPath = (Resolve-Path -LiteralPath $CliPath).Path
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if (Test-Path -LiteralPath $resolvedOutputRoot) {
    $existingOutputFiles = @(Get-ChildItem -LiteralPath $resolvedOutputRoot -File -ErrorAction SilentlyContinue)
    if ($existingOutputFiles.Count -ne 0) {
        throw "Output directory is not empty; choose a new OutputRoot so existing renders are preserved: $resolvedOutputRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
}

# HeapFile.bin is relative to the process working directory. Give the server
# its own directory as well as the clients so it can coexist with the GUI.
$workingDirectory = Join-Path $resolvedOutputRoot 'server-work'
New-Item -ItemType Directory -Path $workingDirectory -Force | Out-Null

function Invoke-CliClient {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath
    )

    $clientWorkingDirectory = Join-Path `
        ([System.IO.Path]::GetTempPath()) `
        "FractalSharkCli-client-$PID-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $clientWorkingDirectory -Force | Out-Null

    Push-Location $clientWorkingDirectory
    try {
        & $resolvedCliPath @Arguments 1> $StdoutPath 2> $StderrPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        Remove-Item -LiteralPath (Join-Path $clientWorkingDirectory 'HeapFile.bin') `
            -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $clientWorkingDirectory -Force -ErrorAction SilentlyContinue
    }

    $stdout = [System.IO.File]::ReadAllText($StdoutPath)
    $stderr = [System.IO.File]::ReadAllText($StderrPath)

    return [PSCustomObject]@{
        ExitCode = $exitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

try {
    Add-Type -AssemblyName System.Drawing
}
catch {
    throw "The example needs System.Drawing to verify that PNG output is not a single color: $($_.Exception.Message)"
}

function Get-DistinctPngSampleCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $bitmap = [System.Drawing.Bitmap]::new($Path)
    try {
        $colors = [System.Collections.Generic.HashSet[int]]::new()
        $stepX = [Math]::Max(1, [int]($bitmap.Width / 32))
        $stepY = [Math]::Max(1, [int]($bitmap.Height / 18))
        for ($y = 0; $y -lt $bitmap.Height; $y += $stepY) {
            for ($x = 0; $x -lt $bitmap.Width; $x += $stepX) {
                [void]$colors.Add($bitmap.GetPixel($x, $y).ToArgb())
            }
        }
        return $colors.Count
    }
    finally {
        $bitmap.Dispose()
    }
}

$serverStdoutPath = Join-Path $resolvedOutputRoot 'server.stdout.txt'
$serverStderrPath = Join-Path $resolvedOutputRoot 'server.stderr.txt'
$serverArguments = @(
    '--server',
    '--endpoint', $Endpoint,
    '--width', '1280',
    '--height', '720'
)

$server = $null
$shutdownResult = $null

# Keep the literals as strings: converting these coordinates to PowerShell
# numeric values would discard the precision that makes these scenes useful.
$scenes = @(
    [PSCustomObject]@{
        Name = 'case17-deep-4e275'
        CenterX = '-1.1875606151506535214184989076671408769190525696010175451825593916126755101486203571690889006861292798616515654863086810677819715107338483538707828140607112277703242918072845109480547211794147082276643736808708389938936182329316854691918395579410414756428423318822635434445703411177543323'
        CenterY = '-0.30181689947157321640500040752731728406793250293933006828422623383188584105025322133596504779620546055827933701266105425726123388747534988178569342680648140454297005133874738241868752292564316201989448356580698121981660531255110355050415753856156261268681245527781807409938630471752023121'
        Zoom = '5.539483e275'
        Iterations = '600008'
    }
    [PSCustomObject]@{
        Name = 'case21-misiurewicz-spar-5e27'
        CenterX = '-0.17300671609209016477613828946770372458888764012137959744393547643020180353562520419771537653705885405166925649896461064949139318317341128327473274939564013861471073570747748645616285570542524292'
        CenterY = '1.062752280849242560235197612681136690229320617854029710218025383997229644009518733109036931633002903184739525797020239071889093827262374176800484266893692470946462136107956050977216146050440014861'
        Zoom = '5.071075e27'
        Iterations = '30000'
    }
    [PSCustomObject]@{
        Name = 'case23-period-145-nucleus-5e27'
        CenterX = '-0.17300671609209016477613828892344937818294446303608009957806281757183195507977072742348294212708588386692297082665678755113272002744005837554630926432536299603122427863386902200227993785523168373931237'
        CenterY = '1.0627522808492425602351976134607669691825780821170443838668797303774200334994733831709142678496588175434975862964984603506843104167050701809784184697102877667661049049374093963121168780435528196423078'
        Zoom = '5.071075e27'
        Iterations = '30000'
    }
    [PSCustomObject]@{
        Name = 'case24-period-148-nucleus-1e28'
        CenterX = '-0.17300671609209016477613828968086492263222461633159461350151396646757272565776747890496804435693173209396615447468249693735678404193195389654559498706732625582328961673621848801329031469756953684956742'
        CenterY = '1.0627522808492425602351976125118914168351481967386057092408603462880065133013660108479767697883824279792457362691209904052076797116947765222391392158185731241429722491655792590185276866057337474176758'
        Zoom = '1.771740e28'
        Iterations = '30000'
    }
)

try {
    $server = Start-Process -FilePath $resolvedCliPath `
        -ArgumentList $serverArguments `
        -WorkingDirectory $workingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $serverStdoutPath `
        -RedirectStandardError $serverStderrPath `
        -PassThru

    $ready = $false
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        if ($server.HasExited) {
            throw "FractalSharkCli server exited before it became ready (exit $($server.ExitCode)). See $serverStderrPath"
        }
        if (Test-Path -LiteralPath $serverStdoutPath) {
            $serverOutput = Get-Content -LiteralPath $serverStdoutPath -Raw
            if ($serverOutput -match 'server listening on') {
                $ready = $true
                break
            }
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not $ready) {
        throw "Timed out waiting for the FractalSharkCli server. See $serverStdoutPath and $serverStderrPath"
    }

    foreach ($scene in $scenes) {
        $outputPath = Join-Path $resolvedOutputRoot "$($scene.Name).png"
        $stdoutPath = Join-Path $resolvedOutputRoot "$($scene.Name).stdout.txt"
        $stderrPath = Join-Path $resolvedOutputRoot "$($scene.Name).stderr.txt"
        $clientArguments = @(
            '--connect',
            '--endpoint', $Endpoint,
            '--render-algorithm', 'GpuHDRx32PerturbedLAv2',
            '--center-x', $scene.CenterX,
            '--center-y', $scene.CenterY,
            '--zoom', $scene.Zoom,
            '--iterations', $scene.Iterations,
            '--width', '1280',
            '--height', '720',
            '--antialiasing', '1',
            '--out', $outputPath,
            '--quiet'
        )

        Write-Host "Rendering $($scene.Name) -> $outputPath"
        $result = Invoke-CliClient -Arguments $clientArguments -StdoutPath $stdoutPath -StderrPath $stderrPath
        if ($result.ExitCode -ne 0) {
            throw "Client failed for $($scene.Name) with exit code $($result.ExitCode). See $stdoutPath and $stderrPath"
        }
        if ($result.Stdout -notmatch 'Frame time: [0-9]+(?:\.[0-9]+)? ms') {
            throw "Client did not report a frame time. See $stdoutPath"
        }
        if (-not [string]::IsNullOrWhiteSpace($result.Stdout)) {
            Write-Host ($result.Stdout.TrimEnd())
        }
    }

    Write-Host 'Queued all four renders; waiting for shutdown to finish PNG output.'
}
finally {
    if ($null -ne $server -and -not $server.HasExited) {
        try {
            $shutdownStdoutPath = Join-Path $resolvedOutputRoot 'shutdown.stdout.txt'
            $shutdownStderrPath = Join-Path $resolvedOutputRoot 'shutdown.stderr.txt'
            $shutdownArguments = @('--connect', '--endpoint', $Endpoint, '--shutdown')
            $shutdownResult = Invoke-CliClient `
                -Arguments $shutdownArguments `
                -StdoutPath $shutdownStdoutPath `
                -StderrPath $shutdownStderrPath
        }
        catch {
            Write-Warning "Could not send server shutdown request: $($_.Exception.Message)"
        }

        try {
            [void]$server.WaitForExit(60000)
        }
        catch {
        }
        if (-not $server.HasExited) {
            Write-Warning "Server did not exit after shutdown; stopping process $($server.Id)"
            Stop-Process -Id $server.Id -Force
        }
    }
}

if ($null -eq $shutdownResult -or $shutdownResult.ExitCode -ne 0) {
    throw "Server shutdown failed. See $resolvedOutputRoot\shutdown.stderr.txt"
}

foreach ($scene in $scenes) {
    $outputPath = Join-Path $resolvedOutputRoot "$($scene.Name).png"
    if (-not (Test-Path -LiteralPath $outputPath)) {
        throw "FractalSharkCli did not finish $outputPath before shutdown returned."
    }

    $distinctSampleCount = Get-DistinctPngSampleCount -Path $outputPath
    Write-Host "$($scene.Name) contains $distinctSampleCount distinct sampled pixel colors"
    if ($distinctSampleCount -le 1) {
        throw "$($scene.Name) produced a genuinely flat PNG: $outputPath"
    }
}

Write-Host "Four distinct PNGs and per-request logs are in $resolvedOutputRoot"
