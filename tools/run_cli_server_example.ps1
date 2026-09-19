# Start one persistent CLI server, submit four render requests, and stop the server after
# its background PNG encoders finish. Each run keeps its images and server logs together.
$ErrorActionPreference = 'Stop'
$totalTime = [System.Diagnostics.Stopwatch]::StartNew()

$cli = (Resolve-Path (Join-Path $PSScriptRoot '..\Release\FractalSharkCli.exe')).Path
# A unique endpoint allows multiple copies of this example to run without colliding.
$runName = 'cli-server-example-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID
$outputRoot = Join-Path (Split-Path $PSScriptRoot -Parent) $runName
$endpoint = "FractalSharkCli-example-$PID"
New-Item -ItemType Directory -Path $outputRoot | Out-Null

# Coordinates stay as strings so PowerShell cannot truncate their precision.
$scenes = @(
    [PSCustomObject]@{
        Name = 'case17-deep-4e275'
        X = '-1.1875606151506535214184989076671408769190525696010175451825593916126755101486203571690889006861292798616515654863086810677819715107338483538707828140607112277703242918072845109480547211794147082276643736808708389938936182329316854691918395579410414756428423318822635434445703411177543323'
        Y = '-0.30181689947157321640500040752731728406793250293933006828422623383188584105025322133596504779620546055827933701266105425726123388747534988178569342680648140454297005133874738241868752292564316201989448356580698121981660531255110355050415753856156261268681245527781807409938630471752023121'
        Zoom = '5.539483e275'
        Iterations = '600008'
    }
    [PSCustomObject]@{
        Name = 'case21-misiurewicz-spar-5e27'
        X = '-0.17300671609209016477613828946770372458888764012137959744393547643020180353562520419771537653705885405166925649896461064949139318317341128327473274939564013861471073570747748645616285570542524292'
        Y = '1.062752280849242560235197612681136690229320617854029710218025383997229644009518733109036931633002903184739525797020239071889093827262374176800484266893692470946462136107956050977216146050440014861'
        Zoom = '5.071075e27'
        Iterations = '30000'
    }
    [PSCustomObject]@{
        Name = 'case23-period-145-nucleus-5e27'
        X = '-0.17300671609209016477613828892344937818294446303608009957806281757183195507977072742348294212708588386692297082665678755113272002744005837554630926432536299603122427863386902200227993785523168373931237'
        Y = '1.0627522808492425602351976134607669691825780821170443838668797303774200334994733831709142678496588175434975862964984603506843104167050701809784184697102877667661049049374093963121168780435528196423078'
        Zoom = '5.071075e27'
        Iterations = '30000'
    }
    [PSCustomObject]@{
        Name = 'case24-period-148-nucleus-1e28'
        X = '-0.17300671609209016477613828968086492263222461633159461350151396646757272565776747890496804435693173209396615447468249693735678404193195389654559498706732625582328961673621848801329031469756953684956742'
        Y = '1.0627522808492425602351976125118914168351481967386057092408603462880065133013660108479767697883824279792457362691209904052076797116947765222391392158185731241429722491655792590185276866057337474176758'
        Zoom = '1.771740e28'
        Iterations = '30000'
    }
)

# The server retains the expensive renderer state between requests. Its output is captured
# beside the PNGs so the example remains easy to inspect after it finishes.
$server = Start-Process -FilePath $cli `
    -ArgumentList @('--server', '--endpoint', $endpoint, '--width', '1280', '--height', '720') `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputRoot 'server.stdout.txt') `
    -RedirectStandardError (Join-Path $outputRoot 'server.stderr.txt') `
    -PassThru

try {
    # Requests are handled in order, but each client returns after the server starts its
    # background PNG encode, allowing encoding to overlap the following render.
    foreach ($scene in $scenes) {
        $output = Join-Path $outputRoot "$($scene.Name).png"
        Write-Host "Rendering $($scene.Name) -> $output"
        & $cli --connect --endpoint $endpoint `
            --render-algorithm GpuHDRx32PerturbedLAv2 `
            --center-x $scene.X --center-y $scene.Y --zoom $scene.Zoom `
            --iterations $scene.Iterations --width 1280 --height 720 `
            --antialiasing 1 --out $output --quiet
        if ($LASTEXITCODE -ne 0) {
            throw "Render failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    # Shutdown drains all outstanding PNG encoders before replying, so every output file is
    # complete when the server exits.
    if (-not $server.HasExited) {
        & $cli --connect --endpoint $endpoint --shutdown
        $shutdownExitCode = $LASTEXITCODE
        if (-not $server.WaitForExit(60000)) {
            Stop-Process -Id $server.Id -Force
            throw 'Server did not exit after shutdown.'
        }
        if ($shutdownExitCode -ne 0) {
            throw "Shutdown failed with exit code $shutdownExitCode"
        }
    }
}

# This includes script setup, server startup, every request, PNG completion, and shutdown.
$totalTime.Stop()
Write-Host ('Total end-to-end time: {0:N1} ms' -f $totalTime.Elapsed.TotalMilliseconds)
Write-Host "Finished PNGs are in $outputRoot"
Get-ChildItem -LiteralPath $outputRoot -Filter '*.png' | Select-Object Name, Length
