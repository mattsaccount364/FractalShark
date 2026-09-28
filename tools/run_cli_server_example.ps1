param(
    [ValidateSet('Batch', 'Sequential', 'Compare')]
    [string]$Mode = 'Batch',
    [string]$OutputRoot
)

# Submit the same scenes sequentially or as one batch. Compare uses a fresh server
# for each pass so reference-orbit state from the first pass cannot affect the second.
$ErrorActionPreference = 'Stop'
$totalTime = [System.Diagnostics.Stopwatch]::StartNew()
$outputRootProvided = [bool]$OutputRoot

$cli = (Resolve-Path (Join-Path $PSScriptRoot '..\Release\FractalSharkCli.exe')).Path
# A unique endpoint allows multiple copies of this example to run without colliding.
$runName = 'cli-server-example-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID
if (-not $OutputRoot) {
    $OutputRoot = Join-Path (Split-Path $PSScriptRoot -Parent) $runName
}
$outputRoot = $OutputRoot
$batchOutput = Join-Path $outputRoot 'batch'
$sequentialOutput = Join-Path $outputRoot 'sequential'
$endpoint = "FractalSharkCli-example-$PID"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
if ($Mode -eq 'Batch' -or $Mode -eq 'Compare') {
    New-Item -ItemType Directory -Path $batchOutput -Force | Out-Null
}
if ($Mode -eq 'Sequential' -or $Mode -eq 'Compare') {
    New-Item -ItemType Directory -Path $sequentialOutput -Force | Out-Null
}

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
    [PSCustomObject]@{
        Name = '1e1105'
        X = '2.88551201093059871274071303800151400053376951368725797081040550949502145160849912266356224118501966464135846631379795058388609385964706486647996582643879478925575952218990751370176258484608485083133367487097906821440830511869909287086597785789826928561826274850445330000681644035759856900986476475356340868534799132831546215764056013357233227447467653379961482697042858229104376560594733683330116841762126291149816626308012303061322151055718776975477001606946693760651780628655609819106485874251187866964764616038084199216787523541106958789789778684139260242292353663372539803694268099057799954487430842363686430930152025258483158501945904136023561632175130396533078460888424445029573162350889097583181352611946864579590776490531359750407648947220112191756348913404966610352672800669965940059429323243984891398477690051433173236645354747635334075445218450766702300989851412169406814176762255852305038946978145919956030591296721320395854117784060672083316748778700788521899422382707620664136521027163977579950831583843911751282971487548367000941118309453976509667412199803621909583556040093508673809100966968254713802886502450262538843122e-1'
        Y = '1.22837636274455449095906129335365068997904564092657092682711609329657475248504117230298594194722893314450582686087175509222694652475593007643450421256169190800610076957439693583571672717854028516924894608054626517771395424214000184977238550994874071548877895866493498484394960925405692074041707187503454315817871339862328188800787008820556680366789006084410733162530706740241276272226746077107376726954027943679186089294724856485367627745940403901785071720399946998747637456940456364561604904824103220284211276485879969539226863099565997769862081352549027498183743229145980993325227542917205312788342014852932791437571527323480742719984655330259153974377870900671860511486568863697185201118310254336995727240122142964919084343712047436064010498241817670904966247730713854167626448917359003570878253357397029983065354473376428170667926009275287396642227784529183790066283190895591560322819425655006015634989296704784329133923932968155785152237168947325332485204646508045257927702536291589877518328412842143905369515797279684857778054057793621817152244795211173327630404929420802627390890880044196672287580528733070422302382619507072155441e-2'
        Zoom = '6.133333E1105'
        Iterations = '250000'
    }
    [PSCustomObject]@{
        Name = 'case17-deep-4e275-try2'
        X = '-1.1875606151506535214184989076671408769190525696010175451825593916126755101486203571690889006861292798616515654863086810677819715107338483538707828140607112277703242918072845109480547211794147082276643736808708389938936182329316854691918395579410414756428423318822635434445703411177543323'
        Y = '-0.30181689947157321640500040752731728406793250293933006828422623383188584105025322133596504779620546055827933701266105425726123388747534988178569342680648140454297005133874738241868752292564316201989448356580698121981660531255110355050415753856156261268681245527781807409938630471752023121'
        Zoom = '5.539483E275'
        Iterations = '600008'
    }
)

if ($Mode -eq 'Compare') {
    & $PSCommandPath -Mode Sequential -OutputRoot $outputRoot
    & $PSCommandPath -Mode Batch -OutputRoot $outputRoot
    foreach ($scene in $scenes) {
        $sequential = Join-Path $sequentialOutput "$($scene.Name).png"
        $batched = Join-Path $batchOutput "$($scene.Name).png"
        if (-not (Test-Path -LiteralPath $sequential) -or -not (Test-Path -LiteralPath $batched)) {
            throw "Missing output for $($scene.Name)"
        }
        $sequentialHash = (Get-FileHash -LiteralPath $sequential -Algorithm SHA256).Hash
        $batchHash = (Get-FileHash -LiteralPath $batched -Algorithm SHA256).Hash
        if ($sequentialHash -ne $batchHash) {
            throw "PNG contents differ for $($scene.Name)"
        }
    }
    $totalTime.Stop()
    Write-Host "All $($scenes.Count) sequential and batch PNG pairs match."
    Write-Host ('Comparison end-to-end time: {0:N1} ms' -f $totalTime.Elapsed.TotalMilliseconds)
    Write-Host "Finished PNGs are in $outputRoot"
    Get-ChildItem -LiteralPath $outputRoot -Filter '*.png' -Recurse | Select-Object FullName, Length
    return
}

# The server retains the expensive renderer state between requests. Its output is captured
# beside the PNGs so the example remains easy to inspect after it finishes.
$serverStdout = Join-Path $outputRoot 'server.stdout.txt'
$serverStderr = Join-Path $outputRoot 'server.stderr.txt'
$server = Start-Process -FilePath $cli `
    -ArgumentList @('--server', '--endpoint', $endpoint, '--width', '3840', '--height', '2160') `
    -WindowStyle Hidden `
    -RedirectStandardOutput $serverStdout `
    -RedirectStandardError $serverStderr `
    -PassThru

$serverReady = $false
try {
    # The client frame timer includes time spent waiting for the server to accept its first
    # connection, so wait for the server's flushed listening message before submitting work.
    $startupWait = [System.Diagnostics.Stopwatch]::StartNew()
    while ($startupWait.Elapsed.TotalSeconds -lt 120) {
        if ($server.HasExited) {
            throw "Server exited before it was ready. See $serverStderr"
        }
        if (Test-Path -LiteralPath $serverStdout) {
            $serverOutput = Get-Content -LiteralPath $serverStdout -Raw -ErrorAction SilentlyContinue
            if ($serverOutput -and $serverOutput.Contains('FractalSharkCli server listening on ')) {
                $serverReady = $true
                break
            }
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not $serverReady) {
        throw "Server did not become ready within 120 seconds. See $serverStderr"
    }

    if ($Mode -eq 'Sequential') {
        # Preserve the original one-image-per-connection path as a baseline.
        foreach ($scene in $scenes) {
            $output = Join-Path $sequentialOutput "$($scene.Name).png"
            Write-Host "Rendering $($scene.Name) -> $output"
            & $cli --connect --endpoint $endpoint `
                --render-algorithm GpuHDRx32PerturbedLAv2 `
                --center-x $scene.X --center-y $scene.Y --zoom $scene.Zoom `
                --iterations $scene.Iterations --width 3840 --height 2160 `
                --antialiasing 1 --out $output --quiet
            if ($LASTEXITCODE -ne 0) {
                throw "Sequential render failed with exit code $LASTEXITCODE"
            }
        }
    }

    if ($Mode -eq 'Batch') {
        $batchFile = Join-Path $outputRoot 'scenes.batch'
        $batchResults = Join-Path $outputRoot 'batch-results.txt'
        $lines = [System.Collections.Generic.List[string]]::new()
        $lines.Add('[defaults]')
        $lines.Add('render-algorithm = GpuHDRx32PerturbedLAv2')
        $lines.Add('width = 3840')
        $lines.Add('height = 2160')
        $lines.Add('antialiasing = 1')
        $lines.Add('quiet = true')
        $lines.Add('out = batch/{name}.png')
        foreach ($scene in $scenes) {
            $lines.Add('')
            $lines.Add("[image $($scene.Name)]")
            $lines.Add("center-x = $($scene.X)")
            $lines.Add("center-y = $($scene.Y)")
            $lines.Add("zoom = $($scene.Zoom)")
            $lines.Add("iterations = $($scene.Iterations)")
        }
        [System.IO.File]::WriteAllLines($batchFile, $lines.ToArray())
        & $cli --connect --endpoint $endpoint --batch $batchFile 2>&1 |
            Tee-Object -FilePath $batchResults
        if ($LASTEXITCODE -ne 0) {
            throw "Batch render failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    # Shutdown drains all outstanding PNG encoders before replying, so every output file is
    # complete when the server exits.
    if (-not $server.HasExited -and -not $serverReady) {
        Stop-Process -Id $server.Id -Force
        $server.WaitForExit()
    }
    elseif (-not $server.HasExited) {
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
if ($Mode -eq 'Batch') {
    $pattern = '^\[(?<frame>\d+)/(?<total>\d+)\] (?<name>\S+) status=(?<status>ok|failed) ' +
        'cli_image_ms=(?<cliImage>\d+)(?: overall_ms=(?<overall>\d+) ' +
        'per_pixel_ms=(?<pixel>\d+) ref_orbit_ms=(?<orbit>\d+))?'
    $timings = @(
        foreach ($line in Get-Content -LiteralPath $batchResults) {
            $match = [regex]::Match($line, $pattern)
            if ($match.Success) {
                if ([int]$match.Groups['total'].Value -ne $scenes.Count) {
                    throw "Unexpected image count in batch result: $line"
                }
                [PSCustomObject]@{
                    Frame = [int]$match.Groups['frame'].Value
                    Name = $match.Groups['name'].Value
                    Status = $match.Groups['status'].Value
                    'Overall (ms)' = if ($match.Groups['overall'].Success) {
                        [long]$match.Groups['overall'].Value
                    } else { $null }
                    'Per pixel (ms)' = if ($match.Groups['pixel'].Success) {
                        [long]$match.Groups['pixel'].Value
                    } else { $null }
                    'RefOrbit (ms)' = if ($match.Groups['orbit'].Success) {
                        [long]$match.Groups['orbit'].Value
                    } else { $null }
                    'CLI image (ms)' = [long]$match.Groups['cliImage'].Value
                }
            }
        }
    )
    if ($timings.Count -ne $scenes.Count) {
        throw "Expected $($scenes.Count) batch timing rows, found $($timings.Count). See $batchResults"
    }
    for ($index = 0; $index -lt $scenes.Count; ++$index) {
        if ($timings[$index].Frame -ne ($index + 1) -or
            $timings[$index].Name -ne $scenes[$index].Name -or
            $timings[$index].Status -ne 'ok' -or
            $null -eq $timings[$index].'Overall (ms)' -or
            $null -eq $timings[$index].'Per pixel (ms)' -or
            $null -eq $timings[$index].'RefOrbit (ms)') {
            throw "Missing or invalid timing for image $($index + 1). See $batchResults"
        }
    }
    $timingsCsv = Join-Path $outputRoot 'batch-timings.csv'
    $timings | Export-Csv -LiteralPath $timingsCsv -NoTypeInformation
    Write-Host 'Per-image timings (ms):'
    $timingTable = $timings | Format-Table Frame, Name, 'Overall (ms)', 'Per pixel (ms)',
        'RefOrbit (ms)', 'CLI image (ms)' -AutoSize |
        Out-String
    Write-Host $timingTable.TrimEnd()
    Write-Host "Saved batch results to $batchResults and $timingsCsv"
}
if (-not $outputRootProvided) {
    Get-ChildItem -LiteralPath $outputRoot -Filter '*.png' -Recurse |
        Select-Object FullName, Length
}
