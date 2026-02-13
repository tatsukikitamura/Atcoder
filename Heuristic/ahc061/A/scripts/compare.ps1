## Compare 3 solvers on all test cases
$ErrorActionPreference = "Continue"

$baseDir = "c:\Users\tatuk\Atcoder\Heuristic\ahc061\A"
$toolsDir = Join-Path $baseDir "tools"
$inputDir = Join-Path $toolsDir "in"

$solvers = @{
    "new(before)"  = Join-Path $baseDir "build\new.exe"
    "attack(after)" = Join-Path $baseDir "build\attack.exe"
}

$inputFiles = Get-ChildItem -Path $inputDir -Filter "*.txt" | Sort-Object Name | Select-Object -First 10

Write-Host "Test cases: $($inputFiles.Count)"
Write-Host "Solvers: $($solvers.Keys -join ', ')"
Write-Host ("=" * 70)

$allResults = @{}

foreach ($solverEntry in $solvers.GetEnumerator()) {
    $solverName = $solverEntry.Key
    $solverPath = $solverEntry.Value
    
    Write-Host "`nTesting: $solverName"
    $scores = @{}
    $count = 0
    
    foreach ($inputFile in $inputFiles) {
        $count++
        $caseName = $inputFile.BaseName
        Write-Host "`r  Progress: $count/$($inputFiles.Count)" -NoNewline
        
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = "cargo"
            $psi.Arguments = "run -q -r --bin tester `"$solverPath`""
            $psi.WorkingDirectory = $toolsDir
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            
            $proc = [System.Diagnostics.Process]::Start($psi)
            $inputContent = Get-Content $inputFile.FullName -Raw
            $proc.StandardInput.Write($inputContent)
            $proc.StandardInput.Close()
            
            $stderr = $proc.StandardError.ReadToEnd()
            $stdout = $proc.StandardOutput.ReadToEnd()
            $completed = $proc.WaitForExit(30000)
            
            if (-not $completed) {
                $proc.Kill()
                $scores[$caseName] = -1
            } else {
                if ($stderr -match "Score = (\d+)") {
                    $scores[$caseName] = [int]$Matches[1]
                } else {
                    $scores[$caseName] = 0
                }
            }
        } catch {
            $scores[$caseName] = 0
        }
    }
    Write-Host ""
    $allResults[$solverName] = $scores
}

# Summary
$solverNames = @("new(before)", "attack(after)")
$caseNames = $inputFiles | ForEach-Object { $_.BaseName } | Sort-Object

Write-Host "`n$("=" * 70)"
Write-Host "`n--- DETAILED PER-CASE RESULTS ---"
$header = "{0,-10}" -f "Case"
foreach ($sn in $solverNames) { $header += "  {0,14}" -f $sn }
$header += "  {0,10}" -f "Best"
Write-Host $header
Write-Host ("-" * 70)

$winCounts = @{}
foreach ($sn in $solverNames) { $winCounts[$sn] = 0 }

foreach ($case in $caseNames) {
    $line = "{0,-10}" -f $case
    $bestScore = 0
    $bestSolver = ""
    foreach ($sn in $solverNames) {
        $score = $allResults[$sn][$case]
        if ($null -eq $score) { $score = 0 }
        $line += "  {0,14}" -f $score
        if ($score -gt $bestScore) {
            $bestScore = $score
            $bestSolver = $sn
        }
    }
    # Count ties
    $winners = @()
    foreach ($sn in $solverNames) {
        $score = $allResults[$sn][$case]
        if ($null -eq $score) { $score = 0 }
        if ($score -eq $bestScore) { $winners += $sn }
    }
    foreach ($w in $winners) { $winCounts[$w]++ }
    
    if ($winners.Count -gt 1) {
        $line += "  {0,10}" -f "TIE"
    } else {
        $line += "  {0,10}" -f $bestSolver
    }
    Write-Host $line
}

Write-Host ("=" * 70)
Write-Host "`n--- SUMMARY ---"
$summaryHeader = "{0,-20} {1,12} {2,10} {3,8} {4,8} {5,6} {6,9}" -f "Solver", "Total Score", "Average", "Min", "Max", "Wins", "Timeouts"
Write-Host $summaryHeader
Write-Host ("-" * 75)

foreach ($sn in $solverNames) {
    $scoresList = @()
    $timeouts = 0
    foreach ($case in $caseNames) {
        $s = $allResults[$sn][$case]
        if ($null -eq $s) { $s = 0 }
        if ($s -eq -1) { $timeouts++ } elseif ($s -gt 0) { $scoresList += $s }
    }
    $total = ($scoresList | Measure-Object -Sum).Sum
    if ($null -eq $total) { $total = 0 }
    $avg = if ($caseNames.Count -gt 0) { [math]::Round($total / $caseNames.Count, 0) } else { 0 }
    $minS = if ($scoresList.Count -gt 0) { ($scoresList | Measure-Object -Minimum).Minimum } else { 0 }
    $maxS = if ($scoresList.Count -gt 0) { ($scoresList | Measure-Object -Maximum).Maximum } else { 0 }
    $wins = $winCounts[$sn]
    
    $line = "{0,-20} {1,12} {2,10} {3,8} {4,8} {5,6} {6,9}" -f $sn, $total, $avg, $minS, $maxS, $wins, $timeouts
    Write-Host $line
}

# Find best
$bestTotal = 0
$bestName = ""
foreach ($sn in $solverNames) {
    $scoresList = @()
    foreach ($case in $caseNames) {
        $s = $allResults[$sn][$case]
        if ($null -eq $s) { $s = 0 }
        if ($s -gt 0) { $scoresList += $s }
    }
    $total = ($scoresList | Measure-Object -Sum).Sum
    if ($null -eq $total) { $total = 0 }
    if ($total -gt $bestTotal) {
        $bestTotal = $total
        $bestName = $sn
    }
}

Write-Host "`n*** BEST SOLVER: $bestName (Total Score: $bestTotal) ***"
