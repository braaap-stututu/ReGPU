# With RGP's Overview -> Pipelines pane open: selects each pipeline row in turn (UI Automation SelectionItemPattern) and
# reads the Events table under it (event ID, call, duration), so every draw gets its time and its pipeline.
# Only changes the row selection. Output: bucket, API PSO hash, event ID, call, duration (us), share.
#   rgpevents.ps1 -Out events-by-pipeline.tsv
param([Parameter(Mandatory)] [string]$Out)
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [Windows.Automation.AutomationElement]
$p = Get-Process RadeonGPUProfiler -ErrorAction Stop | Select-Object -First 1
$root = $A::FromHandle($p.MainWindowHandle)
$null = $root.FindAll([Windows.Automation.TreeScope]::Descendants, [Windows.Automation.Condition]::TrueCondition)
function Grid($type) {
    $cond = New-Object Windows.Automation.PropertyCondition($A::ControlTypeProperty, $type)
    foreach ($g in $root.FindAll([Windows.Automation.TreeScope]::Descendants, $cond)) {
        $h = ($g.FindAll([Windows.Automation.TreeScope]::Children,
            (New-Object Windows.Automation.PropertyCondition($A::ControlTypeProperty, [Windows.Automation.ControlType]::Header))) | ForEach-Object { $_.Current.Name }) -join '|'
        if ($type -eq [Windows.Automation.ControlType]::Tree -and $h -like 'Bucket ID|*') { return $g }
        if ($type -eq [Windows.Automation.ControlType]::Table -and $h -like 'Event ID|*') { return $g }
    }
    return $null
}
$tree = Grid ([Windows.Automation.ControlType]::Tree)
if (-not $tree) { throw 'the Pipelines tree (Bucket ID column) is not on screen' }
$tgp = $tree.GetCurrentPattern([Windows.Automation.GridPattern]::Pattern)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("bucket`tapi_hash`tevent`tcall`tduration_us`tshare")
$done = 0
for ($r = 0; $r -lt $tgp.Current.RowCount; $r++) {
    $cell = $tgp.GetItem($r, 0)
    $bucket = $cell.Current.Name.Trim()
    if ($bucket -notmatch '^\d+$') { continue } # (VS / FS sub-rows)
    $hash = $tgp.GetItem($r, 1).Current.Name.Trim()
    try { $cell.GetCurrentPattern([Windows.Automation.SelectionItemPattern]::Pattern).Select() } catch { continue }
    Start-Sleep -Milliseconds 150
    $table = Grid ([Windows.Automation.ControlType]::Table)
    if (-not $table) { continue }
    $egp = $table.GetCurrentPattern([Windows.Automation.GridPattern]::Pattern)
    for ($e = 0; $e -lt $egp.Current.RowCount; $e++) {
        $id = $egp.GetItem($e, 0).Current.Name.Trim()
        $call = $egp.GetItem($e, 1).Current.Name.Trim()
        $d = ($egp.GetItem($e, 2).Current.Name -replace '\s+', ' ').Trim()
        if ($d -match '([\d.]+)\s*\S*s\s*\[([\d.]+)%\]') { $lines.Add("$bucket`t$hash`t$id`t$call`t$($Matches[1])`t$($Matches[2])") }
        elseif ($id) { $lines.Add("$bucket`t$hash`t$id`t$call`t`t$d") }
    }
    $done++
}
$lines | Set-Content -Path $Out -Encoding utf8
"$done pipelines, $($lines.Count - 1) events -> $Out"
