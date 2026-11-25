# Reads the tables of the open Radeon GPU Profiler window through Windows UI Automation (the accessibility tree Qt
# publishes), so a pane's rows come out as text instead of screenshots. It only reads; it clicks nothing.
#   rgpread.ps1 [-Out file.tsv] [-MaxRows 5000] [-Probe]
#   -Probe: print the element types found (to see whether the tables are exposed at all).
param([string]$Out = '', [int]$MaxRows = 5000, [switch]$Probe)
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$p = Get-Process RadeonGPUProfiler -ErrorAction Stop | Select-Object -First 1
$root = [Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle)
if (-not $root) { throw 'no window' }
$all = $root.FindAll([Windows.Automation.TreeScope]::Descendants, [Windows.Automation.Condition]::TrueCondition)
if ($Probe) {
    "elements: $($all.Count)"
    $all | ForEach-Object { $_.Current.ControlType.ProgrammaticName } | Group-Object | Sort-Object Count -Descending | Select-Object -First 25 | ForEach-Object { "{0,6} {1}" -f $_.Count, $_.Name }
    "--- tables:"
    foreach ($e in $all) {
        $t = $e.Current.ControlType
        if ($t -eq [Windows.Automation.ControlType]::Table -or $t -eq [Windows.Automation.ControlType]::DataGrid -or $t -eq [Windows.Automation.ControlType]::Tree) {
            "{0} '{1}' children {2}" -f $t.ProgrammaticName, $e.Current.Name, $e.FindAll([Windows.Automation.TreeScope]::Children, [Windows.Automation.Condition]::TrueCondition).Count
        }
    }
    return
}
# Every table: its header items, then each row's cells (DataItem children, or the cells by grid position).
$lines = New-Object System.Collections.Generic.List[string]
foreach ($e in $all) {
    $t = $e.Current.ControlType
    if ($t -ne [Windows.Automation.ControlType]::Table -and $t -ne [Windows.Automation.ControlType]::DataGrid -and $t -ne [Windows.Automation.ControlType]::Tree) { continue }
    $lines.Add("### $($t.ProgrammaticName) $($e.Current.Name)")
    $kids = $e.FindAll([Windows.Automation.TreeScope]::Children, [Windows.Automation.Condition]::TrueCondition)
    $row = New-Object System.Collections.Generic.List[string]
    $n = 0
    foreach ($k in $kids) {
        $kt = $k.Current.ControlType
        if ($kt -eq [Windows.Automation.ControlType]::Header) {
            $h = $k.FindAll([Windows.Automation.TreeScope]::Children, [Windows.Automation.Condition]::TrueCondition) | ForEach-Object { $_.Current.Name }
            $lines.Add(($h -join "`t"))
            continue
        }
        $cells = $k.FindAll([Windows.Automation.TreeScope]::Children, [Windows.Automation.Condition]::TrueCondition)
        if ($cells.Count -gt 0) {
            $vals = foreach ($c in $cells) {
                $v = $c.Current.Name
                try { $vp = $c.GetCurrentPattern([Windows.Automation.ValuePattern]::Pattern); if ($vp.Current.Value) { $v = $vp.Current.Value } } catch {}
                $v
            }
            $lines.Add(($vals -join "`t"))
        } else {
            $lines.Add($k.Current.Name)
        }
        if (++$n -ge $MaxRows) { break }
    }
}
if ($Out) { $lines | Set-Content -Path $Out -Encoding utf8; "wrote $($lines.Count) lines to $Out" } else { $lines | Select-Object -First 60 }
