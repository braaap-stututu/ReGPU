# Reads every row of a Radeon GPU Profiler table or tree through UI Automation's grid interface (GridPattern.GetItem
# by row and column), which reaches rows that aren't on screen. Only reads; clicks and scrolls nothing.
#   rgpscroll.ps1 -Out file.tsv [-Which tree|table] [-Index 0]
#   Output: a header line (the column names), then one tab-separated line per row.
param([Parameter(Mandatory)] [string]$Out, [ValidateSet('tree', 'table')] [string]$Which = 'tree', [int]$Index = 0)
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [Windows.Automation.AutomationElement]
$p = Get-Process RadeonGPUProfiler -ErrorAction Stop | Select-Object -First 1
$root = $A::FromHandle($p.MainWindowHandle)
$type = if ($Which -eq 'tree') { [Windows.Automation.ControlType]::Tree } else { [Windows.Automation.ControlType]::Table }
$cond = New-Object Windows.Automation.PropertyCondition($A::ControlTypeProperty, $type)
$grids = $root.FindAll([Windows.Automation.TreeScope]::Descendants, $cond)
if ($grids.Count -le $Index) { throw "no $Which number $Index (found $($grids.Count))" }
$grid = $grids[$Index]
$gp = $grid.GetCurrentPattern([Windows.Automation.GridPattern]::Pattern)
$rowsN = $gp.Current.RowCount; $colsN = $gp.Current.ColumnCount
$headers = $grid.FindAll([Windows.Automation.TreeScope]::Children,
    (New-Object Windows.Automation.PropertyCondition($A::ControlTypeProperty, [Windows.Automation.ControlType]::Header))) | ForEach-Object { $_.Current.Name }
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add(($headers -join "`t"))
for ($r = 0; $r -lt $rowsN; $r++) {
    $vals = for ($c = 0; $c -lt $colsN; $c++) {
        try { ($gp.GetItem($r, $c).Current.Name -replace '\s+', ' ').Trim() } catch { '' }
    }
    $lines.Add(($vals -join "`t"))
}
$lines | Set-Content -Path $Out -Encoding utf8
"$rowsN rows x $colsN columns -> $Out"
