# Reading Radeon GPU Profiler captures without screenshots

Used for FM4's 1440p capture ([../../docs/ports/fm4-vulkan.md](../../docs/ports/fm4-vulkan.md), "Radeon GPU Profiler
capture"). Windows, Python 3; the `.ps1` scripts read the open RGP window through UI Automation and only read.

Take the capture with the Radeon Developer Tool Suite's `RadeonDeveloperPanelCLI` (it runs its own router: don't start
`RadeonDeveloperServiceCLI` beside it). A minimized game presents nothing, so use the CLI's draw-count trigger.

| Script | What |
|---|---|
| `rgpchunks.py <rgp> [--extract <dir>]` | The .rgp's chunks (AMD RDF container); `--extract` writes each pipeline binary (CodeObject) as `<hash>.elf`, plus the PsoCorrelation chunk. |
| `rgppipes.py <dir>` | Per pipeline: API PSO hash, vertex (NGG) and pixel registers, scratch, export count, pixel inputs, kill, instruction counts -> `<dir>/pipelines.csv`. Needs the RDTS's LLVM tools: `RDTS_LC_BIN`, or `RDTS_DIR` = the unzipped suite. |
| `rgpread.ps1`, `rgpscroll.ps1` | The open RGP window's table or tree as TSV (`rgpscroll` reaches rows that are off screen). |
| `rgpevents.ps1` | Selects each row of Overview -> Pipelines in turn and reads its events: every draw with its pipeline and time. |
| `rgpjoin.py` | RGP's Pipelines table joined with `pipelines.csv`: GPU time by registers, scratch, size, exports. |
| `rgpdraws.py` | Draws in order with their pipeline's facts: draws right after a pipeline switch against draws that keep it. |
| `rgpbarriers.py` | The Barriers table, each barrier matched to the event after it. |
