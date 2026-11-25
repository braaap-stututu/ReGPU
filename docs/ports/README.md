# Ports on the Vulkan ReGPU: their records

What the game ports found while moving to the Vulkan ReGPU, kept here because most of it is about ReGPU itself. The
game-side parts (hooks, addresses, launchers) stay in each port's repository.

| File | What |
|---|---|
| [fm4-vulkan.md](fm4-vulkan.md) | FM4 (braaap-stututu/Fuzzy-Rittersport-4-ReGPU-testing, branch `regpu-vulkan`) on this ReGPU, 21-22 Nov 2026: what it needed, every speed step with its numbers, the RGP capture, the GPU experiments, and the plan for the draw thread. Copied from the port's `docs/VULKAN.md`. |
| [fm4-steam-deck.md](fm4-steam-deck.md) | The briefing for optimising FM4 on a Steam Deck (Proton, RADV/ACO): hardware, how to measure there, the switches worth trying, Linux profiling (perf, RGP through Mesa). Written for the portable FM4 test build. |

FM3 (braaap-stututu/FM3-ReGPU, `main` since 21 Nov 2026) is summarised in [../vulkan.md](../vulkan.md) ("Checked with a
game", "CPU side"); its full log is the port's `NOTES.md`, `PERF-RESULTS.md` and `docs/VRAM-UPLOAD-RING.md`.

Paths like `ReGPU-V\work\bench\...` in these records are the development machine's; the generic tools from there are
in this repository's `tools/` (RGP readers in `tools/rgp`, `tools/gpusensors.py`, `tools/resdiff.py`, `tools/deck/sensors.sh` (GPU clock, load, power and temperature on Linux),
`tools/frametimes.py`).
