# FM4 on the Vulkan ReGPU (branch `regpu-vulkan`)

Started 21 Nov 2026. This branch builds FM4 against braaap-stututu/ReGPU, branch `vulkan`: the Vulkan-only ReGPU the
FM3 port moved to. `fm4-perf` keeps the vendored D3D12 copy in `thirdparty/ReGPU`.

## Build

- `FM4_REGPU_DIR=<ReGPU checkout>`: the ReGPU to build. The default is the vendored D3D12 copy.
- `FM4_DXC_DIR=<folder with dxc.exe, dxcompiler.dll, dxil.dll>`: a DXC with SPIR-V output (a DirectXShaderCompiler
  release, or the Vulkan SDK's). The default is `$VULKAN_SDK/Bin`.
- `thirdparty/fidelityfx` must be the FidelityFX SDK **2.4.0** or newer. 2.1.0 has no Vulkan FSR DLLs; without them
  FSR 3/4 fall back to FSR 1.
- `-mcx16` is set for the game and ReGPU. MSVC's `<atomic>` marks its functions `target("cx16")`, so without the flag
  clang-cl calls every `std::atomic` access and fence instead of inlining it.

Development build on the user's PC (`ReGPU-V\work\scripts\buildfm4vk.bat`): the worktree `ReGPU-V\work\fm4vk` is built
in `ReGPU-V\work\fm4vk-build`, with junctions to `target\`'s XenonRecomp, `out\ppc*`, ffmpeg and game data.

## Playing it (the user's PC)

`FM4 Launcher (Vulkan).cmd`, in the workspace folder beside `FM4 Launcher.lnk`, starts the same launcher with
`FM4_EXE` set to `target\build\fm4-vulkan\fm4.exe`. The launcher runs `FM4_EXE` when it is set (`Install.cs`,
`Paths.Exe`). That folder is a copy of the development build, refreshed by hand after a rebuild.

Saves and settings are the same (`Documents\fm4`). The Vulkan shader caches don't clash with D3D12's:
`.vk1.spv` and `vk-pipelines-*.bin` against `.dxbc` and `pipelines.bin`; the `.ucode` files are the same microcode.
The shortcut starts the D3D12 build as before.

## What FM4 needed on Vulkan

- `REGPU_TEXTURE_GAMMA=1`, set in `gpu_host.cpp`. Gamma-signed texture components are converted to linear on fetch,
  as the console does. The Vulkan ReGPU has this opt-in; without it the picture is washed out and too bright (grid
  mean 121 against D3D12's 92, now 90.7). `FM4_NO_TEXTURE_GAMMA=1` turns it off.
- The soft car-shadow cascade switches (`REGPU_DEPTH_AS_8888`, `REGPU_CASCADE_SLOTS`, `REGPU_CASCADE_B_LIT`), ported to
  the Vulkan ReGPU in `0e2cfdd`. They are packed into `tinfo.z` bits 16-23, in the HLSL preamble and the SPIR-V
  translator.
- API: `regpu::DecodeTexture`, `WritePng`, `CompileIdle` and `BenchmarkRecord`/`BenchmarkWriteReport` are now public
  in ReGPU (`cee03cf`). `regpu::LogCrashContext` replaces `RendererLogCrashContext`. `LogStart`/`LogSeconds` were added
  to `src/log.cpp`. `smaa=N` maps to ReGPU's SMAA post-filter.
- Still from FM4's ReGPU, not ported: `REGPU_COLOR_SURFACE_PER_BPP` (photo mode's 64-bit accumulation) and the
  other features in ReGPU's `docs/fm4-features-to-port.md`. Photo mode is untested.

## Speed (1440p 4x MSAA, RX 7900 XTX + 9800X3D)

FM3's ReGPU speed defaults are set in `gpu_host.cpp` (`FM4_NO_FM3_DEFAULTS=1` turns them off): the draw thread
(serial), asynchronous present, the vertex arena, lean fetches, fetch variants, lazy loops and direct dynamic indices.
Without the draw thread, FM4 ran every draw on the command processor.

ReGPU changes made for FM4 (all of them help every port):

- **Resolve write-backs from half-float images:**
  - F16C conversion, column spans worked out once, SSE packing, whole-word swaps (`aae02a0`).
  - Large writes in 32-row bands on the thread pool (`37c283c`).
  - FM4 writes its 512x512 road image back every frame; the old scalar path was about 40% of the draw thread.
  - `REGPU_WRITEBACK_VERIFY=1` compares every write with the scalar path byte for byte (86,016 writes, none differ).
  - `REGPU_SLOW_WRITEBACK=1` and `REGPU_SERIAL_WRITEBACK=1` give the old behaviour.

Frozen grid: 144 fps (launcher switches only) -> 171 (FM3 defaults) -> 241 (fast write-backs). The GPU is now the
limit there (4.1 ms).

AI benchmark race, measured minimized (frame times at the draw thread's frame end, `REGPU_FRAME_TIMES_SWAP`):

| | fps | 1% low | 0.1% low |
|---|---|---|---|
| FM3 defaults, old write-backs | 161.7 | 92.4 | 75.4 |
| fast write-backs | 234.8 | 138.6 | 84.2 |
| + parallel bands (4 runs) | ~238 | ~166 | ~139 |

FM4's own D3D12 build (`fm4-perf`) runs the same race at 158.6 fps (visible window, debug-server fps).

Game side: `hooks/fm4/ring_wait.cpp` (FM3's ring wait) makes the render thread sleep ~100 us between D3D GPU-wait
rounds (`sub_826EDE50`) instead of spinning. Frozen grid, live: guest 7 drops from 95% to 48% of a core and the game
from 3.9-4.4 to 3.6-3.8 cores, with fps the same. `FM4_RING_WAIT=0` spins as before. This matters on 4-core CPUs
(the Deck).

Write watch (`memory.cpp`, from FM3; `Host::watchArm`/`watchClean`/`watchSeq` in `gpu_host.cpp`): the pages of vertex
ranges, index buffers and shader loads are made read-only, and the first write to one faults and marks its 64 KB as
written. ReGPU keeps the hash of a range none of whose pages was written since, instead of hashing it again at its first
use in each frame. `NtReadFile` disarms its buffer first (the kernel's write fails on a read-only page instead of
faulting). About 30 faults a second on the grid. `FM4_NO_WRITE_WATCH=1` turns it off; `FM4_WATCH_LOG=1` logs the
faults. 720p grid, boots alternated: +3-9% against the boot after each.

ReGPU `e30f7cb`: a large vertex buffer's "is it changing" answer is now kept per thread for the frame even when it was
not hashed in it. Before, every fetch of a quiet buffer took a mutex and a map lookup (9% of the draw thread at 720p).
720p grid, live A/B: 263 -> 279 fps.

GPU side (21 Nov 2026 night; the user: "optimize the GPU side until it catches up to the CPU side"):

- The CPU ceiling: the same AI race at 720p 1x (the GPU nearly idle) runs 304 fps, 1% low 202. At 1440p 4x the GPU was
  slower than that ceiling in a quarter of the race's seconds (`work\bench\gpuvscpu.py`).
- **`REGPU_FETCH_SPECIALIZE`, now on by default** (`gpu_host.cpp`; `FM4_NO_FETCH_SPECIALIZE=1` off): each pixel shader's
  fetches compiled for the draw's fetch constants. Live in one boot: GPU 3.59 -> 3.10 ms. AI race at the same clock:
  296.4 -> 307.4 fps, 1% low 204 -> 213, 0.1% low 171 -> 182 (the CPU ceiling). It also ends the "slow boots" (the same
  grid at 3.3-3.6 ms instead of 3.1): every boot with it ran 3.11-3.18. Screenshot: 4 of 3.7 M pixels differ, by <= 7/255.
- `REGPU_VS_SPECIALIZE` (ReGPU `ad8798b`, live, **left off**): vertex shaders compiled for their draws' wrapper words
  (viewport transform, index source, points, rectangles, vertex fetch endian and buffer). Vertex shaders offline: median
  1,441 -> 712 GPU instructions. Grid, live in one boot: GPU 3.14 -> 3.01 ms, screenshots identical. But AI races on an
  idle PC, alternated on/off/off/on/on/off at the same clock: on 227, 256, 274 fps (1% low 138-158), off 305, 289, 273
  (1% low 171-206); race GPU time 3.52 against 3.26 ms. Likely: draws that shared a pipeline split into variants (a
  buffer read in place for one draw and copied for the next), so more switches, and the variants (not in the manifest)
  are built during the race. Worth another try with fewer words specialised (not the buffer) and the variants kept.

On-screen aids (as FM3 and PGR3, drawn by ReGPU's presenter onto the back buffer, never into the game's images; on by
default, set in `gpu_host.cpp`):

- the timer (`REGPU_TIMER`): the log clock to a tenth of a second, top left. `settings.ini` `timer=0` or
  `FM4_NO_TIMER=1` hides it.
- the fps panel (`REGPU_FPS_PANEL`): fps, average and p99 frame time, and a graph of the last 400 frames' times
  (green, yellow over 1.5x the average, red over 2x), top centre. `settings.ini` `fps_panel=0` hides it.
- Both are live switches (debug server `switch REGPU_TIMER 0|1`). The launcher on this branch has checkboxes for them
  (Graphics: "Fps counter and frame-time graph", "Timer"; `settings.ini` `fps_panel`, `timer`). On the user's PC it is
  published to `target\build\launcher-vulkan`, which `FM4 Launcher (Vulkan).cmd` starts. The `fm4-perf` launcher keeps the keys.

Radeon GPU Profiler capture (21 Nov 2026, 1440p 4x grid; `work\bench\rgp`):

- Taken minimized with `work\bench\rgpcapture.py` (RDTS `RadeonDeveloperPanelCLI`, draw mode: a minimized game doesn't
  present, so frame mode never triggers). Read without screenshots: `rgpchunks.py` (the .rgp's pipeline binaries),
  `rgppipes.py` (their registers, scratch, exports, sizes), `rgpscroll.ps1` / `rgpevents.ps1` (the open RGP window's
  tables through UI Automation), `rgpjoin.py`, `rgpdraws.py`, `rgpbarriers.py`.
- Draws are ~94% of the GPU work; barriers 1.7% of the frame (one 126 us drain after the road write-back copy; most
  barriers flush and invalidate every cache); clears ~1%; no decompressions or fast-clear eliminates.
- Wavefront occupancy is low most of the frame and "not limited": the GPU waits between small draws. All 8 hardware
  contexts cycle (2,794 context rolls for 6,048 draws: vertex export count, pixel input count, kill, viewport).
- A small draw right after a pipeline switch: median 7.9 us against 5.3 us without one. Export count changed or not:
  8.3 against 7.8 us.
- The spilling, register-heavy vertex shaders correlated with slow draws, but cutting them (ReGPU `45542e4`,
  `REGPU_SPIRV_FINITE_MUL`: vertex code -17%, pipelines at >= 100 SGPRs 789 -> 507) left the grid's GPU time the same
  (3.50 ms). Not the limit on the 7900 XTX; left off.
- Every colour surface is R16G16B16A16_SFLOAT (2560x2048 for the 1440p main targets).
- Barriers merged back to back and the write-back's host barrier deferred to the next fence (ReGPU `aa92b83`,
  `REGPU_BARRIER_MERGE`, `REGPU_READBACK_BARRIER_DEFER`): the same GPU time (3.52-3.62 against 3.52-3.54 ms); left off.
- Colour surface formats (`REGPU_TRACE_SURFACE_USE`): the main scene draws 804 draws a frame in 7e3 float and 828 in 7e3 as
  16-16-16-16 (64-bit on the console too); the second view 444 in 8-8-8-8. Every surface 32-bit instead
  (`REGPU_EXP_COLOR_FORMAT`, picture wrong): 3.32-3.34 ms against 3.33-3.53 for 64-bit: no gain on the 7900 XTX, so
  per-format surfaces aren't worth building for it. Try it on the Deck (little memory bandwidth).

## Testing (no window in the way)

Scripts in `ReGPU-V\work\bench`:

- `fm4boot.ps1 [-Exe] [-Log] [-User] [-Extra @{}] [-Route]`: boots minimized with the launcher's switches
  (`REGPU_EARLY_FENCES=1`, `REGPU_RESOLVE_WRITEBACK_MAX=1024`), wheel off, and `FM4_USER_DIR` set to a copy of the
  saves, so the shader caches stay apart. `-Route` drives it to the frozen grid.
- `fm4bench.ps1 -Label x [-Seconds 60] [-Extra @{}]`: an AI benchmark race. `FM4_BENCHMARK` with the race AI driving
  the player's car (`hooks/fm4/ai_player.cpp`). It works: the car moves from 6th to 2nd in 30 s. Prints the average,
  p99 and the 1% and 0.1% lows (`ReGPU tools/frametimes.py`).
- `fm4route.py`: presses into the race until a log line appears.
- `resdiff.py`: compares two folders of resolve dumps.
- Profile: `tools/perf/sample.py <pid> topN <s> <exe>`, with `LLVM_SYMBOLIZER` set to the toolchain's.
- `fm4gpu.ps1 -Label x [-Extra @{}]`: the 1440p 4x grid's GPU time for one boot, with the GPU clock.
- `expdraw.py LOG NAME v1 v2 [--switch] [--rounds N]`: sets an env variable (or, `--switch`, a live switch) to each value
  in turn on the running game and prints the GPU time per frame for each: A/B inside one boot.
- `shotab.py NAME OUTDIR`: screenshots with a live switch off and on, compared pixel by pixel.
- `shaderlab\count.py <folder>`: compiles `.spv` files for gfx1100 with the RDTS's amdllpc and counts instructions
  (`regpu_spirv_check --no-dxc --keep <folder> [--vs-spec <vte>] <.ucode files>` writes them).
- `fm4grid.ps1 -Label x [-Extra @{}]`: boots to the 720p 1x grid (CPU-bound), settles, prints the median fps and the
  GPU time per frame. The GPU time of the same frame moves between 2.26 and 3.4 ms from boot to boot, with load on the
  PC from other programs: compare boots only at the same GPU time, or alternate them.
- Always mirror the launcher's switches. Without `REGPU_RESOLVE_WRITEBACK_MAX` the GPU time was 12 ms instead of 4.7.

## Next (start here; state at the end of 22 Nov 2026)

**The race is CPU-bound now, and the limit is ReGPU's draw thread.** At 720p 1x (the CPU ceiling, 304 fps, 1% low
202) the draw thread is busy 98% of the time (median; p10 92%, p90 100%, over 60 s). The GPU reaches 307 fps at
1440p 4x, so making the draw thread faster raises race fps directly, until the GPU or the game's main thread (guest 3)
takes over. The GPU side is parked: see "GPU (parked)" below for what was measured.

What the data says about the draw thread (last caller-level profile, 720p grid, taken **before** ReGPU e30f7cb and FM4
477a996, so stale):

| Where | Share | What |
|---|---|---|
| DrawImpl self time | 17% | per-draw state handling; needs a line-level look |
| Hashing guest data | ~8% | UploadGuest full hash of small uploads 4.4%, index buffer hash 1.9%, CachedIndexList 1.7% |
| GetIndexBuffer | 4.3% | |
| GetPipeline + MakePipelineKey | ~5% | |
| Road write-back (WriteHalfRows etc.) | ~3.4% | half-float conversion and tiling, on the draw thread |
| Shader-load memcmp | 2.3% | |
| memcpy | ~4% | |
| WindowScissor, textures, shader lookups | ~1% each | |

The renderer's timers at 1440p split the draw thread's ~4 ms/frame into constants 1.13 ms (vertex fetch setup 0.38),
"submit" 1.15, pipelines 0.64, textures 0.33, indices 0.30, shaders 0.16. `REGPU_FETCH_SPECIALIZE` (default) adds
per-draw spec hashing (RegisterSpec): check its CPU cost in the new profile.

The plan, in order (agreed with the user as the proposal; nothing started):
1. **Fresh draw-thread profile of the race at 720p** (`FM2_SAMPLE_THREAD=w FM2_SAMPLE_CALLERS=1`, plus line numbers
   inside DrawImpl). Rank everything below from it. ~15 min.
2. **Draw workers** (`REGPU_DRAW_WORKERS`, existing switch): no gain in FM3, but FM3's limit was elsewhere. Here the
   draw thread is the limit, so this could be the biggest single win if FM4's draws allow it. Quick to test.
3. **What "submit" (1.15 ms/frame) is.** If it is command recording on the draw thread, a leaner recorder helps every draw.
4. **Road write-back off the draw thread** (background thread), ~3%.
5. **Targeted cuts from the profile:** remaining hashing, index and pipeline lookups, DrawImpl hot spots.

Expected: 3-5 maybe 10-20% of the draw thread; 2 is the unknown. Measure: `fm4grid.ps1` (720p grid, median fps) and
live A/B with `tools/perf/ab.py ab <SWITCH> 5 4 switch`; confirm with `fm4bench.ps1` races at 720p (alternated boots, an idle
PC). Ask the user before multi-day work (a parallel recorder, draw reordering).

### GPU (parked)

- The race at 1440p 4x reached the CPU ceiling with `REGPU_FETCH_SPECIALIZE` (307 vs 304 fps). The XTX runs at its
  board power limit there (~361 W, ~3,050 MHz, 99-100% busy; `work\bench\gpusensors.py`, logged by `fm4bench.ps1`).
  Where the GPU time goes on the 1440p 4x grid (ReGPU `REGPU_EXP_DRAW`, live):

  | mode | GPU ms |
  |---|---|
  | normal | 3.10 |
  | 1: no draw calls (state still set) | 1.16 |
  | 2: one triangle a draw | 2.35 |
  | 4: one triangle drawn twice | 2.41 |
  | 3: full vertices, 1x1 scissor | 2.14 |
  | 5: one triangle, only draws with no pipeline switch (1,736) | 1.58 |
  | 6: one triangle, only draws right after a switch (1,176) | 2.22 |

  A draw call costs ~20 ns. A draw right after a pipeline switch costs ~0.9 us against ~0.24 us: the switches are ~0.78 ms
  of the 3.1. Pixel shading is ~1 ms. Switches that change the vertex shader cost more than pixel-shader-only ones.
- Tried with no GPU gain on the XTX (all behind switches, off): `REGPU_VS_SPECIALIZE` (lost in races),
  `REGPU_SPIRV_FINITE_MUL`, `REGPU_BARRIER_MERGE` / `REGPU_READBACK_BARRIER_DEFER`, `REGPU_EXP_COLOR_FORMAT`,
  DRAW_CULL. The remaining XTX levers: draw reordering to cut pipeline switches (days, risky for blending order) or
  uniform vertex export counts (~0.1 ms). For the Deck (little bandwidth, few CUs) try `REGPU_EXP_COLOR_FORMAT`,
  `REGPU_SPIRV_FINITE_MUL`, `REGPU_BARRIER_MERGE` there.
- Bench noise: on some boots the same frame took the GPU 1.3x longer (clocks or power). Compare races at the same
  "GPU while 90%+ busy" clock that `fm4bench.ps1` prints.

### Smaller open items

- `REGPU_COPIES_1010102_BIASED=1` (only biased 10-bit copies, never the front buffer) works but gives +0.3%: left off.
- `REGPU_COPIES_1010102` breaks FM4's front buffer (a `k_2_10_10_10` resolve becomes a 10-bit image the presenter
  doesn't take). The presenter would need to accept it.
- Photo mode: `REGPU_COLOR_SURFACE_PER_BPP` is not ported.
