# FM4 on the Steam Deck: briefing for the optimisation session

You are continuing optimisation work on the FM4 recomp (Forza Motorsport 4, recompiled with XenonRecomp, graphics by
ReGPU on Vulkan). Until now all of it was done on the user's PC. This document is everything that session knew that
you need on the Deck. Read it fully before you run anything; `README.md` covers installing and starting the game, and
`docs/FM4-VULKAN.md` has the full PC history.

Written 22 Nov 2026 from FM4 `regpu-vulkan` 464782b and ReGPU `vulkan` a8abaa9 (the build in this folder).

## 1. Working rules (the user's, standing)

- **Measure, don't guess.** Every change gets an A/B with numbers. Alternate runs (A B B A ...), never one of each.
- **Report work per frame** (ms of GPU and of each busy thread per frame), not only fps. fps alone hides where time goes.
- **The Deck is the real weak-CPU test.** The PC is a 9800X3D: a 2-core mask on it is not a weak CPU. What you measure
  here is what counts for weak machines.
- **Small hitches matter.** Fix in-play hitches of any size; rank them by how noticeable they are in context.
- **Upscalers (FSR 1/3) are tested on weak GPUs, i.e. here.** On the PC's 7900 XTX they gain little.
- **GPU hangs: one and stop.** After one driver timeout / GPU hang (a "ring gfx timeout" in `dmesg`, the screen
  freezing), stop that experiment, keep it off, and ask the user before any further run of it.
- **Ask before large scope** (multi-day work, risky rewrites). The user pushes to main/working branches directly; commit
  only when the user asks.
- Keep ReGPU game-neutral: a fix for FM4 alone goes in FM4's code (`gpu_host.cpp`, hooks), not ReGPU.
- New behaviour gets a switch (`REGPU_*` / `FM4_*`) so it can be A/B tested without a rebuild.

## 2. The hardware

Check which Deck it is: `cat /sys/class/dmi/id/product_name` (`Jupiter` = LCD, `Galileo` = OLED).

- APU "Van Gogh" (LCD, 7 nm) or "Sephiroth" (OLED, 6 nm): **4 Zen 2 cores / 8 threads**, 2.4-3.5 GHz, 4 MB L3 (the PC
  has 96 MB of L3 and 8 Zen 5 cores: memory-bound CPU code will look much worse here).
- GPU: RDNA 2, **8 CUs**, 1.0-1.6 GHz (~1.6 TFLOPS FP32), one shader engine. The PC's XTX has 96 CUs.
- 16 GB LPDDR5 shared by CPU and GPU: 88 GB/s (LCD, 5500 MT/s) or 102 GB/s (OLED, 6400 MT/s). The PC has 960 GB/s.
  Memory bandwidth is the most likely GPU limit here.
- APU power 4-15 W, shared between CPU and GPU: a busy CPU takes clock away from the GPU and the other way round.
- Screen 1280x800 (16:10), 60 Hz (LCD) or 90 Hz (OLED).
- Driver: Mesa **RADV** (shader compiler ACO) under Proton's Vulkan. The PC uses AMD's Windows driver (LLPC), so the
  PC's shader instruction counts, scratch use and register numbers do **not** transfer. Re-measure everything.

## 3. Where things stand (from the PC)

- The PC's AI race runs 307 fps at 1440p 4x MSAA; the CPU ceiling (720p 1x) is 304 fps, 1% low 202. There the race is
  limited by **ReGPU's draw thread, busy 98%** (median; p10 92%, p90 100%).
- The guest's main thread (guest thread 3) also runs near 100% of a core in races (game code).
- On the Deck expect both limits to be lower and closer together: 4 slow cores for the draw thread, the replay
  (submit/present) thread, the game's main and render threads, audio and the pool. Find out first which one limits here.

### The threads (what to look for)

- **Command processor** (ReGPU `gpu.cpp`): reads the game's PM4 ring, parses, queues draws to the draw thread.
- **Draw thread** (`REGPU_DRAW_THREAD`, serial): translates each draw: state, constants, fetches, pipelines, textures,
  index buffers, records Vulkan commands. The `perf lite` line's "draw thread N% busy" is this thread.
- **Replay / present thread** (`REGPU_ASYNC_PRESENT`): submit, present, fences.
- **Guest threads** (game code, recompiled PPC): 3 = main game thread; 7 = render thread (sleeps in the ring wait,
  `FM4_RING_WAIT`); others: audio, streaming, physics.
- Pool threads: resolve write-backs in bands, shader and pipeline compiles.

### What the draw thread spends its time on (PC profile, 720p grid; taken before two fixes, so stale)

| Where | Share |
|---|---|
| DrawImpl self time (per-draw state handling) | 17% |
| Hashing guest data: UploadGuest 4.4%, index buffer hash 1.9%, CachedIndexList 1.7% | ~8% |
| GetIndexBuffer | 4.3% |
| GetPipeline + MakePipelineKey | ~5% |
| Road image write-back (WriteHalfRows etc.) | ~3.4% |
| Shader-load memcmp | 2.3% |
| memcpy | ~4% |
| WindowScissor, textures, shader lookups | ~1% each |

ReGPU's timers at 1440p split the draw thread's ~4 ms a frame into constants 1.13 ms (vertex fetch setup 0.38),
"submit" 1.15, pipelines 0.64, textures 0.33, indices 0.30, shaders 0.16. On the Deck these will be 2-4x larger.

### The GPU frame (PC, RGP capture, 1440p 4x grid)

- ~6,000 draws a frame, ~1,170 pipeline switches, 2,794 context rolls. Draws are ~94% of GPU work.
- The GPU waits between small draws: low occupancy, a draw after a pipeline switch ~0.9 us vs ~0.24 us. Pipeline
  switches ~0.78 of 3.1 ms; pixel shading ~1 ms.
- Colour surfaces are all R16G16B16A16_SFLOAT (64-bit). On the console: 804 draws/frame in 7e3 float, 828 in 7e3 as
  16-16-16-16, 444 in 8-8-8-8 (second view).
- Translated shaders are big (vertex ~1,300 instructions median, predication as selects).

On the XTX, bandwidth and shader size didn't matter. On the Deck they may: that is why the switches in section 6 exist.

## 4. Tools in this folder

| What | How |
|---|---|
| Start the game | `./play.sh` (any `REGPU_*`/`FM4_*`/`FM2_*` variable passes through) |
| Frozen-grid test | `tools/deck/grid.sh LABEL [settle] [rounds]`: boots, drives to the grid, freezes, fps rounds, perf lines, sensors -> `out/LABEL.txt`. `KEEP=1` leaves it running for live A/B |
| AI benchmark race | `tools/deck/bench.sh LABEL [seconds]`: the race AI drives, 60 s measured, frame-time stats -> `out/LABEL.txt` |
| Live A/B in one boot | `python3 tools/perf/ab.py ab REGPU_NAME 5 4 switch` (interleaved off/on rounds, fps each) |
| GPU time per value | `python3 tools/bench/expdraw.py fm2.log NAME v1 v2 --switch [--rounds N]` |
| Screenshot A/B | `python3 tools/bench/shotab.py NAME OUTDIR` (a live switch off and on, compared pixel by pixel; needs Pillow) |
| Debug server | `python3 tools/regpu/regpuctl.py status|shot|switch|switches|env|log|capture|quit` (port: `REGPU_DEBUG_PORT`; the scripts use 7351, play.sh alone uses the default 7341) |
| Frame-time stats | `python3 tools/regpu/frametimes.py out/x.csv [from to]` |
| Sensors | `tools/deck/sensors.sh [seconds] [file]`: GPU clock, busy %, APU power, temperature, top CPU clock |

The game's log is `fm2.log` in this folder (the name is historical; `out/play-fm4.log` has Proton's lines).

`perf lite:` lines (with `REGPU_PERF_LITE=1`, set by the scripts), one a second:
`318 fps, 3.14 ms/frame (p99 8.85, max 23.03) | 54 draws/frame | GPU 0.22 ms/frame | draw thread 17% busy | 1 long`.
GPU ms is measured with timestamps: compare it with the frame time to see GPU-bound vs CPU-bound.
`REGPU_PERF=1` adds detailed per-section timers (costs a little itself).

The `tools/bench` and `tools/perf` scripts were written for Windows paths but are plain Python over the debug socket;
`gpusensors.py`, `rgp*.ps1` and `sample.py` are Windows-only (`sensors.sh` replaces the first).

**None of the Deck scripts have been run yet.** Expect to fix small things (paths, Proton start time, `pkill` patterns)
on the first try. Check `fm2.log` and `out/play-fm4.log` when something stalls.

## 5. Measuring on the Deck

- **Fix the clocks for A/B work**: Quick Access menu > Performance > advanced: TDP limit and manual GPU clock (e.g. GPU
  1600 MHz, TDP 15 W). Note them in the results. Without that, the APU moves power between CPU and GPU and runs differ.
  Watch the temperature: a hot Deck throttles. Plugged in, fan audible.
- Then also measure at the defaults (no limits): that's what players get.
- Desktop Mode vs Game Mode: Game Mode runs under gamescope (the real setup); Desktop Mode is easier for scripts. Do A/B
  in one mode, and check the final result in Game Mode (Steam > Add a Non-Steam Game > play.sh; launch options
  `REGPU_PERF_LITE=1 %command%` style variables work there).
- The Deck's own performance overlay (MangoHud: Quick Access > Performance > overlay level 4) shows CPU per-core load,
  GPU load, clocks and power live. MangoHud can also log to CSV (`MANGOHUD_CONFIG=output_folder=...,log_duration=60`,
  `MANGOHUD=1`), independent of the game's own numbers.
- Resolution: 1280x800 is native (`FM2_RESOLUTION=1280x800`). `FM2_RENDER_SCALE` renders below output.
  MSAA (`FM2_MSAA=1|2|4`) is expensive in bandwidth: measure 1x and 4x. Upscalers: `FM2_UPSCALER=fsr1|fsr3`,
  `FM2_UPSCALE_QUALITY=1..4` (FSR turns MSAA and SMAA off).
- Frame cap: the scripts uncap (`FM2_MAX_FPS=1000`, vsync off) to see headroom. For players the target is a steady
  60 (or 90 on OLED) with no hitches: also measure 1% / 0.1% lows and long frames at the cap.
- First boot compiles every shader and pipeline (minutes; Fossilize/Steam's cache doesn't apply to a non-Steam exe).
  Never measure a first boot; run the grid once to warm the caches.

## 6. Switches to test (no rebuild needed)

All are environment variables; the ones marked live can also be flipped while running (`regpuctl.py switch NAME 0|1`,
`ab.py ab NAME ...`). Each gave nothing on the 7900 XTX, but the Deck is a different machine:

| Switch | What | Why it may help here |
|---|---|---|
| `REGPU_EXP_COLOR_FORMAT=b10g11r11` (or `rgba8`, `rgb10a2`) | every EDRAM colour surface 32-bit instead of 64-bit. **Experiment: the picture is wrong where alpha/range matter** | Halves colour bandwidth; tells you how much the 64-bit surfaces cost. If large, build proper per-format surfaces (the 804 + 444 draws that are 32-bit on the console) |
| `REGPU_SPIRV_FINITE_MUL=1` | vertex shaders without the Mul9 (0 x anything = 0) guard where the operands are provably finite: vertex code -17%, fewer SGPRs | ACO and 8 CUs may care about shader size and occupancy |
| `REGPU_BARRIER_MERGE=1`, `REGPU_READBACK_BARRIER_DEFER=1` | consecutive barriers merged; write-back host barrier deferred to the next fence | Fewer cache flushes on a small GPU |
| `REGPU_VS_SPECIALIZE` (live) | vertex shaders compiled per draw's wrapper words | Lost in PC races (more pipeline variants built mid-race): only try with warm caches |
| `REGPU_EXP_DRAW=N` (live, 1-8) | diagnostic: 1 no draw calls, 2 one triangle a draw, 3 full vertices + 1x1 scissor, 5/6 split by pipeline switch... | Where GPU time goes: vertex vs pixel vs fixed per-draw cost |
| `REGPU_FETCH_SPECIALIZE` (on by default, live) | pixel-shader fetches compiled per material | Check it still wins here (more pipelines; ACO compile time) |
| `REGPU_NO_ANISO=1` | no anisotropic filtering | Texture bandwidth |
| `FM2_MSAA=1`, `settings.ini` `smaa=0` | AA cost | Bandwidth |
| `FM4_TEXTURE_CHECK_EVERY_FRAME=1` | opposite of the default stagger: shows the CPU cost of texture checks | CPU |
| `FM4_RING_WAIT=0` | render thread spins instead of sleeping | Should be worse on 4 cores: confirm the default is right |
| `FM4_NO_WRITE_WATCH=1` | no page write watch (hashing instead) | Fault cost under Wine may differ from Windows: confirm the watch still wins |
| `FM4_SYSTEM_MEMORY_RING=1` | upload ring in system memory instead of "GPU upload heap" | On a shared-memory APU these may be the same: measure |

**Draw workers (the big CPU unknown).** On the PC the draw thread is the limit; spreading draws over workers may be the
biggest CPU win, or nothing (in FM3 it measured slower: per-draw cost roughly doubled across threads). FM4 sets
`REGPU_DRAW_THREAD_SERIAL=1`, and setting it to 0 does not undo it (ReGPU checks only that it is set). To test, start
with the FM3 defaults spelled out minus SERIAL, and the workers on:

```sh
FM4_NO_FM3_DEFAULTS=1 REGPU_DRAW_THREAD=1 REGPU_ASYNC_PRESENT=1 REGPU_VERTEX_ARENA=1 REGPU_LEAN_FETCH=1 \
REGPU_FETCH_VARIANTS=1 REGPU_EXP_LAZY_LOOPS=1 REGPU_DIRECT_DYNAMIC_INDICES=1 REGPU_DRAW_WORKERS=2 \
KEEP=1 tools/deck/grid.sh workers2
python3 tools/perf/ab.py ab REGPU_NO_DRAW_WORKERS 5 4 switch     # live: workers off/on in the same boot
```

Only with the upscaler off (workers are disabled under FSR). Try 2 and 3 workers; with 4 cores the game's own threads
compete. If it gains, the default can change in `gpu_host.cpp` (needs a rebuild: section 8).

## 7. Profiling on the Deck

### CPU

1. **The built-in sampler** (in fm4.exe): `FM2_SAMPLE_THREAD=w` samples the draw thread (or a guest thread id, e.g. `3`)
   every millisecond and logs the top 25 functions every 5 s as `profile:` lines in `fm2.log`. `FM2_SAMPLE_CALLERS=1`
   adds two callers, `FM2_SAMPLE_LINES=1` source lines (both live). It needs symbols from `fm4.pdb` through Wine's
   dbghelp. **Untested under Wine**: if the names come out as `?`, fall back to 2.
2. **Linux `perf`** on the Wine process (SteamOS has no `perf` by default; install it in a container or with
   `steamos-readonly disable` + pacman, ask the user first; `sudo sysctl kernel.perf_event_paranoid=1`). Samples
   inside `fm4.exe` show as addresses in the mapped PE image; subtract the load address and resolve with
   `llvm-symbolizer --obj=fm4.exe` (it reads the PDB beside it), on the Deck or on the PC.
3. `perf lite`'s "draw thread N% busy" and MangoHud's per-core load tell you which thread is the limit before you profile.

Per-thread CPU time: `/proc/<pid>/task/*/stat` (fields 14-15, utime/stime in clock ticks) over a few seconds, per
frame. Thread names: `/proc/<pid>/task/*/comm` (Wine passes thread descriptions through).

### GPU

- **RADV's RGP capture**: `MESA_VK_TRACE=rgp MESA_VK_TRACE_TRIGGER=/tmp/rgp-trigger ./play.sh`, then
  `touch /tmp/rgp-trigger` during the frame you want; the `.rgp` lands in `/tmp`. Open it with Radeon GPU Profiler for
  Linux (Radeon Developer Tool Suite), or copy it to the PC where `work\bench\rgp*.py` read it (pipelines, registers,
  barriers, per-draw timings) without the GUI. Needs a presenting window (it triggers at present).
- **Shader statistics from ACO**: `RADV_DEBUG=shaderstats` (or `shaders` for the disassembly) prints per-pipeline
  instruction counts, VGPR/SGPR, spills, waves to stderr (`out/play-fm4.log`). Compare the switches' effect on the
  compiler that actually runs here.
- `REGPU_SHADER_STATS=1`: ReGPU's own per-pipeline sizes in the log.
- `REGPU_EXP_DRAW` modes (section 6): GPU time split without any profiler.
- Validation: `REGPU_VK_VALIDATION=2` (needs the Vulkan validation layers in the prefix; slow).

## 8. Changing code

This build cannot be rebuilt on the Deck as it is. The game is a Windows exe built on the PC with clang-cl; the
recompiled game code (`out/ppc*`, ~170 MB of libraries) comes from the user's own XEX files.

Options, in order of effort (ask the user which one):

1. **Switch-only work on the Deck, code on the PC.** Find the win with switches and profiles here; write down exactly
   what to change; the PC session implements and builds it, the user brings the new `fm4.exe` back. This is the default.
2. **Cross-build on the Deck.** FM2's port has a working Linux cross-build (`linux/build.sh` in the FM2 repo:
   clang-cl + lld-link + an xwin MSVC/SDK sysroot). FM4 doesn't have one yet. It would need clang/lld/llvm, cmake,
   ninja, python3, git, wine (SteamOS: a distrobox / toolbox Arch container), the repos below, the FidelityFX SDK 2.4+,
   ffmpeg, DXC (dxcompiler/dxil + dxc for ReGPU's compute shaders), and either the recompiled `out/ppc*` sources
   (XenonRecomp on the user's XEX; compiling them on 4 Zen 2 cores takes a long time) or the PC's prebuilt
   `fm4_ppc.lib`, `fm4_xmediafacade_ppc.lib`, `fm4_speechfacade_ppc.lib` linked in. After that, a ReGPU change rebuilds
   in minutes (renderer.cpp is one ~18k-line file). This is setup work: ask before starting it.

The source:
- FM4: `github.com/braaap-stututu/Fuzzy-Rittersport-4-ReGPU-testing`, branch `regpu-vulkan`. Game hooks and defaults:
  `src/gpu_host.cpp`; game hooks `src/hooks/fm4/`; kernel `src/kernel/`; write watch `src/memory.cpp`.
- ReGPU: `github.com/braaap-stututu/ReGPU`, branch `vulkan`. `src/gpu/renderer.cpp` (draws, surfaces, resolves,
  textures, pipelines), `src/gpu/gpu.cpp` (command processor, draw thread), `src/gpu/spirv/` (direct SPIR-V shader
  translator), `src/gpu/vk/` (Vulkan layer), `src/render/presenter.cpp`. `CLAUDE.md` and `docs/vulkan.md` there.

## 9. What to do first (suggested plan)

1. Boot once with `./play.sh` and check: it runs, the picture is right (compare with a PC screenshot if in doubt), the
   shader compile finishes, no errors in `fm2.log` (search `error`, `failed`, `VK_ERROR`). Note how long the first
   boot took.
2. Baselines, clocks fixed: `tools/deck/grid.sh base1`, `bench.sh race1` at 1280x800 MSAA 1x, then MSAA 4x, each
   twice alternated. Read for each: fps, 1%/0.1% lows, GPU ms/frame vs frame time, draw thread busy %, GPU clock and
   power. **Decide: GPU-bound or CPU-bound, and which thread.**
3. If **GPU-bound**: `REGPU_EXP_DRAW` modes to split the GPU frame, then the bandwidth switches
   (`REGPU_EXP_COLOR_FORMAT`, MSAA, aniso), then `REGPU_SPIRV_FINITE_MUL`, barriers; an RGP capture for the top cost;
   FSR 1/3 at 1280x800 for the player-facing result.
4. If **CPU-bound**: which thread (perf lite, MangoHud per-core, `/proc` thread times); the sampler on it; draw workers;
   `FM4_NO_WRITE_WATCH` / `FM4_RING_WAIT` / texture-check sanity checks under Wine.
5. Hitches: run a race at the 60 fps cap and look for long frames (`perf lite` "N long", `frametimes.py`'s frames over
   2x/4x the median). Shader compiles mid-race are the usual suspect on a first run.
6. Write the results in `DECK-RESULTS.md` in this folder (date, Deck model, clocks/TDP, each run's numbers, what you
   concluded, the code changes you recommend). The user carries it back to the PC session.
