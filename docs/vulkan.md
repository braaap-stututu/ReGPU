# The Vulkan renderer

ReGPU renders with Vulkan. `src/gpu/renderer.cpp` was converted from Direct3D 12 in place (branch `vulkan`, November
2026: [vulkan-conversion-plan.md](vulkan-conversion-plan.md)); the last Direct3D 12 renderer is the tag
`d3d12-reference`, kept for comparing frames. The structure, the Xenos behaviour and the `REGPU_*` switches are the
D3D12 renderer's: where a D3D12 object was, a Vulkan one is.

## Requirements

- **Vulkan 1.3**: dynamic rendering, synchronization2, maintenance4, timeline semaphores, descriptor indexing
  (update-after-bind, partially bound, runtime arrays: 65,536 sampled images per stage), `VK_EXT_robustness2`'s null
  descriptors, BC texture compression, independent blend, sampler anisotropy, cube arrays. A GPU without them ends the
  process at start with a FATAL line naming what is missing.
- **Used where the driver has them:** `VK_EXT_conditional_rendering` (REGPU_DRAW_CULL's predication; without it the
  draws are drawn), `VK_EXT_external_memory_host` (guest memory read in place; without it vertex data is copied),
  `VK_KHR_push_descriptor`, `VK_EXT_memory_budget`, `VK_NVX_binary_import` + `VK_NVX_image_view_handle` (DLSS),
  `VK_EXT_primitive_topology_list_restart` (strips with the cut index on list-class pipelines),
  `VK_KHR_compute_shader_derivatives` / NV, `VK_EXT_mutable_descriptor_type`, subgroup size control, 8-bit integers and
  integer dot products (FidelityFX's FSR 3/4 and XeSS create their pipelines on ReGPU's device and need these),
  storage image reads and writes without a format (the post filters and passes), precise occlusion queries, pipeline
  statistics queries.
- **At build time:** DXC with SPIR-V output (the Vulkan SDK's `Bin/dxc.exe`, or `REGPU_DXC_EXE`) compiles ReGPU's
  own compute shaders into the library (`tools/embed_hlsl.py`). The Vulkan headers, volk (the loader) and VMA (memory)
  are vendored in `thirdparty/`; nothing links against `vulkan-1`.
- **At run time:** `dxcompiler.dll` and `dxil.dll` beside the executable (a DirectXShaderCompiler release, or the
  Vulkan SDK's: the Windows SDK's copy has no SPIR-V output). The translated shaders are compiled with them and cached.

## The pieces (`src/gpu/vk`)

| File | What |
|---|---|
| `device.*` | The instance, the GPU (`REGPU_ADAPTER`: a name part, an index, `high` / `low`), the device and queue, VMA, the pipeline cache (`vk-pipelines-<vendor>-<device>.bin` in `Config::userDir`), the queue timeline, `Submit`. A host's device is adopted instead (`Config::hostDevice` and the rest). `REGPU_VK_VALIDATION=1` the validation layer, `2` with synchronization validation; its messages go to `regpu.log`. |
| `recorded.h` | `vk::RecordedList`: the renderer's calls recorded on the draw thread, replayed into a command buffer on a worker (as D3D12's `RecordedList` did). Render passes begin lazily on the first draw or clear and end at barriers, copies and the like; predication becomes conditional rendering; queries around one draw stay inside the render pass. Hand-offs end, queue and submit buffers in stream order; `HandOffNew` lets the replay thread begin the next buffer itself (command pools are single-threaded). |
| `resources.*` | Buffers and images in VMA memory; each image carries its layout and last use (`Transition`, `Barriers`: synchronization2), replacing D3D12's resource states; the graveyard (deferred destruction on the queue timeline); format properties. |
| `bindings.*` | The translated shaders' bindings: set 0 (the constant blocks as dynamic uniform ranges of the upload ring, `gmem`, `gphys`, the eight samplers), sets 1-3 the bindless 2D, cube and 3D arrays (bound once per command buffer). |
| `pipelines.*` | `VkPipeline`s from the pipeline key's state (dynamic viewport, scissor, stencil reference, blend constants and topology). |
| `passes.*` | ReGPU's own passes (resolve blits, FSR 1, gamma, the post filters, draw culling): HLSL to SPIR-V with any entry point, the descriptor set layout read from the SPIR-V, transient descriptor sets per frame slot, constants from the ring. |
| `vk_shader.*`, `adapt_hlsl.cpp` | HLSL to SPIR-V through DXC (in process, else the `dxc` executable), with a disk cache; `AdaptHlsl` maps the translator's declarations onto the bindings. |
| `vk_loader.*` | volk: Vulkan loaded at run time. |

## How the renderer uses them

- **Frames:** three frame slots, each with its own command pools (one per thread that records: the main stream and
  each draw-piece thread), upload ring, set 0s and garbage. A slot is reused once the renderer's fence (a timeline
  semaphore with D3D12's values) says its last frame has completed.
- **Images:** shared through `ImageRef` / `BufferRef` (an intrusive count, as `ComPtr` was); the last release buries
  them on the device timeline, and the frame garbage holds references until the slot comes round.
- **Textures:** uploaded through the ring; bindless entries get views of their own where the texture needs a swizzle
  (BC4 as `rrrr`, R8 as `rrr1`), another format or another dimension.
- **Resolves:** a blit pass per resolve (depth through the read-only depth layout, the D24S8 pack reading the stencil
  aspect), `REGPU_RESOLVE_TWO_STEP`'s hardware resolve, write-backs through readback buffers (a host-read barrier at
  each frame end).
- **Motion vectors:** b2 reads last frame's vertex constants where that frame left them: a second set 0 per slot whose
  b2 is over the previous slot's ring.
- **Upscalers:** FSR 3/4 through FidelityFX's Vulkan API (`amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll`),
  DLSS through NGX's Vulkan API, XeSS through `xess_vk.h`.
- **Presenting:** `render/presenter.cpp` makes a Vulkan swap chain on the window (FIFO with vsync, else immediate);
  the frame end blits the picture onto the output canvas scaled to the window, or draws the display gamma pass
  straight into the back buffer.

## Tests

Each builds as its own target and must pass with `REGPU_VK_VALIDATION=2` and no validation messages
(`work/scripts/vktests.ps1` on the development machine runs them all):

| Target | What |
|---|---|
| `regpu_vkprobe` | The device, a timeline submission, the translator's preamble to SPIR-V. |
| `regpu_vkrecord` | `vk::RecordedList`: clears inside and outside passes, an empty query. |
| `regpu_vkresources` | Every format the renderer uses, an upload and readback round trip, a depth clear, the graveyard. |
| `regpu_vkdraw` | A draw through the translated shaders' bindings: set 0's constants, gmem, gphys and a bindless texture. `REGPU_VKDRAW_GPL=1`/`2`: from pipeline libraries; `REGPU_VKDRAW_VSOUT=1`: the XTX miscompile below; `REGPU_VKDRAW_DUMP=1`: the SPIR-V to files. |
| `regpu_vkpasses` | A compute and a graphics pass from HLSL, their layouts from SPIR-V, transient sets, read back. |

## Checked with a game (FM3, 20 Nov 2026)

FM3 (the `regpu-vulkan` branch of its port) on an RX 7900 XTX against its D3D12 build:
- **Validation:** none on the race grid and menus under synchronization validation; none with FXAA, SMAA, CMAA 2,
  SSAA, FSR 1/3 or XeSS; two messages from inside FSR 4's own shaders (not ReGPU's).
- **Image:** the race grid matches (mean difference under 1/255; the rest is animated crowd and sub-pixel edges).
- **Speed:** level (the 9800X3D's graphics too, since multisampled colour surfaces stay in GENERAL: 22 -> 56 fps in
  the race at 720p 2x, D3D12 59). 4K 4x MSAA with CMAA 2: 230 fps and 4.18 ms of GPU a frame against 232 fps and 4.14 ms; without the
  filter 278 against 284 fps. Per pass (`REGPU_GPU_PASSES=1`), the main scene pass is ~5% slower and spread over many
  shaders (the driver's SPIR-V against its DXIL compile of the same HLSL); CMAA 2 is slightly faster. At 1080p both
  hold the 360 fps cap with the draw thread equally busy. FM3's benchmark race at 4K (40 s, the AI driving): 233.1
  against 233.5 fps, GPU 4.11 ms both, p99 4.89 against 5.76 ms (steadier on Vulkan).
- **Measuring:** the light performance log's "GPU ms/frame" (timestamps around each frame) and `REGPU_GPU_PASSES=1`
  (per pass) / `REGPU_GPU_SHADERS=1` (per draw, by shaders) work as on D3D12. Compare the same settings on both sides:
  a profile with a different `aa=` makes one look slower.

## The direct SPIR-V translator (the default)

`src/gpu/spirv` translates Xenos microcode straight to SPIR-V (its own builder, no glslang, no DXC), mirroring
hlsl.cpp: pixel shaders (ALU, control flow, texture fetches as TFetch and the lean rules under `REGPU_LEAN_FETCH`, the
renderer's no-kill, plain-fetch and specialised variants) and vertex shaders (the guest program as a function, the
VSOut wrapper: guest index fetch, rectangles and points, the viewport transform; baked vertex fetches).
`REGPU_SPIRV_DIRECT` (start-time; on by default since 21 Nov 2026): unset or `1` both stages, `ps` or `vs` one, `0`
none (everything through HLSL and DXC, the reference); shaders the game rewrote or a debug edit changed,
and vertex shaders under `REGPU_DYNAMIC_VFETCH` / `REGPU_LEAN_VFETCH` / `REGPU_PLAIN_MUL_VS` / triple / stereo, keep HLSL.
`regpu_spirv_check` runs it over captured microcode with spirv-val.
- FM3, both stages: ~0.7 ms a shader against 100-900 ms through DXC; a cold boot reaches the race grid in 12 s (as warm)
  against 42 s; the frame within ~0.3/255 of the HLSL path; no validation messages (grid, menus, a race, every AA mode).
- GPU: the RX 7900 XTX's 4K race 4-5% faster (238-240 fps against 229); the 9800X3D's graphics ~4% slower (54 against
  56.6). VK_AMD_shader_info (register counts per pipeline) would show whether register use costs RDNA2 occupancy.

## Pipelines: dynamic state, libraries

- Depth, stencil, cull and depth bias are dynamic for the draws' pipelines (renderer.cpp DynamicDrawState,
  PipelineLookupKey, vk::DrawState): FM3's cold boot to the grid 17.6 s against 26.3, 2,938 pipelines against 3,800,
  +3% GPU at 1080p. Blend dynamic too with `REGPU_VK_DYNAMIC_BLEND=1` (EDS3; +10% GPU at 1080p on an RX 7900 XTX).
- Graphics pipeline libraries (VK_EXT_graphics_pipeline_library): in the code, off. `REGPU_VK_PIPELINE_LIBRARY=1`
  enables the extension and builds the draws' pipelines from four cached parts, fast-linked, with an optimised (LTO)
  build swapped in later on a low-priority thread (`REGPU_VK_NO_LTO=1`: fast links only). Do not turn it on for play:
  on the RX 7900 XTX (driver 26.11.2) FM3 loses the device at its first frames, with or without LTO, dynamic state or
  a pipeline cache, while `regpu_vkdraw` with `REGPU_VKDRAW_GPL=1` / `2` draws correctly. Finding the pipeline needs a
  crash dump (Radeon GPU Detective), not more boots.

### A shader compiler fault on the RX 7900 XTX

`regpu_vkdraw` with `REGPU_VKDRAW_VSOUT=1` draws with the translated shaders' interface (VSOut). Its pixel shader reads
one interpolant and all four bindings (constants, gmem, gphys, a bindless texture). On the RX 7900 XTX (driver
26.11.2, LLPC) it writes (0, 0, 0, 0). Drop any one of the five reads and it is right; the SPIR-V validates and is
plain; the integrated Radeon (the same driver release, AMD's other shader compiler) draws it correctly. FM3's shaders
make that combination all the time and render right, so the trigger is narrower than this; but it is a miscompile in
the driver, and the same GPU is the one the pipeline libraries hang. A candidate for an AMD report, and a reason to
re-check both after a driver update.

### Motion vectors (temporal upscalers)

TAA, FSR 3/4 and XeSS need motion vectors, so the scene's draws use motion variants of their shaders. The vertex shader
runs the guest program a second time with last frame's constants (`vcPrev`, set 0 binding 2) and outputs both clip
positions (locations 17 and 18). The pixel shader adds the motion target (location 1) and the reactive mask (location 2).

The HLSL motion variants were made from the base shader, so motion draws lost the alpha-test-free, plain-fetch and
specialised fetch variants. A direct shader's motion variant now comes from the direct translator: `SpirvOptions::motion`
on top of the variant's own options. It is made on a pipeline-pool thread and falls back to HLSL.
`REGPU_MOTION_HLSL=1` gives the HLSL motion variants, as before.

FM3 race grid, 1440p, RX 7900 XTX (21 Nov 2026):

| | GPU ms per frame, HLSL motion | GPU ms per frame, direct motion |
|---|---|---|
| TAA | 2.72 | 2.27 |
| FSR 3 Quality | 2.79 | 2.40 |
| native 4x MSAA (no motion) | 2.21 | 2.21 |

- The draw thread does about 0.2 ms a frame more: a motion draw now picks its fetch variant, as non-motion draws do.
  On the CPU-bound grid that is 306 fps against 327.
- In the same work, two draw-thread costs that hit every MSAA-off configuration:
  - `StateFromKey` called `getenv("REGPU_NO_ALPHA_DITHER")` on every pipeline lookup: 27% of the draw thread with TAA.
  - The temporal stats were locked increments: about 7%.
- Why upscaling saves little in FM3: the scene's main pass costs about 1.5 ms at 1440p and barely depends on its pixel
  count (4x MSAA +0.04 ms, 4x the pixels +0.6 ms), while FSR 3's pass is 0.45 ms (Quality) and FSR 4's about 2 ms on
  RDNA 3. Most of the pass is fixed cost: vertex work and per-draw overhead.

## Pipeline stalls

New pipelines are built when a draw first needs them; on an RX 7900 XTX a build that misses the driver's cache takes
100-250 ms. FM3 (21 Nov 2026): a race load drew 69 new pipelines in one frame (2 s), races had 150-280 ms frames.
Start-time switches; the first three are on by default since the user compared presets in play (21 Nov 2026: no
pipeline stalls left in races, new pipelines show as a frame or two of pop-in):
- `REGPU_DEFERRED_PIPELINES` (`0` off): a draw that must wait for its pipeline records a deferred bind
  (`RecordedList::BindPipelineDeferred`) and the draw thread goes on; the build goes to the front of the pool and the
  replay thread waits for it. A frame's new pipelines build in parallel instead of one by one.
- `REGPU_PIPELINE_BUDGET_MS=<ms>` (default 0; `off`: draws wait as before): past that much build time in a frame (waited, or estimated for deferred binds),
  screen-pass draws whose pipeline isn't built are dropped for the frame and built in the background. Off-screen
  passes and clears still wait (they may be drawn once and kept). `0`: screen-pass draws never wait.
- `REGPU_COMPILE_PRIORITY=<Win32 level>` (default -1; -2 lowest, -15 idle) for the shader and pipeline threads,
  `REGPU_COMPILE_THREADS=<n>` shader compile workers (default all threads but one), `REGPU_PIPELINE_THREADS=<n>`.
- `REGPU_BUDGET_OFFSCREEN` (`0` off; with a budget): once the game has settled (no load hold for 4 s), off-screen and
  depth-only draws over the budget are dropped too. During loads they still wait: liveries are built once then.
- `pipelines: draw waits for <vs>/<ps>: <why>`: the first 200 draws a run that waited for a pipeline, and why they
  couldn't skip (off-screen pass, depth only, clear, must-draw screen pass, during a load).
- The log line `pipelines: last second N draws dropped by the budget, M deferred binds (average build X ms)`.
- The direct translator (above) makes shaders and their variants in ~1 ms, where DXC took 200-1600 ms for each spec
  variant (the draw's constants baked in, so new ones come with each new scene).
- The warm-up (PrecompileAll) skips stored pipelines whose shaders aren't there yet (static shaders come later); the
  seed step (BuildSeedPipelines) builds them, and counts a pipeline as built only when it was started (it took any
  the manifest knew).
- Direct shaders' fetch and spec variants are made on a pipeline-pool thread, not the draw thread
  (`REGPU_VARIANT_INLINE=1`: as before). Spec variants can't be built ahead (their constants come with the draw); the
  draw uses the base pipeline meanwhile, and "seed coverage: built while drawing" no longer counts them.
  `REGPU_LOG_UNPREPARED=1` logs the first 40 drawn pipelines that weren't built ahead.
- Retired images (`ReleaseWithFrame`: idle EDRAM surfaces, resized targets) are destroyed on a deleter thread:
  freeing a dedicated allocation is a kernel call of up to ~10 ms. FM3's cockpit camera left the other view's depth
  surfaces idle, and their release made a 16-18 ms frame. `REGPU_DESTROY_ON_DRAW_THREAD=1`: as before.
- Pipelines built during play don't hit the driver cache when their create info changes: a complete pipeline's dynamic
  states must stay in the same order (28a4d62).

## Not done yet

- **VR (OpenXR):** the session is on Vulkan (`XR_KHR_vulkan_enable`, else enable2; render/xr.cpp). Checked against a
  runtime without a headset only (it waits and retries); run it with a headset: screen, cockpit and stereo modes.
- **DLSS and XeSS** were converted without their SDKs on the converting machine: build them once with the SDKs.
- **Linux:** the renderer is Vulkan, but the window, the overlay text (GDI), threads and a few file paths are Win32.
- **Not run yet with a game:** VR with a headset, DLSS (no NVIDIA GPU here), FSR 4 beyond its start.
