# Converting ReGPU to Vulkan: the plan

## The decision (20 Nov 2026)

**Vulkan is ReGPU's one render backend.** Direct3D 12 is not kept beside it and there is no abstraction layer
between the two (the "A or B" choice in the first version of this plan is settled against both). The renderer
`src/gpu/renderer.cpp` is converted in place to Vulkan. Once the Vulkan renderer runs the test titles, the D3D12
code, fxc/d3dcompiler, DXGI and the experimental second renderer (`src/gpu/vk/renderer_vk.cpp`) are deleted.

**D3D12 as a reference.** It stays available only as a frozen reference: the tag **`d3d12-reference`** (= `main` at
`982cb25`, the last D3D12 build) and a build of it kept on this machine, for frame-by-frame comparisons. Fixes
are no longer made to it.

Read [vulkan.md](vulkan.md) for the experimental backend we start from, and [merge-2026-11.md](merge-2026-11.md) for
how the code got here.

## Where things stand (`d3d12-reference`)

- **`src/gpu/renderer.cpp`: 19.4k lines.**
  - About 1,400 of them touch D3D12 types or calls.
  - The rest is Xenos behaviour, and that is what the conversion must keep: EDRAM surfaces, resolves, tiling
    merges, write-backs, fetch setup, queries, upscaling and grading, and every per-game-tested `REGPU_*` switch.
- **Other D3D12 code:**
  - `presenter.cpp` (135 of 992 lines) and `recorded_list.h` (81 of 500).
  - `xr.cpp` (OpenXR on D3D12, 78 of 1,238).
  - The upscaler glue: `fsr3.cpp`, `dlss.cpp`, `xess.cpp`.
  - The shader compilers: `d3dcompiler.cpp` (fxc) and `dxc.cpp` (DXIL).
- **Already independent of the API, and kept:**
  - `gpu.cpp`, `ucode.cpp`, `hlsl.cpp` (its HLSL compiles to SPIR-V through DXC; `vk::AdaptHlsl` maps the bindings).
  - `texture.cpp` and `indices.cpp`.
  - `constants.h`: the constant blocks, byte for byte.
- **`src/gpu/vk` (from the FM2 port) is the parts bin:**
  - `vk_loader` (run-time loading, no link to vulkan-1) and `vk_context` (device, features, descriptor indexing).
  - `vk_swapchain` and `vk_shader` (DXC to SPIR-V, disk cache).
  - `vk_texture` (bindless arrays, uploads, BC formats) and `vk_surface` (EDRAM surfaces as images).
  - `vk_resolve` and `vk_blit` (resolve and blit passes), `vk_draw` (rings, descriptor sets), `vk_state` (pipeline
    state from registers).
  - `renderer_vk.cpp` is a working reference for each D3D12 piece's Vulkan form (FM2 ran it), then goes.
  - `regpu_vkprobe` passes on the RX 7900 XTX.

## The public API changes (regpu.h)

All of these are breaking changes for the ports. They land together, with the ports' updates (FM3, PGR3, PGR4,
FH1/Glue2GPU and the rest) described in the README.

- `Config::renderer` goes away.
- `Config::hostDevice` / `hostQueue` become a host `VkInstance`, `VkPhysicalDevice`, `VkDevice`, queue family and
  `VkQueue`.
- `Host::presentImage` hands over a `VkImage` with its layout and a semaphore, instead of an `ID3D12Resource` and its
  state.
- `hostWindow` stays: ReGPU makes the `VkSurfaceKHR` from the HWND on Windows, or from a Wayland/X11 handle on Linux.
- The README build section drops fxc, d3dcompiler_47.dll and REGPU_FSR1_CS's fxc step. It adds the Vulkan headers
  and DXC's dxcompiler.dll / dxil.dll.
- Glue2GPU's plugin asks ReXGlue for the `vulkan` backend instead of `d3d12`.

## Design rules for the converted renderer

- **Same structure, same switches.** `renderer.cpp` keeps its functions, data and `REGPU_*` switches. Where a D3D12
  object was, a Vulkan one is. A reviewer should be able to diff the old and new function side by side.
- **Recording.** `RecordedList` keeps its job (record on the draw thread, replay on another) with Vulkan ops
  replayed into a `VkCommandBuffer`. Draw workers use secondary command buffers or their own primary ones.
- **Synchronisation.** synchronization2 throughout. A small layout tracker per image replaces D3D12 resource states:
  - EDRAM surfaces and resolved images.
  - Implicit promotion/decay doesn't exist in Vulkan, so every transition the D3D12 code relied on implicitly becomes
    explicit.
- **Binding.**
  - The shaders' model stays: `gtex`/`gcube`/`gvol` bindless (descriptor indexing, update-after-bind), eight samplers,
    two constant blocks as dynamic UBOs, and `gmem`/`gphys` storage buffers.
  - Root constants become push constants.
  - `VK_EXT_descriptor_heap` can come later as a fast path.
- **Memory.**
  - Texture heaps and placed resources become VkDeviceMemory blocks with suballocation (or VMA).
  - The upload ring stays host-visible; ReBAR is a device-local, host-visible memory type.
  - Guest memory as `gphys` uses `VK_EXT_external_memory_host` where available, else copies.
- **Pipelines.** `PipelineKey` and the pipeline manifest/seed stay. VkPipelineCache plus the disk cache replace the
  D3D12 pipeline library. Dynamic state replaces per-state PSOs where the key allows.
- **Shaders.** Translated shaders and ReGPU's own passes all go HLSL to SPIR-V with DXC:
  - blits and resolves;
  - `fsr1.hlsl` and `postaa*.hlsl`;
  - gamma and enhance passes, motion variants.
  - The fxc-specific workarounds and DXBC caches go. The shader cache key changes (all caches rebuild).
- **Queries and predication.** Occlusion queries become a `VkQueryPool` (VK_QUERY_TYPE_OCCLUSION, precise).
  SetPredication becomes `VK_EXT_conditional_rendering`, falling back to CPU decisions where it is missing.
- **Debugging.**
  - The debug server, captures and `regpuctl` keep working; screenshots read back a VkImage.
  - Validation layers on behind `REGPU_VK_VALIDATION`.
  - RenderDoc and RGP for GPU work. RGA compiles a SPIR-V shader to RDNA ISA, to see its register use and
    instruction count (the road shader from Tankyx's notes). GFXReconstruct (in the Vulkan SDK) captures a frame and
    replays it without the game.
- **Upscalers on Vulkan:** the FSR SDK 2.4.0 ships signed Vulkan DLLs (`amd_fidelityfx_loader_vk`, `upscaler_vk`,
  `framegeneration_vk`) at parity with DX12, FSR 4 included.
  - Its Vulkan entry point is `ffx_api_vk.h`. Inputs go to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
  - The SDK needs Vulkan 1.2; ReGPU's baseline is 1.3.
- **Linux testing:** the user's Steam Deck (RADV) and a Linux boot of their PC with the RX 7900 XTX.

## Lessons from FM2's Vulkan backend

These are notes from Tankyx's agent, who wrote FM2's Vulkan backend (`src/gpu/vk`), received 20 Nov 2026. The
measurements are FM2's, at 1080p. FM2's environment variable names are `FM2_*`; in ReGPU's copy they are `REGPU_*`.

**What not to repeat**
- **Async pipeline and shader compilation were tried and reverted.** Some frames skipped about 1,000 draws, and the
  1% lows did not improve.
- **The rule for us:** a draw never skips because its pipeline isn't ready. ReGPU's D3D12 path already works this
  way, and the converted renderer keeps it:
  - a plain-fetch variant builds in the background while the draw uses the base pipeline;
  - everything else builds on the draw thread, or comes prebuilt from the manifest.

**Integrated GPU (RADV APU): GPU-bound, 28-30 fps, GPU frame about 32 ms**
- D3D12 through vkd3d-proton gets about 35 fps on the same driver. The gap is in what is asked of the driver, not
  in CPU work.
- **16x anisotropy:** it costs Vulkan about 10 ms against D3D12's 4.
  - With anisotropy off, the two backends are level: 41.9 fps on Vulkan against 40.4 on D3D12.
  - The road pixel shader `ff5c8bbf7b62e0f2` alone takes 12.8 ms. One 84-vertex draw takes 9.5 ms.
  - **Test:** make that shader return a constant on both backends (FM2's `*_DEBUG_PS`). If the gap stays, the cause
    is overdraw or lost early depth rejection. If it closes, it's the sampling.
  - For ReGPU: compare the sampler setup in the bring-up (`MaxAnisotropy`, mip filtering, LOD bias) with D3D12's.
    s0 is 16x unless `REGPU_ANISO_FROM_FETCH` is set.
- **Alpha-test-free pixel shader variants:** every translated pixel shader ends in a discard switch, which can cost
  early depth rejection.
  - FM2's Vulkan backend never had these variants. ReGPU's renderer has them: pipeline flag 0x400 with
    `NoKillShader`.
  - The in-place conversion keeps them automatically. The gain on the APU has never been measured.
- **Plain-fetch variants:** 2.5% less GPU time on a 4070 under D3D12.
  - ReGPU has them as `REGPU_FETCH_VARIANTS` (`fetchMask`, `FetchVariantOf`), together with
    `REGPU_FETCH_SPECIALIZE`.
  - They carry over the same way.
- **Default anisotropy on integrated GPUs:** going from 16x to off gains about 12 fps. 4x and 8x were not measured.
  Phase 3 should pick a default for iGPUs from a measurement.
- **FSR 1 in the present pass:** it needs no motion vectors, and `fsr1.hlsl` already exists. It is the cheapest
  upscaler to bring up first in phase 4.

**RTX 5090: CPU-bound, about 207 fps, GPU frame about 1.3 ms**
- The render thread is the limit. FM2's Vulkan was already about three times D3D12 (about 70 fps) here.
- **Prebuild pipelines at load.**
  - On NVIDIA, about 38 pipelines are still built on the render thread on every warm run (none on the APU). The 1%
    low is 108 fps against a 207 average.
  - The model is ReGPU's `pipelines.manifest`, rebuilt at boot. Make sure the Vulkan side feeds
    `VkPipelineCache` from it, and find out why NVIDIA still misses (likely a key or state not in the manifest).
- **Profile before changing anything else.** Use `REGPU_VK_SAMPLE=1` here, or the converted renderer's own timers.
  In FM2's last timed frame:

  | Bucket | Time |
  |---|---|
  | Untimed work inside the draw function | 1.34 ms |
  | Setup | 0.59 ms |
  | Descriptor binds | 0.32 ms |
  | Draw calls | 0.13 ms |

- **Bind only set 0 per draw.** `vk_draw.cpp` rebinds all four sets (with the dynamic UBO offsets) on every draw.
  - Sets 1-3 (the bindless textures) don't change between draws.
  - Binding them once per command buffer and set 0 per draw is a small, safe saving.
  - This should be the rule from the start of phase 2.

## How the core is converted (decided 20 Nov 2026)

**The whole core moves in one block.** The user chose this over a temporary D3D12-to-Vulkan bridge.
- Draws, EDRAM, resolves and textures all share one command stream, so they cannot move one at a time.
- FM3 does not render on the branch until the core is done. `main` stays playable.

**Every commit still builds.** The Vulkan code goes in beside the D3D12 code; D3D12 is deleted only at the end.
`tools/d3d12_left.py` will count what remains.

**The order:**

| Step | Work | Status |
|---|---|---|
| 1 | Vendored Vulkan-Headers 1.4.369, volk (the loader) and VMA. Vulkan is always built. | Done, `2ec5f4b` |
| 2 | **Device** (`src/gpu/vk/device.{h,cpp}`) | Done, `266fac0` |
| 3 | **`RecordedList` with Vulkan operations** (`src/gpu/vk/recorded.h`, test `regpu_vkrecord`) | Done, `8d57e6b` |
| 4 | **Resources** | Next |
| 5 | **Pipelines and shaders** | |
| 6 | **The renderer, function by function** | |
| 7 | **First boot**, only once the user agrees, then slice-by-slice comparison with `d3d12-reference` (phase 2 below) | |

**Step 2, the device:**
- Vulkan 1.3, with dynamic rendering, synchronization2, maintenance4 and descriptor indexing (update-after-bind and
  partially bound arrays).
- Timeline semaphores, robustness2 null descriptors, a VMA allocator and the pipeline cache.
- Optional: conditional rendering, external host memory, push descriptors and the swap chain.
- The presenter creates it and hands it out where `RenderDevice()`/`RenderQueue()` hand out D3D12's today.

**Step 3, `RecordedList` with Vulkan operations:**
- Replay goes into a `VkCommandBuffer`.
- Barriers become sync2 barriers, driven by a per-image layout tracker.
- Render targets: the replay begins and ends `vkCmdBeginRendering` lazily. Setting targets only records them; the first
  draw or clear begins rendering; a barrier, copy or new target ends it.
- Root constants become push constants.

**Steps 2 and 3, what to know:**
- `vk::Submit`, `vk::Completed` and `vk::WaitFor` on the queue timeline replace D3D12's fences. The queue needs a
  lock (`Device::queueLock`), where a D3D12 queue was free-threaded.
- **Queries:** `RecordedList` begins and ends occlusion queries outside rendering, so a query can span target changes.
  That costs a render pass break per query; measure it once the game runs.
- **Validation layer:** the LunarG Vulkan SDK 1.4.363.1 is installed on the user's PC.
  - `REGPU_VK_VALIDATION=1` turns on the layer, `2` adds synchronization validation. Messages go to `regpu.log`.
  - Both tests run clean at both levels.
- **Barriers into attachment layouts:** render passes always load and store their targets, so a barrier into an
  attachment layout must cover reads as well as writes. Use `vk::kColorTargetAccess` and `vk::kDepthTargetAccess`.
  Synchronization validation caught this on its first run.

**Step 4, resources:**
- Committed and placed resources, and the texture heaps, become VMA allocations.
- The descriptor heaps become descriptor sets:
  - set 0: the constant blocks as dynamic UBOs, plus the upload ring and guest memory as storage buffers;
  - set 1: the samplers;
  - sets 2-4: bindless 2D, cube and 3D.
- RTV and DSV handles become `VkImageView`s.

**Step 5, pipelines and shaders:**
- The root signature becomes a pipeline layout, and `PipelineKey` builds a `VkPipeline` with dynamic viewport,
  scissor, stencil reference and blend constants.
- `PipelineLibrary` becomes `VkPipelineCache` plus the manifest.
- `dxc.cpp` compiles to SPIR-V (`vk::AdaptHlsl`). The fxc path goes.

**Step 6, the renderer, function by function:** in the order of the D3D12 map (`Init`, `GetSurface`,
`BeginFrameIfNeeded`, `EnsurePipeline`, `Apply`, textures, resolves, `RendererOnSwapImpl`, then the extras).

## Phases

The work happens on the long-lived branch **`vulkan`**. `main` stays at the D3D12 renderer until the branch renders
the test titles, then the branch replaces it.

0. **Ground truth.**
   - Choose fixed scenes in FM3 and PGR3 (and one FH1 scene via Glue2GPU if possible).
   - Capture them on the `d3d12-reference` build with `regpuctl.py shot` and save them as references.
   - Write a frame-diff script.
   - Run FM3 and PGR3 on `main` first, so merge regressions are known before the conversion starts.
1. **Bring-up.**
   - Vulkan device, window and swap chain from the presenter.
   - The upload ring, and guest memory as storage buffers.
   - Shaders through DXC to SPIR-V, one pipeline, the first draws into an EDRAM surface.
   - The first resolve, presented.
   - Use `renderer_vk.cpp` and FM2's notes for every piece.
2. **The renderer core, slice by slice**, diffing against the references after each slice:
   - textures (decode, upload, bindless, cubes, 3D);
   - EDRAM surfaces and MSAA;
   - resolves (blits, writes back to memory, bands, enclosing images, tiling merges);
   - draws (vertex/index paths, the vertex arena, primitive types, the alpha test, motion variants);
   - occlusion queries and predication;
   - the frame end, grading and present.
3. **Threads and speed:** the draw thread, workers, async present and the replay thread on Vulkan command buffers;
   pipeline compile workers and caches; the texture streaming threads. Measure against the reference build's fps.
4. **Upscaling and extras:**
   - FSR 1, then FSR 3/4 (FidelityFX Vulkan backend), DLSS (NGX Vulkan) and XeSS (Vulkan).
   - Post AA (FXAA, SMAA, CMAA 2, TAA), SSAA and EQAA (or its Vulkan absence).
   - Hyper mode, VR (OpenXR Vulkan binding) and triple screen.
5. **Host API and ports** (together with stage R3 of [modular-runtimes.md](modular-runtimes.md), so the ports take
   one set of breaking changes): the regpu.h changes above, Glue2GPU, and a port each (FM3 first) updated against the
   branch.
6. **Remove D3D12:** DXGI, d3d12, d3dcompiler, fxc, DXIL, the FidelityFX DX12 libs, `renderer_vk.cpp` and the
   `REGPU_VULKAN` option (Vulkan is always on). Merge `vulkan` into `main`.
7. **Linux native.** Win32 pieces behind the host: windowing (SDL3 or Wayland/X11 surfaces), threads, DLL loading
   (vulkan-1 / libvulkan.so.1, dxcompiler).
   - Build with clang.
   - Ship in the Steam Linux Runtime 4.0 SDK.
   - Test on RADV (Steam Deck), ANV, NVK and NVIDIA's proprietary driver.

## Before starting

- **FM4's features:** decide which of [fm4-features-to-port.md](fm4-features-to-port.md) are wanted. A D3D12-side
  feature is cheaper to port before phase 2 than after, or it can be ported straight into the Vulkan renderer later.
- **Baseline:** the Vulkan baseline is Vulkan 1.3 with descriptor indexing, synchronization2, dynamic rendering, push
  descriptors and robustness2. Optional: conditional rendering, external memory host, descriptor heap. The research
  report in the ReGPU-V workspace has the driver coverage.
- **Local build:** see `G:\claude code projects\ReGPU-V\work\scripts\README.md` on this machine.
