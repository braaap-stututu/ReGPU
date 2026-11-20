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
  - RenderDoc and RGP for GPU work.

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
5. **Host API and ports:** the regpu.h changes above, Glue2GPU, and a port each (FM3 first) updated against the
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
