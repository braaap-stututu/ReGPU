# Converting ReGPU to Vulkan: the plan

The goal is ReGPU running on Vulkan: a native Linux and Steam Deck path without vkd3d-proton, and Vulkan on Windows.
The user will give the specifics; this file is the shared starting point. Read [vulkan.md](vulkan.md) for the
backend as it is, and [merge-2026-11.md](merge-2026-11.md) for how the code got here.

## Where things stand (20 Nov 2026, `main` at `182afe2`)

- **One renderer, D3D12:** `src/gpu/renderer.cpp`, 19.4k lines. About 1,400 of those lines touch D3D12 types or calls.
  The rest is Xenos behaviour:
  - EDRAM surfaces, resolves, tiling merges and write-backs;
  - texture and vertex fetch setup;
  - occlusion queries;
  - upscaling and grading;
  - all the per-game-tested switches.
- **D3D12 coupling outside it:**

  | File | Lines touching D3D12 | Total lines |
  |---|---|---|
  | `presenter.cpp` | 135 | 992 |
  | `recorded_list.h` | 81 | 500 |
  | `xr.cpp` (OpenXR on D3D12) | 78 | 1,238 |

  The upscaler glue (`fsr3.cpp`, `dlss.cpp`, `xess.cpp`) and the shader compilers (`d3dcompiler.cpp`, `dxc.cpp`) are
  small.
- **API-neutral already:**
  - the command processor and the `Vd*` exports (`gpu.cpp`);
  - the microcode decoder (`ucode.cpp`) and the translator (`hlsl.cpp`, HLSL that DXC can take to SPIR-V);
  - texture decoding (`texture.cpp`) and index expansion (`indices.cpp`);
  - the host API (`regpu.h`).
- **A second renderer, Vulkan:** `src/gpu/vk`, 8.3k lines, from the FM2 port, behind `REGPU_VULKAN`. It reimplements
  the draw, resolve and texture logic on its own, and lacks everything the D3D12 renderer gained since mid-November.
  It builds, opens a device and compiles the preamble (`regpu_vkprobe`), but has never run a game in ReGPU.
- **`RecordedList` (`recorded_list.h`):** the renderer already records every command into a buffer replayed on
  another thread. Its ~40 ops mirror `ID3D12GraphicsCommandList`, so it is a natural seam for a backend interface:
  - `DrawInstanced`, `DrawIndexedInstanced`, `Dispatch`;
  - `ResourceBarrier`, `OMSetRenderTargets`, the root parameters;
  - copies, resolves, queries, predication;
  - `Call`.

## The decision to make first

Pick one of these two ways. Everything after depends on it.

**A. Grow `src/gpu/vk` to parity.** Two renderers stay. Each D3D12-only feature (the gap list in vulkan.md) is
reimplemented in `renderer_vk.cpp`.
- **Pro:** quick early results; FM2 already ran it.
- **Con:** every Xenos fix must be made twice, forever. The FM2 backend already fell behind in a few weeks.

**B. One renderer over a small GPU interface, with D3D12 and Vulkan beneath it (recommended).** `renderer.cpp` keeps all
the Xenos logic. Its D3D12 calls go through an interface shaped like `RecordedList`: resources, views and descriptors,
pipelines, commands, barriers, queries. `src/gpu/vk` is the parts bin for the Vulkan side:
- context and loader;
- swap chain;
- SPIR-V compiling and the pipeline cache;
- descriptor layouts;
- its resolve and blit shaders.

- **Pro:** every fix lands once; D3D12 stays as the reference, and the two backends can be compared frame by frame.
- **Con:** a long refactor before Vulkan draws anything new; the barrier and descriptor models have to be designed
  carefully (D3D12 resource states vs Vulkan layouts; root signatures vs descriptor sets and push constants).

The Vulkan research report (`reports/Vulkan renderer porting and optimization.md` in the ReGPU-V workspace) argues for
B. It recommends a thin RHI shaped like D3D12, with D3D12 kept as the reference backend and the two compared by image
diffs. Its Vulkan baseline is the feature set vkd3d-proton needs:
- Vulkan 1.3, descriptor indexing and push descriptors, robustness2;
- `VK_EXT_descriptor_heap` as the fast path where present.

## Phases (for B; A reuses phases 0, 1 and 5)

0. **Ground truth.** Pick the test titles (FM3, PGR3) and fixed scenes. Capture D3D12 reference frames with
   `regpuctl.py shot` and the frame comparison tools. Add a Vulkan validation-layer run, plus RenderDoc and Radeon GPU
   Profiler (RGP) captures on the RX 7900 XTX. Write a frame-diff script for D3D12 against Vulkan output.
1. **Make the existing Vulkan backend draw a game.** This is the quickest way to find what breaks in the shared
   pieces:
   - run FM3 with `REGPU_RENDERER=vulkan`;
   - fix the vulkan.md gaps it hits (`vf.w` units, 3D textures, the deferred swap chain so Vulkan can own the window);
   - record which D3D12-only switches the title needs.
2. **Define the interface.** Start from `RecordedList`'s ops and the resource creation in `renderer.cpp`: textures,
   EDRAM surfaces, heaps and placed resources, upload rings, the GPU upload heap, readbacks, query heaps. Decide:
   - **Barriers:** explicit, recorded once and translated per backend (D3D12 states and enhanced barriers; Vulkan
     sync2 layouts and access masks).
   - **Binding:** keep the bindless model the shaders use (`gtex`, `gcube`, `gvol`, eight samplers, two constant
     blocks, `gmem`/`gphys`), mapped to descriptor heaps (D3D12) and descriptor indexing or descriptor heap (Vulkan).
   - **Shaders:** DXC to DXIL and to SPIR-V from the same HLSL. The D3D12 path still uses fxc (DXBC) by default; moving
     it to DXC first keeps one compiler.
   - **Pipelines:** one key (`PipelineKey`), one disk cache per backend; the manifest and seeds stay shared.
3. **Move the D3D12 renderer onto the interface** with no behaviour change. Prove it with the phase 0 frames,
   byte-identical where possible. Do it in slices: upload and texture creation, then surfaces and resolves, then draws,
   then queries and predication, then the frame end and present.
4. **Implement the Vulkan backend of the interface** from `src/gpu/vk`'s pieces. Bring the slices up in the same order,
   diffing against D3D12 at each step.
5. **Presenter and host integration on Vulkan:**
   - Vulkan swap chain and frame pacing;
   - `REGPU_ASYNC_PRESENT`;
   - overlay and gamma;
   - the host presenter API (a `VkImage` handed over in place of an `ID3D12Resource`);
   - Wayland and X11 on Linux.
6. **The rest:**
   - Upscalers: FSR 3/4 have Vulkan backends; DLSS and XeSS have Vulkan SDKs.
   - Post AA, VR (OpenXR has a Vulkan binding) and triple screen.
   - Hyper mode.
   - Draw workers on Vulkan secondary or parallel command buffers.
7. **Linux native build:**
   - Toolchain: XenonRecomp output already builds with clang; replace the Win32-only pieces (window, threads, DLL
     loading) behind the host.
   - Distribute in the Steam Linux Runtime 4.0 SDK.

## Housekeeping before starting

- **FM3/PGR3 on `main`:** test FM3 and PGR3 against `main` (D3D12) first, so later diffs aren't blamed on the
  conversion. Then point their submodules at `main` with an empty ReGPU patch stack.
- **FM4's features:** decide which of [fm4-features-to-port.md](fm4-features-to-port.md) matter. Port them before
  phase 3 if they are D3D12 renderer changes (cheaper than porting twice).
- **Local build:** see `G:\claude code projects\ReGPU-V\work\scripts\README.md` on this machine. It covers the build
  scripts, the dependencies in `work\deps`, and how to run `regpu_vkprobe`.
