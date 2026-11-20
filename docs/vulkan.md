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
| `regpu_vkdraw` | A draw through the translated shaders' bindings: set 0's constants, gmem, gphys and a bindless texture. |
| `regpu_vkpasses` | A compute and a graphics pass from HLSL, their layouts from SPIR-V, transient sets, read back. |

## Not done yet

- **VR (OpenXR):** its session is still the D3D12 one (`XR_KHR_D3D12_enable`); builds take the stub (`REGPU_VR` logs
  that it is not available) until it moves to `XR_KHR_vulkan_enable2`.
- **DLSS and XeSS** were converted without their SDKs on the converting machine: build them once with the SDKs.
- **Linux:** the renderer is Vulkan, but the window, the overlay text (GDI), threads and a few file paths are Win32.
- **Run with a game:** see the plan's status.
