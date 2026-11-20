# The Vulkan backend (experimental)

ReGPU draws with Direct3D 12. `src/gpu/vk` is a second backend that draws the same command stream with Vulkan. It
came from the Forza Motorsport 2 port (tankyx's renderer, FM2 `ecf4675`, where it is the default on Linux since
0.12.0). It is the starting point for moving ReGPU to Vulkan, not a finished replacement.

## Building and choosing it

- Configure with `-DREGPU_VULKAN=ON`. `REGPU_VULKAN_HEADERS` is the folder that holds `vulkan/vulkan.h` (a
  [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) checkout's `include`). If it is empty, CMake uses
  `find_package(Vulkan)`. Vulkan itself is loaded at run time (`vk_loader.cpp`), so nothing links against it.
- Shaders go to SPIR-V through `dxcompiler.dll` and `dxil.dll` from a
  [DirectXShaderCompiler release](https://github.com/microsoft/DirectXShaderCompiler/releases). The port ships them
  beside its executable. The Windows SDK's copy is built without SPIR-V output and fails with "SPIR-V CodeGen not
  available".
- At run time, set `Config::renderer = regpu::Config::Renderer::Vulkan`, or `REGPU_RENDERER=vulkan` (`d3d12`
  overrides the other way). Without `REGPU_VULKAN` in the build, Direct3D 12 is used either way.
- Smoke test: `cmake --build <build> --target regpu_vkprobe`, then run it with the two DLLs beside it. It opens the
  device and compiles the translator's preamble to SPIR-V the way the backend does.

## How it fits

The command processor calls the same renderer entry points either way. When the backend is enabled,
`RendererSetShader`, `RendererOnShaderLoad`, `RendererOnDraw` and `RendererOnSwap` forward to `renderer_vk.cpp`
(`REGPU_VK_IF_ENABLED` in `renderer.cpp`). The Direct3D 12 lookahead and occlusion queries stand down, so occlusion
queries answer "visible".

The backend uses the shared parts of ReGPU:

- the microcode decoder and the HLSL translator (`ucode.cpp`, `hlsl.cpp`);
- texture decoding (`texture.cpp`);
- the index expansion (`indices.cpp`);
- the constant block layout (`constants.h`, which must stay byte for byte equal to `hlsl.cpp`'s cbuffers);
- the overlay, display gamma and compile counters (`renderer.h`).

It presents through its own Vulkan swap chain when it can claim the window (`PresenterClaimWindow`). Otherwise it
reads each frame back and hands it to the Direct3D 12 presenter (`PresenterExchangeImage`).

## What was changed from FM2's copy

- Switches `FM2_*` became `REGPU_*`. The FM2-only parts were left out: the boot screen's game textures, the static
  shader list (`Host::staticShaders` is used instead) and the dealer easter egg constants.
- Pipeline layout: ReGPU's translator declares eight samplers (s0-s7: the four guest samplers, then linear wrap at
  2x, 4x, 8x and 16x anisotropy) and a 3D texture array (`gvol`, space3). `vk::AdaptHlsl` turns the samplers into one
  `gSamplers[8]` array binding, and the layout has eight sampler slots and a fourth bindless set.
- Texture info: mirrored addressing moved to `tinfo.x` bits 28-31, where ReGPU's shaders read it (bits 20-22 are the
  anisotropy there). The pixel position (`pinfo.z` bit 16) and point snap (bit 17) follow the Direct3D 12 path's defaults.

## Known gaps

These are not done yet. They are the work list for the conversion:

- **3D textures:** set 3 is bound to the 2D array, and the backend never sets up a 3D fetch (`tinfo.w` is 1 or 3).
- **Vertex fetch bounds:** `vf.w` holds the buffer size in bytes, where ReGPU's shaders expect dwords. The bounds
  clamp is four times looser than on Direct3D 12; nothing reads out of range because of it.
- **Features the Direct3D 12 path has since gained:** none of these exist on this backend. Each is an existing
  `REGPU_*` feature in `renderer.cpp` and needs Vulkan support of its own.
  - FM3/PGR3's draw thread, workers and async present.
  - Occlusion queries, predicated tiling merges, resolve bands and enclosing resolves.
  - Temporal upscalers (FSR 3/4, DLSS, XeSS), post AA, VR and triple screen.
  - Hyper mode.
  - The host device and presenter API (`Config::hostDevice`, `Host::presentImage`).
- **Not tested with a game in ReGPU.** It builds, links, opens a device and compiles the shader preamble
  (`regpu_vkprobe`). FM2 ran it against its own copy of the renderer, before these merges.
