# ReGPU
<img width="2100" height="1500" alt="reGPU society" src="https://github.com/user-attachments/assets/5554e4b0-5b9f-4e8d-9839-314dc3e4eb7f" />

The Xbox 360 GPU (Xenos), on Vulkan, for native PC ports of Xbox 360 games made with
[XenonRecomp](https://github.com/hedge-dev/XenonRecomp).

ReGPU takes the GPU's half of a port off your hands. The recompiled game talks to it exactly as it
talked to the console: it writes command packets into a ring buffer and calls the kernel's `Vd*` video
functions. ReGPU runs the command processor, translates the Xenos shader microcode straight to SPIR-V
(or through HLSL and DXC, the reference path), emulates
EDRAM, resolves, MSAA and the texture formats, presents the frames, and can upscale them (FSR 1,
optionally FSR 3/4, DLSS and XeSS).

It came out of a working port, Forza Motorsport 2, where it renders the whole game. Everything
specific to that game has been removed; what remains is the console's behaviour, plus hooks for the
workarounds a particular game needs.

Since November 2026 Vulkan is its one render backend. Forza Motorsport 3 (braaap-stututu/FM3-ReGPU) and
Forza Motorsport 4 (braaap-stututu/Fuzzy-Rittersport-4-ReGPU-testing, branch `regpu-vulkan`) run on it, at or
above their Direct3D 12 builds' speed. [`docs/vulkan.md`](docs/vulkan.md) has the renderer, what each game needed
and the speed work with its numbers; [`docs/ports`](docs/ports) the ports' full records (including the Steam Deck
briefing); [`docs/research`](docs/research) the background research. The last Direct3D 12 renderer is the tag
`d3d12-reference` and the branch `legacy-d3d12`.

## Requirements

- **XenonRecomp.** ReGPU is for XenonRecomp ports and builds against it: it uses XenonRecomp's
  `PPCContext` and calling convention to provide the console's `__imp__Vd*` video exports and to call
  the game's graphics interrupt. Your recompiled game links them like any other import. (Ports made
  some other way are not supported.)
- Windows 10/11 (Linux is the next step: the renderer is Vulkan, the window and some tools are Win32 so far), a
  Vulkan 1.3 GPU and driver (dynamic rendering, synchronization2, descriptor indexing, robustness2's null
  descriptors), clang-cl (as XenonRecomp's output needs), CMake 3.25+, Ninja, Python 3 and DXC with SPIR-V output
  (a build step compiles and embeds ReGPU's own compute shaders: the Vulkan SDK's `Bin/dxc.exe`, found on its own, or
  `REGPU_DXC_EXE`). Vulkan is loaded at run time (volk) and the headers, volk and VMA are in `thirdparty/`: no SDK
  is needed to build beyond DXC.
- The AMD FidelityFX SDK (MIT) for the FSR 1 shader headers. FSR 3/4 needs its signed Vulkan runtime DLLs, DLSS
  NVIDIA's DLSS SDK and XeSS Intel's XeSS SDK; all are off by default.

## Building

ReGPU is a static library your port links. In your port's CMake:

```cmake
set(REGPU_XENONRECOMP "${CMAKE_SOURCE_DIR}/thirdparty/XenonRecomp")   # XenonRecomp checkout
set(REGPU_PPC_DIR     "${CMAKE_SOURCE_DIR}/out/ppc")                  # your game's XenonRecomp output (ppc_config.h)
set(REGPU_FFX         "${CMAKE_SOURCE_DIR}/thirdparty/fidelityfx/Kits/FidelityFX")
add_subdirectory(thirdparty/ReGPU)
target_link_libraries(game PRIVATE regpu)
```

The renderer is Vulkan ([`docs/vulkan.md`](docs/vulkan.md)); it was Direct3D 12 up to the tag `d3d12-reference`
([`docs/vulkan-conversion-plan.md`](docs/vulkan-conversion-plan.md)).

Options: `REGPU_FSR3` (with the FidelityFX signed Vulkan DLLs), `REGPU_DLSS` and `REGPU_DLSS_DIR`, `REGPU_XESS_DIR`.

`REGPU_FSR1_CS` takes a pre-built `fsr1_cs.h` (the header `tools/embed_hlsl.py` writes into the build's
`generated` folder). With it the build copies that header and needs no Python, DXC or FidelityFX headers,
which helps cross-builds from Linux. The header has to come from the same `src/gpu/fsr1.hlsl`: rebuild it
when that file or its entry points change.

Ship DXC's `dxcompiler.dll` and `dxil.dll` (the DirectXShaderCompiler release, MIT and University of Illinois
licences, or the Vulkan SDK's) beside your executable: the translated shaders are compiled to SPIR-V with it at run
time (and cached on disk, `Config::userDir`). With `REGPU_FSR3`, also `amd_fidelityfx_loader_vk.dll` and
`amd_fidelityfx_upscaler_vk.dll` (the FidelityFX SDK's `signedbin`).

To check that ReGPU builds on its own: `cmake --build build --target regpu_linkcheck`.

## Hooking it up

Everything is in [`include/regpu/regpu.h`](include/regpu/regpu.h). Before the game initialises
its graphics:

```cpp
#include <regpu/regpu.h>

regpu::Config& config = regpu::GetConfig();
config.width = 1920; config.height = 1080;
config.msaa = 4;
config.userDir = savesFolder / "regpu";      // shader and pipeline caches
config.windowTitle = L"My Game";

regpu::Host& host = regpu::GetHost();
host.base = guestMemoryBase;                 // the guest's 4 GB address space
host.alloc = [](bool physical, uint32_t size, uint32_t align) { return MyGuestAlloc(physical, size, align); };
host.createThread = [](const char* name, void (*body)(PPCContext&)) { MyCreateHostGuestThread(name, body); };
host.currentPcr = [] { return MyCurrentThread().pcr; };
host.guestCall = [](PPCContext& ctx, uint32_t address) { MyGuestCall(ctx, address); };
host.interruptLockAcquire = [](int cpu) { MyInterruptLock(cpu); };
host.interruptLockRelease = [] { MyInterruptUnlock(); };

regpu::Start();   // window, device, command processor
```

Your runtime already has these: they are the pieces any XenonRecomp port builds for its own threads
and memory. ReGPU never touches the rest of the kernel (files, threads, XAM, audio), which stays yours.

ReGPU's threads run until the process exits. A host that frees the guest's memory or threads while
the process lives on (a shutdown) calls `regpu::StopInterrupts()` first: when it returns, no graphics
interrupt callback is running in guest code and none will start, and the periodic shader scan of
guest memory has ended (it waits up to `waitMs` in all, 500 ms by default, for a callback and a scan
pass in flight, and returns false if either was still running then). ReGPU is not idle afterwards:
every thread acting on work already queued can still read or write guest memory for it, among them
the command processor, the draw workers, the fence writer and the texture prehash workers (and, with
their debug switches, the overwrite checker and the debug server), and the host callbacks (`onSwap`,
`resolveWriteback` and the others) can still be called, among them `logThreadStacks` from the hang
watchdog once no frame has been swapped for 10 s. Without the call nothing changes.

The renderer currently assumes the game draws at 1280x720, as most Xbox 360 games do (see
[`docs/xenos-notes.md`](docs/xenos-notes.md)).

### Your own device and presenter

By default ReGPU creates its own Vulkan instance and device (on the high-performance GPU; `REGPU_ADAPTER` picks
another) and presents through its own swap chain, on its own window or on `Config::hostWindow`. A host that already
draws (its own menus, overlays, a recorder) can hand over its device, or the device and the presenting: a host
device without a host presenter is fine, a host presenter needs the host device and queue.

| Option | Effect |
|---|---|
| `Config::hostInstance`, `hostPhysicalDevice`, `hostDevice`, `hostQueue`, `hostQueueFamily`, `hostDeviceExtensions` | Your `VkInstance`, `VkPhysicalDevice`, `VkDevice` and a `VkQueue` of `hostQueueFamily` (graphics and compute): ReGPU renders on them instead of creating its own, and chooses no GPU (`REGPU_ADAPTER` does nothing). Set all four handles or none (some alone end the process at `Start` with a FATAL line in `regpu.log`). The device must be Vulkan 1.3 with ReGPU's baseline enabled (`regpu.h` lists it: dynamic rendering, synchronization2, maintenance4, timeline semaphores, descriptor indexing with update-after-bind, robustness2's null descriptors); `hostDeviceExtensions` lists the device extensions you enabled, and ReGPU uses the optional ones it finds there (`VK_KHR_swapchain`, `VK_EXT_conditional_rendering`, `VK_EXT_external_memory_host`, `VK_EXT_memory_budget`, ...). ReGPU submits on `hostQueue` under its own lock: don't submit on that queue from another thread at the same time. `REGPU_VK_VALIDATION` then turns nothing on (it is yours to enable). |
| `Host::presentImage` | You present. ReGPU opens no window and no swap chain, and hands you each finished picture (`PresentImage`: a `VkImage`, R8G8B8A8, already encoded for display, its layout, and the optional progress overlay with its place): once for every guest swap once the renderer has started (from the first draw on), after that frame's commands were submitted to the queue (not waited for), and for each hold and first-boot screen. Frames and first-boot screens come on the renderer's thread; a hold comes on whichever thread waits for a shader or pipeline, which with `REGPU_DRAW_WORKERS` can be a draw piece worker. Return the image and overlay in the layouts given, with your work on them submitted on `Config::hostQueue` (the queue ReGPU renders on) before the callback returns: ReGPU hands over no semaphore, so that queue (with your barriers from all commands and memory writes) is what orders your reads after its writes and its next writes after your reads. Do not keep either pointer past the call: ReGPU writes both again for the next picture. It therefore needs the host device (`Config::hostDevice` and the rest); without them the process ends at `Start` with a FATAL line in `regpu.log` (not with `REGPU_HEADLESS` or `REGPU_NO_HOST_PRESENT=1`, which never call it). `regpu::SetFullscreen` does nothing then (the window is yours); `SetOutputSize` changes the size of the pictures you get. `REGPU_NO_HOST_PRESENT=1` (read at `Start`; flipping it later does nothing) ignores the callback, and ReGPU presents itself: it creates its own swap chain on `Config::hostWindow` (or its own window), which fails if you already have a swap chain on that window. |

### Game-specific fixes

When a game needs something the console did that isn't the GPU's job, or a workaround for one of
its shaders, use the optional hooks instead of changing ReGPU:

| Hook | For |
|---|---|
| `rewriteShader` | Edit a shader's translated HLSL before it compiles, and tag it |
| `onDraw` | Set four `float4` constants (`vuser`) for draws with tagged shaders |
| `resolveWriteback` | Copy a resolved image back to memory because the game reads it on the CPU (screenshots, thumbnails, focus readback) |
| `staticShaders`, `bootScreen` | Compile every shader in the game's files on first boot, with your own progress screen |
| `onChar`, `onSwap` | Typed text (cheat codes), a callback per frame |
| `logThreadStacks`, `currentStack`, `slowFrame`, `cpuUsage` | Diagnostics |
| `SetGameWidescreen(true)` | The game itself draws wider than 16:9, so ReGPU leaves the field of view alone |

## Debugging

Start the game with `REGPU_DEBUG=1` and drive it from outside with
[`tools/regpuctl.py`](tools/regpuctl.py): press buttons, take screenshots, capture a frame, edit
shaders live, flip `REGPU_*` switches. [`CLAUDE.md`](CLAUDE.md) is the playbook.
F12 in the game window captures the next frame as `regpuctl.py capture` does (`frame.png` in the
working directory, its textures and resolves under `out/` there); `REGPU_NO_CAPTURE_KEY=1` turns the
key off, for a game that uses F12 itself, and leaves the debug server's capture working.
[`tools/xbdm.py`](tools/xbdm.py) and [`tools/xbox_fb.py`](tools/xbox_fb.py) take reference
screenshots and memory reads from a real (modded) console over XBDM.

## Credits and licences

- [XenonRecomp](https://github.com/hedge-dev/XenonRecomp) (MIT), which ReGPU builds against.
- [Xenia](https://github.com/xenia-project/xenia) (BSD 3-Clause)'s research into the Xenos GPU informed this
  renderer. A few routines follow Xenia's closely: the PWL gamma conversion (`renderer.cpp`), the texture
  tiling and packed-mip offsets (`texture.cpp`). For those:

  > Copyright (c) 2015, Ben Vanik. All rights reserved.
  >
  > Redistribution and use in source and binary forms, with or without modification, are permitted
  > provided that the following conditions are met:
  > - Redistributions of source code must retain the above copyright notice, this list of conditions
  >   and the following disclaimer.
  > - Redistributions in binary form must reproduce the above copyright notice, this list of
  >   conditions and the following disclaimer in the documentation and/or other materials provided
  >   with the distribution.
  > - Neither the name of the project nor the names of its contributors may be used to endorse or
  >   promote products derived from this software without specific prior written permission.
  >
  > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  > IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  > FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
  > CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  > DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  > DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  > IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
  > OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
- AMD FidelityFX SDK (MIT); NVIDIA DLSS SDK (its own licence, only with `REGPU_DLSS`); Intel XeSS SDK (its own
  licence, only with `REGPU_XESS_DIR`); DirectXShaderCompiler (`dxcompiler.dll`, `dxil.dll`: University of Illinois
  and MIT licences); Vulkan-Headers (Apache 2.0 / MIT), volk (MIT), Vulkan Memory Allocator (MIT) in `thirdparty/`.

ReGPU is released under the [MIT licence](LICENSE).
