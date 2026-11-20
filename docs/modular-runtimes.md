# Modular CPU-side integration: runtimes and consoles

Status: **design, 20 Nov 2026.** Nothing here is built yet. This follows the user's direction:
- **ReGPU will serve more than one console.** A PS3 recompilation on ReGPU is in the works.
- **Xbox 360 ports use ReXenon** as the recompiler, unless something better comes along.
- **Switching easily.** Moving a port from XenonRecomp, "xenosrecomp", ReXGlue and the like to ReGPU plus ReXenon
  should be easy.

Read it together with [vulkan-conversion-plan.md](vulkan-conversion-plan.md); the two are sequenced together at the
end.

## What we have today (measured)

**How ReGPU is tied to the CPU side.** The tie is small and in three places:
- **Exports.** 21 guest exports, all in `src/gpu/gpu.cpp`:
  - `VdSwap`, `VdInitializeRingBuffer`, `VdSetGraphicsInterruptCallback`, `VdQueryVideoMode`, and so on.
  - Each is defined with `PPC_FUNC(__imp__Vd...)` through `src/kernel/imports.h`.
- **The `Host` callbacks** in `regpu.h`:
  - `base`, `alloc`, `createThread`, `currentPcr`, `guestCall`, the interrupt lock, and the write watch.
  - Three call sites use them.
- **The build.** ReGPU is compiled with the game's own `ppc_config.h` and XenonRecomp's `ppc_context.h`
  (`REGPU_PPC_DIR`, `REGPU_XENONRECOMP`), or with ReXGlue's `rex/ppc/context.h` (`REGPU_REXGLUE`). So ReGPU is
  rebuilt per game, against that game's recompiler headers.
- **A second way in:** ReXGlue's kernel implements the `Vd*` exports itself and calls `include/regpu/driver.h`.
  Glue2GPU is the plugin built on that.

**The ports carry their own runtime.** FM3, PGR3 and FM4 (and FM2/PGR4, which they come from) each hold their own
copy of the same Xbox 360 runtime, all from FM2:

| Port | `src/kernel` + memory, loader, MMIO, indirect calls, modules | `gpu_host.cpp` (ReGPU glue) |
|---|---|---|
| FM3 | 8,026 lines | 698 |
| PGR3 | 8,155 lines | 752 |
| FM4 (`fm4-perf`) | 9,092 lines | 149 |

- The kernel file lists are identical. FM3 and PGR3 differ in only 14 kernel files, and those are local fixes.
- **This copied runtime, not ReGPU, is what makes switching expensive.** A fix to `KeWaitForSingleObject` or XAM
  in one port never reaches the others.
- **An example from FM3 session 47.** FM3's main thread called the game's app tick about 1.1 million times a second
  between frames, wasting up to 98% of a core.
  - The fix (`FM3_TICK_WAIT`) is a hook in FM3's runtime.
  - ReGPU's VdSwap limiter had hidden the spin in some boots.
  - Other titles likely spin the same way. A shared runtime would make the tool to find and fix it available to
    every port.

## The target structure

```
ReGPU (repo)
├─ core/                console-neutral: Vulkan device and presenter, upscalers and post AA, DXC→SPIR-V and the
│                       pipeline/shader caches, the upload ring, bindless textures, live switches, log, debug server,
│                       frame pacing, settings
├─ consoles/
│   ├─ x360/            Xenos: PM4 command processor, the Vd* exports, ucode→HLSL, EDRAM and resolves, Xenos texture
│   │                   formats and tiling (what src/gpu is today, minus core)
│   └─ ps3/             RSX (later): FIFO and NV40-class methods, vertex/fragment program translation, local and
│                       IO-mapped memory, tiles and zcull, the cellGcm exports
├─ runtimes/
│   ├─ x360/            the shared Xbox 360 runtime, from the ports' copies: kernel HLE (Ke/Mm/Io/Rtl/Xam/crypt),
│   │                   threads and objects, XEX loader, VFS, MMIO, indirect calls, recompiled modules
│   └─ adapters/        one small header per recompiler's calling convention (rexenon.h, xenonrecomp.h, rexglue.h)
└─ plugins/glue2gpu     ReXGlue ports on ReGPU (as now)
```

**The rules:**
- **The core never includes a console or recompiler header.** A console frontend never includes a recompiler header.
  Only an adapter header does, and it is compiled in the game's own source files, never in the ReGPU library.
- **One ReGPU library build per console.** It no longer needs to be rebuilt per game: the game-specific `ppc_config.h`
  only reaches the adapter.
- **Guest memory stays a direct pointer** (`base + address`, big-endian loads). The hot paths (the command processor,
  fetch and write-back) gain no indirection.

## The runtime interface (sketch)

This replaces the `PPCContext&` parts of `Host`. It is shared by every console with a PowerPC guest (360 and PS3;
the PS3's SPUs would need their own adapter if a title's GPU work runs there).

```cpp
namespace regpu {
// An opaque guest call frame: the adapter knows what is inside it.
struct GuestFrame;

struct GuestRuntime {
    uint8_t* base;                                        // guest address A is at base + A
    uint32_t (*alloc)(MemoryKind kind, uint32_t size, uint32_t align);
    void     (*createThread)(const char* name, void (*body)(GuestFrame&));
    uint64_t (*call)(uint32_t address, const uint64_t* args, int count);  // a guest function, by address
    void     (*interruptEnter)(int cpu);                  // the console's interrupt context
    void     (*interruptLeave)();
    uint32_t (*currentThreadBlock)();                     // 360: the KPCR; PS3: the PPU thread id
    // optional: write watch, thread CPU use, stacks: as in Host today
};

// What a console frontend exports to the guest: name, then a handler that reads its arguments.
struct GuestArgs {                                        // filled by the adapter's thunk
    uint64_t arg(int i) const;                            // r3-r10, then stack slots, per the console's ABI
    void     ret(uint64_t value);
};
using ExportFn = void (*)(GuestArgs&);
const ExportTable& Exports();                             // {"VdSwap", fn}, ...
}
```

**An adapter** is one header (`runtimes/adapters/rexenon.h`) that the game includes in a single source file. It
expands `REGPU_BIND_EXPORTS()` into:
- one `PPC_FUNC(__imp__VdSwap) { GuestArgs a{ctx, base}; regpu::x360::VdSwap(a); }` per entry in the table;
- `GuestRuntime` functions that build that recompiler's context for `call` and `createThread`.

The guest's existing thunks and import names keep working.

| Recompiler or runtime | Adapter |
|---|---|
| ReXenon | `rexenon.h`, the reference. XenonRecomp's `PPCContext` and `PPC_FUNC`, plus ReXenon's register options and `PPC_UNIMPLEMENTED` traps. |
| XenonRecomp | `xenonrecomp.h`. ReXenon keeps XenonRecomp's format, so this is mostly the same file. It exists for ports that haven't moved yet. |
| ReXGlue | Two choices: keep the `driver.h` route (ReXGlue's kernel owns the exports, as Glue2GPU does), or use `rexglue.h` to bind ReGPU's export table into rex's import table. |
| Other C++-output recompilers ("xenosrecomp" and the like) | A new adapter header of roughly 100-200 lines, if the output is a C++ register context with one function per guest function. |
| An interpreter or JIT (an emulator's CPU core) | `GuestRuntime` and the export table are enough: the emulator's HLE dispatcher calls `Exports()` by name. |

## Switching

**At build time** (CMake, set once per port):
- `REGPU_CONSOLE` = `x360` | `ps3`. This picks the frontend and the runtime folder.
- `REGPU_RECOMPILER` = `rexenon` (the default for 360) | `xenonrecomp` | `rexglue`. This picks the adapter header and
  the include paths. FM3's `FM3_RECOMPILER` (rexenon / xenonrecomp) is the working model; it moves into ReGPU as
  `regpu_add_game(<target> CONSOLE x360 RECOMPILER rexenon PPC_DIR <out/ppc> ...)`.
- `REGPU_RUNTIME` = `shared` (the default: `runtimes/x360`) | `own` (the port keeps its own kernel and only uses the
  adapter). `own` is how an existing port moves over one step at a time.

**At run time:**
- The CPU runtime is linked in, so it cannot change at run time.
- What does change at run time are the console's quirks and the host GPU profile, as now: `REGPU_*` live switches,
  and a profile for weak GPUs.

**Per-console capabilities.** A frontend declares what it uses from the core, and the core makes no Xenos
assumptions:

| Aspect | Xenos | RSX |
|---|---|---|
| Render targets | EDRAM and resolves | Surfaces in local memory, sampled directly |
| Memory | 512 MB unified | 256 MB local plus IO-mapped main memory |
| Shaders | Unified Xenos microcode | NV40 vertex and fragment programs |
| Command stream | PM4 packets | FIFO methods |

## Moving a 360 port to ReGPU + ReXenon

These steps go in the README once built.
1. **From XenonRecomp:** re-run the recompile with ReXenon, using the same TOML.
   - Its `allow_unresolved = false` lists every gap XenonRecomp silently left out. Fix those first.
   - Point `regpu_add_game` at the new output, with `RECOMPILER rexenon`.
2. **From a port with its own copied kernel** (FM2's family):
   - Start with `REGPU_RUNTIME=own` and the adapter. Nothing else changes.
   - Then delete kernel files one at a time as `runtimes/x360` covers them, comparing a boot each time.
3. **From ReXGlue:**
   - Glue2GPU already runs ReXGlue ports on ReGPU, unchanged.
   - Moving fully means recompiling with ReXenon and taking `runtimes/x360` in place of rex's kernel.
   - That is the same path as step 1, plus mapping rex's import names. A table of those will be written once a
     ReXGlue port is tried.
4. **Checks** (in `tools/`, to be written):
   - `regpu_port_check`: lists unresolved imports against `runtimes/x360` and the export table.
   - A boot test: run to the first VdSwap headless.

## Stages and sizes

| Stage | What | Size (estimate) | When |
|---|---|---|---|
| R1 | The `GuestRuntime` and `GuestArgs` interface, plus the export table. `gpu.cpp`'s 21 exports go through it. Add `rexenon.h` and `xenonrecomp.h`. ReGPU builds once without a game's `ppc_config.h`. | ~1 k lines changed | Beside the Vulkan conversion: it is CPU-side and touches no rendering |
| R2 | `runtimes/x360` from FM3's copy (the newest of the family). Merge in PGR3's and FM4's fixes, the same way the ReGPU forks were merged. `REGPU_RUNTIME=own/shared`. | ~9 k lines moved, 14+ files merged | After R1 |
| R3 | `regpu_add_game`, the port-check tool, and the migration guide. FM3 and PGR3 move to the shared runtime. | Medium | With Vulkan phase 5 (host API and ports), so the ports take one set of breaking changes, not two |
| R4 | The split into core and `consoles/x360` (directories and CMake, no behaviour change) | Large but mechanical | After Vulkan phase 6, never during the in-place conversion (the plan's "same structure" rule) |
| R5 | `consoles/ps3` (RSX) and its runtime adapter | Large | When the PS3 project's needs are known |

## Open questions

- **The PS3 project:**
  - Where is it, who runs it, and which recompiler does it use?
  - Does it already link ReGPU, and if so, where does it hook RSX?
  - The design of R5 depends on the answers. It isn't public on GitHub under "ReGPU".
- **"xenosrecomp":** which project this is (it isn't found under that name), so its adapter can be checked.
- **Where `runtimes/x360` lives:** in the ReGPU repo for now (the user asked for new files to stay here). It could
  become a repository of its own once a second console runtime exists.
