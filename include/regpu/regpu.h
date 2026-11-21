#pragma once
// ReGPU: the Xbox 360 GPU (Xenos) for XenonRecomp ports, on Direct3D 12.
//
// The game fills in Config and Host before calling Start, and implements nothing else of the GPU:
// ReGPU provides the console's Vd* video exports (PPC_FUNC __imp__Vd...), runs the command processor
// and the graphics interrupt, translates shaders, emulates EDRAM and resolves, and presents the frames.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct PPCContext; // XenonRecomp (ppc_context.h)

namespace regpu
{
// Display and quality settings. Read at start and, for most fields, again each frame.
struct Config
{
    uint32_t width = 1280, height = 720; // output resolution
    uint32_t aspectW = 0, aspectH = 0;   // forced aspect ratio (0 = from the resolution)
    bool fullscreen = false;             // borderless fullscreen on the primary display
    bool vsync = true;
    uint32_t maxFps = 60;                // frame cap
    bool enhance = false;                // enhanced lighting post pass (experimental)
    uint32_t msaa = 1;                   // samples per pixel: 1 (off), 2, 4 or 8
    // Upscaling: the 3D scene renders smaller and is upscaled before post-processing and the
    // HUD, which stay at full resolution. Fsr3, Fsr4 and Dlss need REGPU_FSR3 / REGPU_DLSS builds. The temporal ones
    // (Fsr3, Fsr4, Dlss) run only on frames with a 3D scene of their own, and need MSAA off: with one chosen the scene
    // draws at 1x. Fsr4: FSR 4 where the GPU offers it (RDNA 4), else FSR 3. Xess: Intel XeSS (libxess.dll beside the
    // exe), else FSR 1.
    enum class Upscaler : uint32_t { Off, Fsr1, Fsr3, Dlss, Fsr4, Xess };
    Upscaler upscaler = Upscaler::Off;
    uint32_t quality = 1;   // 0 native (anti-aliasing only), 1 quality, 2 balanced, 3 performance, 4 ultra performance
    float sharpness = 0.5f; // 0-1
    // Anti-aliasing filters (FM3 port, session 36). postAA: a filter on the finished 3D scene, before the title's own
    // post-processing and HUD (FXAA, SMAA, CMAA 2: need a REGPU_AA_DIR build; TAA: the motion vectors and jitter of the
    // temporal upscalers, so not together with DLSS or FSR 3). ssaa: the game renders at this multiple of the shown size
    // and each frame is filtered down at the present (1 = off; 1.5 and 2 sensible). eqaa: AMD's enhanced quality AA
    // (more coverage than colour samples) where the driver offers it as an MSAA quality level.
    enum class PostAA : uint32_t { Off, Fxaa, Smaa, Cmaa2, Taa };
    PostAA postAA = PostAA::Off;
    float ssaa = 1.0f;
    bool eqaa = false;
    std::filesystem::path userDir = "."; // shader and pipeline caches, dumps, regpu.log, regpu-bad-pipelines.txt
    std::wstring windowTitle = L"ReGPU";
    // A window the host already owns (an HWND): ReGPU presents into it instead of opening its own.
    // The host then handles closing, resizing and fullscreen; windowTitle and fullscreen are unused.
    void* hostWindow = nullptr;
    // A Vulkan device the host already has: ReGPU renders on it instead of creating its own, and chooses no GPU.
    // hostInstance (VkInstance), hostPhysicalDevice (VkPhysicalDevice), hostDevice (VkDevice) and hostQueue (VkQueue,
    // of hostQueueFamily, with graphics and compute) all, or none (some alone end the process at Start with a FATAL
    // line in regpu.log). The device must be Vulkan 1.3 with dynamic rendering, synchronization2, maintenance4,
    // timeline semaphores, descriptor indexing (update-after-bind, partially bound, runtime arrays), robustness2's
    // null descriptors and VK_EXT_robustness2, and the instance VK_KHR_surface (+ the platform's) when ReGPU presents.
    // hostDeviceExtensions lists the device extensions the host enabled: ReGPU uses the optional ones it finds there
    // (VK_KHR_swapchain, VK_EXT_conditional_rendering, VK_EXT_external_memory_host, VK_EXT_memory_budget, ...).
    // ReGPU submits on hostQueue under its own lock: the host must not use that queue from another thread at the same
    // time. The host keeps all of it alive while ReGPU runs.
    void* hostInstance = nullptr;
    void* hostPhysicalDevice = nullptr;
    void* hostDevice = nullptr;
    void* hostQueue = nullptr;
    uint32_t hostQueueFamily = 0;
    std::vector<std::string> hostDeviceExtensions;
    // Upload ring per frame (constants, vertices, textures), in MB; 0 is ReGPU's default (64 with
    // direct fetch). About 11 KB a draw goes to constants alone: titles with thousands of draws need
    // more, or their last draws are dropped. REGPU_RING_MB overrides it.
    uint32_t uploadRingMB = 0;
    // Old CPUs: renderer optimisations that help fast PCs but cost slow ones are off (the staged draw
    // constants: an FX-8350 lost time copying them).
    bool lowPerformance = false;

    // The scene's resolution as a fraction of the output (1 when upscaling is off).
    float SceneScale() const;
    // The size the game renders at: the resolution, narrowed or shortened to the forced aspect
    // ratio (the rest of the screen is black bars).
    void RenderSize(uint32_t& w, uint32_t& h) const;
    // The size the frame is shown at: RenderSize without supersampling.
    void ShownSize(uint32_t& w, uint32_t& h) const;
};

// A shader found in the game's files or memory: its microcode as in guest memory (big endian).
struct ScannedShader
{
    bool pixel;
    std::vector<uint32_t> code;
    uint32_t address = 0; // guest address of the microcode (0: not from guest memory)
};

// A resolve (an EDRAM copy to memory), as offered to Host::resolveWriteback.
struct ResolveCopy
{
    uint32_t base;          // physical address of the image (its top-left; tiles land in one image)
    uint32_t width, height; // the whole image, in game pixels
    uint32_t format;        // the Xenos colour format it is stored as (6: 8888)
    bool depth;             // a depth copy (written as the console stores D24FS8 in 32-bit words)
    uint32_t samples;       // the EDRAM surface's MSAA: 1, 2 or 4
    int32_t top, bottom;    // the rows this resolve covers (a tile of the image)
};

// Resolved images stay on the GPU; one the game reads on the CPU must also be written to memory.
struct Writeback
{
    enum class Mode { None, Now, Deferred };
    // Now: written before the game continues (a stall). Deferred: written when this frame's slot comes
    // round again, a frame or two late, without a stall (for something read every frame).
    Mode mode = Mode::None;
    // Deferred only: the part of the image to write (game pixels); the default is all of it.
    uint32_t x0 = 0, y0 = 0, x1 = ~0u, y1 = ~0u;
};

// A finished picture, as handed to Host::presentImage.
struct PresentImage
{
    enum class Kind : uint32_t { Frame, Hold, Boot };
    Kind kind;              // Frame: a guest swap. Hold: the last picture again, while ReGPU waits. Boot: the first-boot screen.
    uint64_t frame;         // ReGPU's frame number
    uint64_t image;         // a VkImage, R8G8B8A8_UNORM, already encoded for display; 0: a Frame with nothing presentable
    uint32_t width, height; // the image's size
    uint32_t imageLayout;   // the VkImageLayout it is in
    uint64_t overlay;       // optional VkImage, R8G8B8A8_UNORM: ReGPU's progress text
    uint32_t overlayWidth, overlayHeight;
    uint32_t overlayX, overlayY; // where it goes on the image (its top left)
    uint32_t overlayLayout;
};

// What ReGPU needs from the game's runtime. Required unless marked optional.
struct Host
{
    // The guest's 4 GB address space: guest address A is at base + A.
    uint8_t* base = nullptr;
    // Optional: the file mapping of the 512 MB physical memory, if the game maps it from one
    // (on Windows, not used by the Vulkan renderer: it imports the view's memory, VK_EXT_external_memory_host).
    void* physicalSection = nullptr;
    // Guest memory: physical (GPU-visible) or virtual, at least `align` aligned. Returns the guest address.
    uint32_t (*alloc)(bool physical, uint32_t size, uint32_t align) = nullptr;
    // Runs body on a new thread that can call guest code (a host-created guest thread).
    void (*createThread)(const char* name, void (*body)(PPCContext&)) = nullptr;
    // The calling guest thread's processor control region (KPCR) guest address.
    uint32_t (*currentPcr)() = nullptr;
    // Calls guest code at address with ctx's registers (the graphics interrupt callback).
    void (*guestCall)(PPCContext& ctx, uint32_t address) = nullptr;
    // The guest's interrupt lock for one CPU: interrupt callbacks run holding it.
    void (*interruptLockAcquire)(int cpu) = nullptr;
    void (*interruptLockRelease)() = nullptr;

    // Optional: guest thread CPU use since the last call (" t3:85% t7:40%"), for the perf log.
    std::string (*cpuUsage)(double wallSeconds) = nullptr;
    // Optional: every shader in the game's files, for compiling them all ahead of time on first boot.
    std::vector<ScannedShader> (*staticShaders)() = nullptr;
    // Optional: whether staticShaders' list is already saved (a fast call: no search).
    bool (*staticShadersSaved)() = nullptr;
    // Optional: the title's shipped pipeline list, as pipelines.manifest text ("P vs ps blend depth
    // stencilMaskMode flags [biasScale biasOffset]" in hex, one a line: hashes and render state, no game
    // data), typically merged from players' manifests. Those not built yet are built once, after the static
    // shaders, with the boot screen up; those needing a shader the game makes at run time are built in the
    // background once it exists. Built ones join pipelines.manifest. REGPU_NO_SEED_PIPELINES=1 skips it.
    const char* (*pipelineSeed)() = nullptr;
    // Optional: a 1280x720 RGBA8 screen for the first-boot shader compile (title, progress done/total;
    // total 0 while searching). False: ReGPU's plain progress overlay is shown instead.
    bool (*bootScreen)(const wchar_t* title, uint32_t done, uint32_t total, std::vector<uint8_t>& rgba) = nullptr;
    // Optional: the game's own fix to a shader, made to its translated HLSL before it compiles (a
    // game-specific workaround: the translation stays generic). `hash` identifies the shader (the
    // debug tools print it); patched copies the title makes of it reuse the result. Returns a tag of
    // the game's choosing (0: none); draws using a tagged shader go to onDraw.
    uint32_t (*rewriteShader)(uint64_t hash, std::string& hlsl, bool pixel) = nullptr;
    // Optional: which resolves to write back to memory, and how. Without it, the 8888 colour resolves
    // whose size is listed in REGPU_RESOLVE_WRITEBACK ("WxH,WxH") are written at once.
    Writeback (*resolveWriteback)(const ResolveCopy& copy) = nullptr;
    // Optional: a draw with a tagged shader (rewriteShader), before it is recorded. `c` is the vertex
    // shader's constants (as the title set them); `user` is four float4 the rewritten shaders can read
    // as vuser[0..3] (zero unless set here).
    void (*onDraw)(uint64_t vs, uint64_t ps, uint32_t vsTag, uint32_t psTag, const float c[256][4], float user[4][4]) = nullptr;
    // Optional: a draw with a tagged shader, before it is recorded: true clears the bound colour target
    // over the draw's scissor to `colour` first. For pixels a title leaves unwritten and reads back, which
    // on the console hold whatever another view of the same EDRAM left there (a depth surface, say), and
    // here hold the colour surface's previous contents.
    bool (*clearBeforeDraw)(uint64_t vs, uint64_t ps, uint32_t vsTag, uint32_t psTag, float colour[4]) = nullptr;
    // Optional: each frame the game finishes (VdSwap), on the renderer's thread.
    void (*onSwap)(uint64_t frame) = nullptr;
    // Optional: the host presents. ReGPU then opens no window and no swap chain, and calls this instead of
    // presenting: once for every guest swap once the renderer has started (from the first draw on; not
    // with the REGPU_TEXVIEW debug view; Kind::Frame), after the swap's command buffer was submitted to
    // the queue (vkQueueSubmit2; not waited for), and for each hold or boot screen. Frames and boot
    // screens come on the renderer's thread
    // (the command processor's, or the draw thread's with REGPU_DRAW_THREAD); a hold comes on whichever
    // thread waits for a shader or pipeline, which with REGPU_DRAW_WORKERS can be a draw piece worker.
    // The host may transition image and overlay on its own command buffer and must return them to the
    // layouts given before it returns.
    // That work on image and overlay is submitted on Config::hostQueue (the queue ReGPU renders on) before
    // this returns: ReGPU hands over no semaphore, so that queue (with the host's barriers, from all commands
    // and all memory writes) is what orders the host's reads after ReGPU's writes and ReGPU's next writes
    // after the host's reads. The host keeps neither image past the call: ReGPU writes both again for the
    // next picture. presentImage therefore needs Config::hostDevice and the rest; without them the process
    // ends at Start with a FATAL line in
    // regpu.log (not with REGPU_HEADLESS or REGPU_NO_HOST_PRESENT=1, which never call it). The pointer
    // is read at each call: a host that clears it (at its shutdown) is not called again. SetFullscreen
    // does nothing then (the window and its presentation are the host's), and the picture's size follows
    // SetOutputSize.
    // REGPU_NO_HOST_PRESENT=1 (read at Start; flipping it later does nothing): the callback is ignored and
    // ReGPU presents itself, creating its own swap chain on Config::hostWindow (or its own window), which
    // fails if the host already has a swap chain on that window.
    void (*presentImage)(const PresentImage& picture) = nullptr;
    // Optional: a character typed into the game window (WM_CHAR), e.g. for cheat codes.
    void (*onChar)(wchar_t c) = nullptr;
    // Optional diagnostics: log every host thread's stack (a wait that looks hung); the calling
    // thread's stack as text (a crash); a frame that took `ms` (a spike profiler's dump).
    void (*logThreadStacks)(const char* why) = nullptr;
    std::string (*currentStack)(int maxFrames) = nullptr;
    void (*slowFrame)(double ms) = nullptr;
    // Optional: write tracking of guest physical memory, for the vertex arena (REGPU_VERTEX_ARENA). watchArm makes the
    // pages of [offset, offset + bytes) (physical offsets) report their next write, false if they can't be watched;
    // watchClean: no page of the range written since it was armed, nor since sequence `since`; watchSeq: the current
    // write sequence number. Without them the arena hashes each buffer once a frame.
    bool (*watchArm)(uint32_t offset, uint32_t bytes) = nullptr;
    bool (*watchClean)(uint32_t offset, uint32_t bytes, uint64_t since) = nullptr;
    uint64_t (*watchSeq)() = nullptr;
};

Config& GetConfig();
Host& GetHost();

// Opens the window and the Vulkan device, and starts the command processor. Call once, after filling
// in Config and Host, before the game calls VdInitializeRingBuffer.
void Start();
// The debug server's controller (REGPU_DEBUG=1; tools/regpuctl.py press/hold/stick/trigger): the game
// merges it into player 1's pad where it reads XInput (XInputGetState / XamInputGetState). False when
// nothing is held. Buttons use the XInput bits.
struct DebugPad
{
    uint16_t buttons = 0;
    uint8_t leftTrigger = 0, rightTrigger = 0;
    int16_t lx = 0, ly = 0, rx = 0, ry = 0;
    bool sticks = false, triggers = false; // replace the pad's sticks / triggers instead of adding to them
};
bool DebugPadState(DebugPad& pad);

// Whether the game window has keyboard focus (read keyboard input only then).
bool WindowHasFocus();
// Logs the command processor's state (ring pointers, what it waits on, interrupts, swaps): for a
// watchdog when the game looks hung.
void LogState();
// Call from the game's crash handler, on the crashing thread: if it crashed inside the graphics driver's
// pipeline compiler (some drivers fault on a few pipelines), the pipeline is logged and refused from the
// next start on (regpu-bad-pipelines.txt in Config::userDir), so its draws are dropped instead.
// It also writes out the log lines still queued (REGPU_LOG_ASYNC).
void LogCrashContext();

// The game draws its 3D view wider than 16:9 itself (a widescreen patch): ReGPU then leaves its
// field of view alone instead of widening it (Hor+) at wide resolutions.
void SetGameWidescreen(bool on);
// Call when the game has stopped drawing (the window is closing): flushes the shader and pipeline caches.
void SaveCaches();
// True when no shader or pipeline compile is running and none finished in the last 2 s: for a tool that steps through
// the game's screens to warm the caches, and waits between steps.
bool CompileIdle();
// Benchmark: the presented frames' times are recorded while on; the report (a text file) gives the average, 1% and 0.1%
// lows, the slowest frame, pacing and a 5-second timeline, with the CPU, GPU and settings.
void BenchmarkRecord(bool on);
void BenchmarkWriteReport(const char* path);
// Decodes mip 0 of a 2D texture held outside guest memory (a texture file's data after its header) to RGBA8,
// row-major. `fetch` is the texture fetch constant (6 dwords, host order); false for a format ReGPU can't decode or
// data shorter than the texture.
bool DecodeTexture(const uint32_t fetch[6], const uint8_t* data, size_t size, uint32_t& width, uint32_t& height,
    std::vector<uint8_t>& rgba);
// Writes RGBA8 as a PNG (for a game's preview tools).
bool WritePng(const std::filesystem::path& path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgba);
// For a host that shuts down while ReGPU's threads still run: call this before the guest's memory and
// threads go away. When it returns, no interrupt callback is running in guest code and none will be
// started; interrupts raised later are dropped. It also ends the periodic shader scan of guest memory:
// no scan pass is running and none will start. It waits at most waitMs milliseconds in all for a
// callback and a scan pass in flight (a callback stuck in guest code must not hang the host's shutdown)
// and returns whether neither was left running. Calling it again is harmless. Not to be called from
// inside a callback. After it returns ReGPU is not idle: every thread acting on work already queued
// can still read or write guest memory for it, among them the command processor, the draw workers, the
// fence writer and the texture prehash workers (and, with their debug switches, the overwrite checker
// and the debug server), and the host callbacks (onSwap, resolveWriteback and the others) can still be
// called, among them logThreadStacks from the hang watchdog once no frame has been swapped for 10 s.
bool StopInterrupts(uint32_t waitMs = 500);
// The output (drawing) resolution while the game runs: Config's width and height at start. Taken at
// the start of the next frame (the host GPU idle; surfaces, upscaler and swap chain rebuilt). The
// upscaler and its quality (Config::upscaler, quality) are read every frame and change without it.
void SetOutputSize(uint32_t width, uint32_t height);
// Windowed: the game window's client area to exactly width x height pixels (a resolution change
// from a host's options screen; ReGPU's own window opens at Config's width and height this way).
// Not shrunk to fit the screen; kept with its title bar on it. Maximised or fullscreen: unchanged.
// Works on Config::hostWindow too.
void SetWindowSize(uint32_t width, uint32_t height);
// Borderless fullscreen on the window's display, or a window again (at the output size), while the game
// runs. Config::fullscreen is the starting mode. Not with Config::hostWindow (the host owns its window),
// nor with Host::presentImage (ReGPU has no swap chain then): the call then does nothing.
void SetFullscreen(bool on);
// The anti-aliasing sample count while the game runs (Config's msaa at start): taken at the start of the
// next frame with the host GPU idle; the surfaces are rebuilt and every pipeline is built again for the
// new count (a hitch while they compile). Lowered to what the device supports.
void SetMsaa(uint32_t samples);
// Changes the frame cap and vsync while the game runs (a host's options screen); Config's maxFps
// and vsync are the starting values.
void SetFrameLimit(uint32_t maxFps, bool vsync);
// Texture filtering while the game runs: -1 the fetch's own anisotropy (the default), 0 none, else
// that many samples at most (2-16) for every wrapping linear fetch. Applies to the 2D pixel fetches of
// REGPU_LEAN_FETCH (its sampler table); others keep their samplers.
void SetAnisotropy(int maxAnisotropy);

// The GPU ReGPU renders on and its memory, for a host's logs (VK_EXT_memory_budget where the driver has it, else the
// heap sizes as budgets and no usage). Local: the GPU's own memory (an iGPU's carve-out); shared: system memory the GPU
// reads. False before the device exists (Start).
struct GpuMemory
{
    char name[256] = {};
    uint32_t vendorId = 0, deviceId = 0;
    bool integrated = false;
    uint64_t localUsage = 0, localBudget = 0, sharedUsage = 0, sharedBudget = 0;
};
bool GetGpuMemory(GpuMemory& out);

// VR (REGPU_VR, a build with OpenXR; src/render/xr.cpp). The host poses its camera with the headset and tells ReGPU
// which pose each game frame was rendered with; frames without one go on the virtual screen.
namespace vr
{
struct HeadPose
{
    float orientation[4] = { 0, 0, 0, 1 }; // x, y, z, w (OpenXR: right-handed, y up, -z forward)
    float position[3] = {};                 // metres from where the runtime was recentred
    float fovLeft = 0, fovRight = 0, fovUp = 0, fovDown = 0; // the headset's eye field of view (radians, left eye)
};
// The head's pose predicted for the next headset frame. false: no running headset session.
bool GetHeadPose(HeadPose& out);
// Once per game frame: whether the newest game frame's 3D view is head-tracked (pose or nullptr) and its vertical field
// of view (radians, the whole angle). Decides stereo and the HUD panel; the pose a presented frame was drawn with comes
// from SubmitTaggedPose.
void SubmitFramePose(const HeadPose* pose, float renderFovY);
// Exact pairing of poses with frames (FM3 port, session 39): a title whose logic runs more often than it draws (or on
// another thread than the one that draws) can't pair poses with frames by counting. The host puts `tag` (1-65535) into
// a vertex shader constant of the main view as the float bits 0x00C00000 | tag (a denormal: e.g. a view matrix's
// unused m03, which changes nothing drawn) and hands the pose here; the draw thread reads the newest tag of each frame's
// main view and the headset gets that frame with that pose.
constexpr uint32_t kPoseTagBits = 0x00C00000u;
void SubmitTaggedPose(uint32_t tag, const HeadPose& pose, float renderFovY);
// The head's present place and heading become the origin (straight ahead, level): poses, the virtual screen and the
// HUD panel follow. F8 in the game window does the same.
void Recentre();
} // namespace vr

// Triple screen (REGPU_TRIPLE=1, FM3 port session 39): one simulation, three screens. The main view (the frame from
// its first full-width draw to the swap) is drawn three times, once per screen, side by side in a window three screens
// wide (the resolution is the three together). The title's camera must draw with a frustum wide enough for all three
// (its culling); each screen's pass maps that frustum's clip position (x, y, z, w) to the screen's own projection:
//   x' = a x + b w,  w' = c x + d w,  y' = e y,  z' = z + (w' - w)
// (a turn about the eye and an off-axis projection onto the screen's plane; depth exact for z = w + B). The host works
// these out from its screens' layout (sizes, distance, bezels, gaps, angles) in the camera's own clip convention.
// views: 0 left, 1 centre, 2 right. Once per game frame while a 3D view is drawn; without a recent call a frame is
// drawn once, in the middle.
namespace triple
{
struct View
{
    float a = 0, b = 0, c = 0, d = 1, e = 0;
};
void SubmitViews(const View (&views)[3]);
} // namespace triple
} // namespace regpu
