// VR through OpenXR: see xr.h. Our own code; the design follows the order Forza Horizon 1: ReImagined's VR took (a
// virtual screen first, then a head-tracked view, then true stereo), not its code.
#include "render/xr.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "live_switch.h"
#include "log.h"
#include "regpu/regpu.h"

#if !REGPU_OPENXR

namespace regpu::xr
{
bool Wanted()
{
    static bool logged = false;
    if (getenv("REGPU_VR") && !logged)
    {
        logged = true;
        Log("vr: REGPU_VR is set but this build has no OpenXR (REGPU_OPENXR_DIR)");
    }
    return false;
}
void Start(ID3D12Device*, ID3D12CommandQueue*) {}
void CaptureFrame(ID3D12GraphicsCommandList*, ID3D12Resource*) {}
bool StereoFrame(float&, float&) { return false; }
bool MonoHudPanel() { return false; }
uint32_t PaceCap() { return 0; }
void SetCaptureTag(uint32_t) {}
ID3D12Resource* StereoTarget(int, uint32_t, uint32_t) { return nullptr; }
void StereoHudCaptured() {}
}
namespace regpu::vr
{
bool GetHeadPose(HeadPose&) { return false; }
void SubmitFramePose(const HeadPose*, float) {}
void SubmitTaggedPose(uint32_t, const HeadPose&, float) {}
void Recentre() {}
}

#else

#include <windows.h>

#include <d3d12.h>
#include <unknwn.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#define XR_USE_GRAPHICS_API_D3D12
#define XR_USE_PLATFORM_WIN32
#define XR_NO_PROTOTYPES
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

using Microsoft::WRL::ComPtr;

namespace
{
// The OpenXR functions used, fetched from the loader at run time (no link to it).
#define REGPU_XR_GLOBAL(X) X(xrCreateInstance) X(xrEnumerateInstanceExtensionProperties)
#define REGPU_XR_INSTANCE(X)                                                                                                     \
    X(xrDestroyInstance) X(xrGetSystem) X(xrGetSystemProperties) X(xrGetD3D12GraphicsRequirementsKHR) X(xrCreateSession)          \
    X(xrDestroySession) X(xrCreateReferenceSpace) X(xrDestroySpace) X(xrEnumerateSwapchainFormats) X(xrCreateSwapchain)          \
    X(xrDestroySwapchain) X(xrEnumerateSwapchainImages) X(xrAcquireSwapchainImage) X(xrWaitSwapchainImage)                      \
    X(xrReleaseSwapchainImage) X(xrPollEvent) X(xrBeginSession) X(xrEndSession) X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame)     \
    X(xrResultToString) X(xrEnumerateViewConfigurationViews) X(xrLocateSpace) X(xrLocateViews)

PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr = nullptr;
#define REGPU_XR_DECLARE(name) PFN_##name name = nullptr;
REGPU_XR_GLOBAL(REGPU_XR_DECLARE)
REGPU_XR_INSTANCE(REGPU_XR_DECLARE)
#undef REGPU_XR_DECLARE

// The finished frame for the headset: written where the presenter copies into the back buffer, read by the headset
// thread; both on the one queue, so in submission order. Kept in COMMON between the two.
std::mutex g_screenMutex;
ComPtr<ID3D12Resource> g_screen;
std::vector<ComPtr<ID3D12Resource>> g_oldScreens; // replaced ones, kept a while (commands in flight may read them)
std::atomic<bool> g_capturing{ false };
// Head-tracked frames (regpu::vr): the pose each game frame was rendered with, in frame order, paired with the captured
// frames in order; the screen's own pose comes with it.
struct FramePose
{
    bool valid = false;
    regpu::vr::HeadPose pose;
    float fovY = 0;
};
// Poses by tag (SubmitTaggedPose), and the tag of the frame CaptureFrame records next (SetCaptureTag).
struct TaggedPose
{
    uint32_t tag = 0;
    FramePose pose;
};
TaggedPose g_tagged[1024];                // (g_taggedMutex) by tag & 1023
std::mutex g_taggedMutex;
std::atomic<uint32_t> g_captureTag{ 0 };
std::atomic<uint32_t> g_newestTag{ 0 };
FramePose g_screenPose;             // (g_screenMutex) the captured frame's
// For GetHeadPose from the game's threads: the running session's spaces and the next display time.
std::mutex g_poseMutex;
XrSession g_poseSession = XR_NULL_HANDLE;
XrSpace g_poseLocal = XR_NULL_HANDLE, g_poseView = XR_NULL_HANDLE;
std::atomic<int64_t> g_predicted{ 0 };
XrFovf g_eyeFov{};
XrFovf g_eyeFovs[2]{}; // (g_poseMutex) both eyes' fields of view, for the lens crop
std::atomic<uint64_t> g_captured{ 0 };
// Headset pacing: the headset's display rate (0 without a running session), and a wake-up when a game frame is captured.
std::atomic<uint32_t> g_headsetHz{ 0 };
// Frame-pose pairing check (logged with the 30 s line): frames whose tag had no pose (overwritten or never handed in).
std::atomic<uint64_t> g_posesMissing{ 0 };
std::mutex g_capturedMutex;
std::condition_variable g_capturedCv;
// Stereo (REGPU_VR=stereo): the two eyes' images of the newest stereo frame (renderer.cpp RendererStereoCaptureEye
// grades into them; COMMON between uses), and whether the frame CaptureFrame took was one.
ComPtr<ID3D12Resource> g_eyes[3];          // (g_screenMutex) [2]: the HUD panel's image (premultiplied alpha)
std::atomic<bool> g_hudThisFrame{ false };  // the HUD written since the last CaptureFrame
bool g_screenHud = false;                   // (g_screenMutex) the captured frame has its HUD in g_eyes[2]
std::atomic<bool> g_eyesThisFrame{ false }; // both eyes written since the last CaptureFrame
bool g_screenStereo = false;                // (g_screenMutex) the captured frame has its eyes in g_eyes
// The newest frame pose handed in (for StereoFrame on the draw thread): head-tracked or not, its field of view; and the
// headset's eye distance.
std::atomic<bool> g_lastPosed{ false };
std::atomic<float> g_lastFovY{ 0.0f };
std::atomic<float> g_ipd{ 0.064f };
std::atomic<bool> g_recentre{ false }; // regpu::vr::Recentre or F8: the headset thread recentres at its next frame
ComPtr<ID3D12Device> g_device;
ComPtr<ID3D12CommandQueue> g_queue;

std::string ResultText(XrInstance instance, XrResult r)
{
    char text[XR_MAX_RESULT_STRING_SIZE] = "";
    if (instance && xrResultToString)
        xrResultToString(instance, r, text);
    return std::string(text) + " (" + std::to_string(int(r)) + ")";
}

void Barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}

float EnvFloat(const char* name, float fallback)
{
    const char* v = getenv(name);
    return v && *v ? float(atof(v)) : fallback;
}

bool LoadLoader()
{
    HMODULE dll = LoadLibraryW(L"openxr_loader.dll");
    if (!dll)
    {
        Log("vr: openxr_loader.dll not found beside the game: VR off");
        return false;
    }
    xrGetInstanceProcAddr = reinterpret_cast<PFN_xrGetInstanceProcAddr>(GetProcAddress(dll, "xrGetInstanceProcAddr"));
    if (!xrGetInstanceProcAddr)
        return false;
#define REGPU_XR_LOAD(name) xrGetInstanceProcAddr(XR_NULL_HANDLE, #name, reinterpret_cast<PFN_xrVoidFunction*>(&name));
    REGPU_XR_GLOBAL(REGPU_XR_LOAD)
#undef REGPU_XR_LOAD
    return xrCreateInstance && xrEnumerateInstanceExtensionProperties;
}

bool LoadInstanceFunctions(XrInstance instance)
{
    bool ok = true;
#define REGPU_XR_LOAD(name)                                                                                                      \
    if (XR_FAILED(xrGetInstanceProcAddr(instance, #name, reinterpret_cast<PFN_xrVoidFunction*>(&name))) || !name)                \
    {                                                                                                                            \
        Log("vr: OpenXR function %s missing", #name);                                                                            \
        ok = false;                                                                                                              \
    }
    REGPU_XR_INSTANCE(REGPU_XR_LOAD)
#undef REGPU_XR_LOAD
    return ok;
}

// Half the eye distance for stereo: the headset's (from its two views), or REGPU_VR_IPD (millimetres).
float HalfIpd()
{
    static const float fixed = EnvFloat("REGPU_VR_IPD", 0.0f) * 0.001f;
    return 0.5f * (fixed > 0.0f ? fixed : g_ipd.load(std::memory_order_relaxed));
}

bool StereoMode()
{
    static const bool on = [] {
        const char* v = getenv("REGPU_VR");
        return v && strcmp(v, "stereo") == 0;
    }();
    return on;
}

bool CockpitMode()
{
    static const bool on = [] {
        const char* v = getenv("REGPU_VR");
        return v && strcmp(v, "cockpit") == 0;
    }();
    return on;
}

// Headset pacing (REGPU_VR_PACE; the port turns it on in cockpit and stereo, 0 = off): the game's frame cap follows the
// headset's display rate, and the headset thread waits a little for a game frame that is nearly done.
bool PaceOn()
{
    static const bool on = [] {
        const char* v = getenv("REGPU_VR_PACE");
        return v && *v && strcmp(v, "0") != 0;
    }();
    return on;
}

// The part of an eye's image its lens shows (session 39 of the FM3 port): the frame is drawn symmetric, 16:9, with the
// vertical field of view fitted to the lens (and a margin), so it reaches far past the lens at the sides. Each eye is
// handed to the runtime as the rectangle of its own lens field (plus REGPU_VR_CROP_MARGIN degrees, default 3, for the
// runtime's reprojection) with that rectangle's exact angles. tanX/tanY: the frame's half extents (tangents).
void LensCrop(const XrFovf& lens, float tanX, float tanY, uint32_t w, uint32_t h, XrRect2Di& rect, XrFovf& fov)
{
    static const float margin = EnvFloat("REGPU_VR_CROP_MARGIN", 3.0f) * 3.14159265f / 180.0f;
    static LiveSwitch full("REGPU_VR_NO_CROP"); // A/B: the whole frame for each eye, as before
    float l = -tanX, r = tanX, u = tanY, d = -tanY;
    const bool valid = lens.angleRight > lens.angleLeft && lens.angleUp > lens.angleDown && lens.angleLeft > -1.5f && lens.angleRight < 1.5f &&
                       lens.angleUp < 1.5f && lens.angleDown > -1.5f;
    if (valid && !full)
    {
        l = std::max(-tanX, tanf(std::max(lens.angleLeft - margin, -1.5f)));
        r = std::min(tanX, tanf(std::min(lens.angleRight + margin, 1.5f)));
        u = std::min(tanY, tanf(std::min(lens.angleUp + margin, 1.5f)));
        d = std::max(-tanY, tanf(std::max(lens.angleDown - margin, -1.5f)));
    }
    const int32_t x0 = std::clamp(int32_t(floorf((l / tanX + 1) * 0.5f * float(w))), 0, int32_t(w) - 1);
    const int32_t x1 = std::clamp(int32_t(ceilf((r / tanX + 1) * 0.5f * float(w))), x0 + 1, int32_t(w));
    const int32_t y0 = std::clamp(int32_t(floorf((1 - u / tanY) * 0.5f * float(h))), 0, int32_t(h) - 1);
    const int32_t y1 = std::clamp(int32_t(ceilf((1 - d / tanY) * 0.5f * float(h))), y0 + 1, int32_t(h));
    rect = { { x0, y0 }, { x1 - x0, y1 - y0 } };
    // The rectangle's own angles (whole pixels).
    fov.angleLeft = atanf((2.0f * float(x0) / float(w) - 1) * tanX);
    fov.angleRight = atanf((2.0f * float(x1) / float(w) - 1) * tanX);
    fov.angleUp = atanf((1 - 2.0f * float(y0) / float(h)) * tanY);
    fov.angleDown = atanf((1 - 2.0f * float(y1) / float(h)) * tanY);
}

// One OpenXR instance + session at a time; the thread makes a new one after a loss or when no headset was there yet.
struct Session
{
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace local = XR_NULL_HANDLE, view = XR_NULL_HANDLE;
    // `local` after a recentre is a LOCAL space moved to the head's pose then; the runtime's own and replaced ones are
    // kept until the session ends (another thread may still be locating in them).
    XrSpace baseLocal = XR_NULL_HANDLE;
    std::vector<XrSpace> oldSpaces;
    XrSwapchain quad = XR_NULL_HANDLE;
    uint32_t quadW = 0, quadH = 0;
    // Stereo: one swapchain image holding both eyes side by side (left in the left half).
    XrSwapchain pair = XR_NULL_HANDLE;
    uint32_t pairW = 0, pairH = 0;
    std::vector<ID3D12Resource*> pairImages;
    // The HUD panel (stereo frames whose title draws its HUD apart: renderer.cpp StereoHudDraw).
    XrSwapchain hud = XR_NULL_HANDLE;
    uint32_t hudW = 0, hudH = 0;
    std::vector<ID3D12Resource*> hudImages;
    int64_t quadFormat = 0;
    std::vector<ID3D12Resource*> images;
    static constexpr uint32_t kLists = 3;
    ComPtr<ID3D12CommandAllocator> allocators[kLists];
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t fenceValues[kLists] = {};
    uint64_t nextFence = 1;
    uint32_t listIndex = 0;
    HANDLE fenceEvent = nullptr;
    bool running = false;
    uint32_t recommendedW = 0, recommendedH = 0;
    bool densityLogged = false;

    ~Session()
    {
        g_capturing = false;
        g_headsetHz = 0;
        {
            std::lock_guard lock(g_poseMutex);
            g_poseSession = XR_NULL_HANDLE;
        }
        if (view)
            xrDestroySpace(view);
        if (quad)
            xrDestroySwapchain(quad);
        if (pair)
            xrDestroySwapchain(pair);
        if (hud)
            xrDestroySwapchain(hud);
        for (XrSpace sp : oldSpaces)
            xrDestroySpace(sp);
        if (baseLocal && baseLocal != local)
            xrDestroySpace(baseLocal);
        if (local)
            xrDestroySpace(local);
        if (session)
            xrDestroySession(session);
        if (instance)
            xrDestroyInstance(instance);
        if (fenceEvent)
            CloseHandle(fenceEvent);
    }

    // false: try again later (no runtime, no headset yet, ...). `quiet`: the reason was logged before.
    bool Create(bool quiet)
    {
        uint32_t count = 0;
        xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
        std::vector<XrExtensionProperties> exts(count, { XR_TYPE_EXTENSION_PROPERTIES });
        xrEnumerateInstanceExtensionProperties(nullptr, count, &count, exts.data());
        bool d3d12 = false;
        for (const auto& e : exts)
            d3d12 |= strcmp(e.extensionName, XR_KHR_D3D12_ENABLE_EXTENSION_NAME) == 0;
        if (!d3d12)
        {
            if (!quiet)
                Log("vr: the OpenXR runtime has no Direct3D 12 support (%u extensions; is a runtime installed and active?)", count);
            return false;
        }
        const char* enabled[] = { XR_KHR_D3D12_ENABLE_EXTENSION_NAME };
        XrInstanceCreateInfo ci{ XR_TYPE_INSTANCE_CREATE_INFO };
        strcpy_s(ci.applicationInfo.applicationName, "ReGPU");
        ci.applicationInfo.applicationVersion = 1;
        strcpy_s(ci.applicationInfo.engineName, "ReGPU");
        ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        ci.enabledExtensionCount = 1;
        ci.enabledExtensionNames = enabled;
        if (XrResult r = xrCreateInstance(&ci, &instance); XR_FAILED(r))
        {
            if (!quiet)
                Log("vr: xrCreateInstance failed: %d", int(r));
            return false;
        }
        if (!LoadInstanceFunctions(instance))
            return false;
        XrSystemGetInfo si{ XR_TYPE_SYSTEM_GET_INFO };
        si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        if (XrResult r = xrGetSystem(instance, &si, &system); XR_FAILED(r))
        {
            if (!quiet)
                Log("vr: no headset yet (%s): trying again every few seconds", ResultText(instance, r).c_str());
            return false;
        }
        XrSystemProperties props{ XR_TYPE_SYSTEM_PROPERTIES };
        xrGetSystemProperties(instance, system, &props);
        XrGraphicsRequirementsD3D12KHR req{ XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR };
        if (XrResult r = xrGetD3D12GraphicsRequirementsKHR(instance, system, &req); XR_FAILED(r))
        {
            Log("vr: xrGetD3D12GraphicsRequirementsKHR failed: %s", ResultText(instance, r).c_str());
            return false;
        }
        const LUID ours = g_device->GetAdapterLuid();
        if (ours.LowPart != req.adapterLuid.LowPart || ours.HighPart != req.adapterLuid.HighPart)
            Log("vr: the headset runtime wants another graphics card than the one the game renders on (set it in the launcher)");
        XrGraphicsBindingD3D12KHR binding{ XR_TYPE_GRAPHICS_BINDING_D3D12_KHR };
        binding.device = g_device.Get();
        binding.queue = g_queue.Get();
        XrSessionCreateInfo sci{ XR_TYPE_SESSION_CREATE_INFO };
        sci.next = &binding;
        sci.systemId = system;
        if (XrResult r = xrCreateSession(instance, &sci, &session); XR_FAILED(r))
        {
            Log("vr: xrCreateSession failed: %s", ResultText(instance, r).c_str());
            return false;
        }
        XrReferenceSpaceCreateInfo rs{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        rs.poseInReferenceSpace.orientation.w = 1;
        if (XrResult r = xrCreateReferenceSpace(session, &rs, &local); XR_FAILED(r))
        {
            Log("vr: xrCreateReferenceSpace failed: %s", ResultText(instance, r).c_str());
            return false;
        }
        baseLocal = local;
        rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        if (XrResult r = xrCreateReferenceSpace(session, &rs, &view); XR_FAILED(r))
        {
            Log("vr: xrCreateReferenceSpace (view) failed: %s", ResultText(instance, r).c_str());
            return false;
        }
        for (uint32_t i = 0; i < kLists; i++)
            if (FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[i]))))
                return false;
        if (FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0].Get(), nullptr, IID_PPV_ARGS(&list))))
            return false;
        list->Close();
        if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        Log("vr: headset session created: %s (vendor %u)", props.systemName, props.vendorId);
        return true;
    }

    // The quad's swapchain at the screen's size (made again when the game's output size changes).
    bool EnsureQuad(uint32_t w, uint32_t h)
    {
        if (quad && quadW == w && quadH == h)
            return true;
        if (quad)
        {
            WaitIdle();
            xrDestroySwapchain(quad);
            quad = XR_NULL_HANDLE;
            images.clear();
        }
        uint32_t count = 0;
        xrEnumerateSwapchainFormats(session, 0, &count, nullptr);
        std::vector<int64_t> formats(count);
        xrEnumerateSwapchainFormats(session, count, &count, formats.data());
        // The back buffer is R8G8B8A8_UNORM holding gamma-encoded colour: an sRGB swapchain of the same layout takes
        // its bytes as they are (a copy, no shader) and the runtime reads them as sRGB.
        quadFormat = 0;
        for (int64_t want : { int64_t(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), int64_t(DXGI_FORMAT_R8G8B8A8_UNORM) })
            if (std::find(formats.begin(), formats.end(), want) != formats.end())
            {
                quadFormat = want;
                break;
            }
        if (!quadFormat)
        {
            Log("vr: the runtime offers no RGBA8 swapchain format (%u formats)", count);
            return false;
        }
        XrSwapchainCreateInfo ci{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
        ci.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        ci.format = quadFormat;
        ci.sampleCount = 1;
        ci.width = w;
        ci.height = h;
        ci.faceCount = 1;
        ci.arraySize = 1;
        ci.mipCount = 1;
        if (XrResult r = xrCreateSwapchain(session, &ci, &quad); XR_FAILED(r))
        {
            Log("vr: xrCreateSwapchain %ux%u failed: %s", w, h, ResultText(instance, r).c_str());
            quad = XR_NULL_HANDLE;
            return false;
        }
        xrEnumerateSwapchainImages(quad, 0, &count, nullptr);
        std::vector<XrSwapchainImageD3D12KHR> imgs(count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR });
        xrEnumerateSwapchainImages(quad, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
        for (const auto& i : imgs)
            images.push_back(i.texture);
        quadW = w, quadH = h;
        Log("vr: virtual screen %ux%u (%s)", w, h, quadFormat == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? "sRGB" : "linear format: colours may look washed out");
        return true;
    }

    // The stereo pair's swapchain: two eyes of w x h side by side.
    bool EnsurePair(uint32_t w, uint32_t h)
    {
        if (pair && pairW == w && pairH == h)
            return true;
        if (pair)
        {
            WaitIdle();
            xrDestroySwapchain(pair);
            pair = XR_NULL_HANDLE;
            pairImages.clear();
        }
        if (!quadFormat && !EnsureQuad(w, h)) // (picks the format)
            return false;
        XrSwapchainCreateInfo ci{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
        ci.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        ci.format = quadFormat;
        ci.sampleCount = 1;
        ci.width = w * 2;
        ci.height = h;
        ci.faceCount = 1;
        ci.arraySize = 1;
        ci.mipCount = 1;
        if (XrResult r = xrCreateSwapchain(session, &ci, &pair); XR_FAILED(r))
        {
            Log("vr: xrCreateSwapchain %ux%u (stereo) failed: %s", w * 2, h, ResultText(instance, r).c_str());
            pair = XR_NULL_HANDLE;
            return false;
        }
        uint32_t count = 0;
        xrEnumerateSwapchainImages(pair, 0, &count, nullptr);
        std::vector<XrSwapchainImageD3D12KHR> imgs(count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR });
        xrEnumerateSwapchainImages(pair, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
        for (const auto& i : imgs)
            pairImages.push_back(i.texture);
        pairW = w, pairH = h;
        Log("vr: stereo, %ux%u per eye", w, h);
        return true;
    }

    // The HUD panel's swapchain (w x h, the format the screen's uses).
    bool EnsureHud(uint32_t w, uint32_t h)
    {
        if (hud && hudW == w && hudH == h)
            return true;
        if (hud)
        {
            WaitIdle();
            xrDestroySwapchain(hud);
            hud = XR_NULL_HANDLE;
            hudImages.clear();
        }
        if (!quadFormat && !EnsureQuad(w, h))
            return false;
        XrSwapchainCreateInfo ci{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
        ci.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        ci.format = quadFormat;
        ci.sampleCount = 1;
        ci.width = w;
        ci.height = h;
        ci.faceCount = 1;
        ci.arraySize = 1;
        ci.mipCount = 1;
        if (XrResult r = xrCreateSwapchain(session, &ci, &hud); XR_FAILED(r))
        {
            Log("vr: xrCreateSwapchain %ux%u (HUD) failed: %s", w, h, ResultText(instance, r).c_str());
            hud = XR_NULL_HANDLE;
            return false;
        }
        uint32_t count = 0;
        xrEnumerateSwapchainImages(hud, 0, &count, nullptr);
        std::vector<XrSwapchainImageD3D12KHR> imgs(count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR });
        xrEnumerateSwapchainImages(hud, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
        for (const auto& i : imgs)
            hudImages.push_back(i.texture);
        hudW = w, hudH = h;
        Log("vr: HUD panel %ux%u", w, h);
        return true;
    }

    void WaitIdle()
    {
        const uint64_t v = nextFence++;
        g_queue->Signal(fence.Get(), v);
        if (fence->GetCompletedValue() < v)
        {
            fence->SetEventOnCompletion(v, fenceEvent);
            WaitForSingleObject(fenceEvent, 2000);
        }
    }

    // Copies the latest captured frame into the quad's next image.
    bool DrawQuad(FramePose& pose, bool& stereo, bool& withHud)
    {
        ComPtr<ID3D12Resource> screen, eyes[3];
        {
            std::lock_guard lock(g_screenMutex);
            screen = g_screen;
            pose = g_screenPose;
            stereo = g_screenStereo && g_eyes[0] && g_eyes[1] && pose.valid;
            // The HUD on its panel: stereo frames, and head-tracked mono ones (cockpit) whose HUD was taken apart.
            withHud = pose.valid && g_screenHud && g_eyes[2];
            if (stereo)
                eyes[0] = g_eyes[0], eyes[1] = g_eyes[1];
            eyes[2] = withHud ? g_eyes[2] : nullptr;
        }
        if (stereo)
            return DrawPair(eyes, withHud);
        if (!screen)
            return false;
        const D3D12_RESOURCE_DESC d = screen->GetDesc();
        if (!EnsureQuad(uint32_t(d.Width), d.Height))
            return false;
        uint32_t hudIndex = 0;
        if (withHud)
        {
            const D3D12_RESOURCE_DESC hd = eyes[2]->GetDesc();
            XrSwapchainImageAcquireInfo hai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
            withHud = EnsureHud(uint32_t(hd.Width), hd.Height) && XR_SUCCEEDED(xrAcquireSwapchainImage(hud, &hai, &hudIndex));
            if (withHud)
            {
                XrSwapchainImageWaitInfo hwi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
                hwi.timeout = XR_INFINITE_DURATION;
                xrWaitSwapchainImage(hud, &hwi);
            }
        }
        uint32_t index = 0;
        XrSwapchainImageAcquireInfo ai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
        if (XR_FAILED(xrAcquireSwapchainImage(quad, &ai, &index)))
            return false;
        XrSwapchainImageWaitInfo wi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
        wi.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(quad, &wi);
        // (This list's previous use has finished: kLists frames ago.)
        const uint32_t li = listIndex++ % kLists;
        if (fence->GetCompletedValue() < fenceValues[li])
        {
            fence->SetEventOnCompletion(fenceValues[li], fenceEvent);
            WaitForSingleObject(fenceEvent, 1000);
        }
        allocators[li]->Reset();
        list->Reset(allocators[li].Get(), nullptr);
        ID3D12Resource* image = images[index];
        Barrier(list.Get(), screen.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(list.Get(), image, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyResource(image, screen.Get());
        Barrier(list.Get(), image, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
        Barrier(list.Get(), screen.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (withHud)
            CopyHud(hudIndex, eyes[2].Get());
        list->Close();
        ID3D12CommandList* lists[] = { list.Get() };
        g_queue->ExecuteCommandLists(1, lists);
        fenceValues[li] = nextFence++;
        g_queue->Signal(fence.Get(), fenceValues[li]);
        XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
        xrReleaseSwapchainImage(quad, &ri);
        if (withHud)
            xrReleaseSwapchainImage(hud, &ri);
        return true;
    }

    // The HUD's image into the HUD panel's swapchain image `index` (in the open list).
    void CopyHud(uint32_t index, ID3D12Resource* src)
    {
        ID3D12Resource* hudImage = hudImages[index];
        Barrier(list.Get(), hudImage, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        Barrier(list.Get(), src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyResource(hudImage, src);
        Barrier(list.Get(), src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        Barrier(list.Get(), hudImage, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    // Copies the two eyes of the newest stereo frame into the pair's next image, side by side.
    bool DrawPair(ComPtr<ID3D12Resource> (&eyes)[3], bool& withHud)
    {
        const D3D12_RESOURCE_DESC d = eyes[0]->GetDesc();
        if (!EnsurePair(uint32_t(d.Width), d.Height))
            return false;
        uint32_t hudIndex = 0;
        if (withHud)
        {
            const D3D12_RESOURCE_DESC hd = eyes[2]->GetDesc();
            XrSwapchainImageAcquireInfo hai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
            withHud = EnsureHud(uint32_t(hd.Width), hd.Height) && XR_SUCCEEDED(xrAcquireSwapchainImage(hud, &hai, &hudIndex));
            if (withHud)
            {
                XrSwapchainImageWaitInfo hwi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
                hwi.timeout = XR_INFINITE_DURATION;
                xrWaitSwapchainImage(hud, &hwi);
            }
        }
        uint32_t index = 0;
        XrSwapchainImageAcquireInfo ai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
        if (XR_FAILED(xrAcquireSwapchainImage(pair, &ai, &index)))
            return false;
        XrSwapchainImageWaitInfo wi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
        wi.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(pair, &wi);
        const uint32_t li = listIndex++ % kLists;
        if (fence->GetCompletedValue() < fenceValues[li])
        {
            fence->SetEventOnCompletion(fenceValues[li], fenceEvent);
            WaitForSingleObject(fenceEvent, 1000);
        }
        allocators[li]->Reset();
        list->Reset(allocators[li].Get(), nullptr);
        ID3D12Resource* image = pairImages[index];
        Barrier(list.Get(), image, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        for (int e = 0; e < 2; e++)
        {
            Barrier(list.Get(), eyes[e].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{ image, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX }, src{ eyes[e].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
            dst.SubresourceIndex = src.SubresourceIndex = 0;
            const D3D12_BOX box{ 0, 0, 0, std::min(uint32_t(eyes[e]->GetDesc().Width), pairW), std::min(eyes[e]->GetDesc().Height, pairH), 1 };
            list->CopyTextureRegion(&dst, e ? pairW : 0, 0, 0, &src, &box);
            Barrier(list.Get(), eyes[e].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
        Barrier(list.Get(), image, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
        if (withHud)
            CopyHud(hudIndex, eyes[2].Get());
        list->Close();
        ID3D12CommandList* lists[] = { list.Get() };
        g_queue->ExecuteCommandLists(1, lists);
        fenceValues[li] = nextFence++;
        g_queue->Signal(fence.Get(), fenceValues[li]);
        XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
        xrReleaseSwapchainImage(pair, &ri);
        if (withHud)
            xrReleaseSwapchainImage(hud, &ri);
        return true;
    }

    // The head's pose now (in the runtime's LOCAL space) becomes the origin: position, and its heading about the
    // vertical only (the horizon stays level).
    void Recentre(XrTime time)
    {
        XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
        if (!time || XR_FAILED(xrLocateSpace(view, baseLocal, time, &loc)) || !(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
            return;
        const XrQuaternionf q = loc.pose.orientation;
        // Forward (-z) turned by q, its heading in the horizontal plane.
        const float fx = -(2 * (q.x * q.z + q.y * q.w)), fz = -(1 - 2 * (q.x * q.x + q.y * q.y));
        const float yaw = atan2f(-fx, -fz);
        XrReferenceSpaceCreateInfo rs{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        rs.poseInReferenceSpace.orientation = { 0, sinf(yaw * 0.5f), 0, cosf(yaw * 0.5f) };
        rs.poseInReferenceSpace.position = (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) ? loc.pose.position : XrVector3f{ 0, 0, 0 };
        XrSpace made = XR_NULL_HANDLE;
        if (XR_FAILED(xrCreateReferenceSpace(session, &rs, &made)))
            return;
        {
            std::lock_guard lock(g_poseMutex);
            if (local != baseLocal)
                oldSpaces.push_back(local);
            local = made;
            g_poseLocal = made;
        }
        Log("vr: recentred (heading %.0f deg)", yaw * 57.2958f);
    }

    // Runs until the session ends or is lost.
    void Run()
    {
        const float width = EnvFloat("REGPU_VR_SCREEN_WIDTH", 2.2f), distance = EnvFloat("REGPU_VR_SCREEN_DISTANCE", 2.5f);
        const float height = EnvFloat("REGPU_VR_SCREEN_HEIGHT", 0.0f);
        const float hudWidth = EnvFloat("REGPU_VR_HUD_WIDTH", 1.4f), hudDistance = EnvFloat("REGPU_VR_HUD_DISTANCE", 1.1f);
        const float hudHeight = EnvFloat("REGPU_VR_HUD_HEIGHT", -0.05f);
        uint64_t frames = 0, shown = 0;
        auto lastLog = std::chrono::steady_clock::now();
        for (;;)
        {
            XrEventDataBuffer ev{ XR_TYPE_EVENT_DATA_BUFFER };
            while (xrPollEvent(instance, &ev) == XR_SUCCESS)
            {
                if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
                {
                    const auto& sc = *reinterpret_cast<XrEventDataSessionStateChanged*>(&ev);
                    Log("vr: session state %d", int(sc.state));
                    if (sc.state == XR_SESSION_STATE_READY)
                    {
                        XrSessionBeginInfo bi{ XR_TYPE_SESSION_BEGIN_INFO };
                        bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                        if (XrResult r = xrBeginSession(session, &bi); XR_FAILED(r))
                            Log("vr: xrBeginSession failed: %s", ResultText(instance, r).c_str());
                        else
                        {
                            running = true;
                            g_capturing = true;
                            // What the headset asks for per eye, to compare with what the game's frame gives across
                            // the lens (logged at the first stereo frame).
                            uint32_t n = 0;
                            XrViewConfigurationView cv[2] = { { XR_TYPE_VIEW_CONFIGURATION_VIEW }, { XR_TYPE_VIEW_CONFIGURATION_VIEW } };
                            if (XR_SUCCEEDED(xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &n, cv)) && n)
                            {
                                recommendedW = cv[0].recommendedImageRectWidth, recommendedH = cv[0].recommendedImageRectHeight;
                                Log("vr: the headset recommends %ux%u per eye (most %ux%u)", recommendedW, recommendedH, cv[0].maxImageRectWidth,
                                    cv[0].maxImageRectHeight);
                            }
                            std::lock_guard lock(g_poseMutex);
                            g_poseSession = session, g_poseLocal = local, g_poseView = view;
                        }
                    }
                    else if (sc.state == XR_SESSION_STATE_STOPPING)
                    {
                        running = false;
                        g_capturing = false;
                        g_headsetHz = 0;
                        {
                            std::lock_guard lock(g_poseMutex);
                            g_poseSession = XR_NULL_HANDLE;
                        }
                        xrEndSession(session);
                    }
                    else if (sc.state == XR_SESSION_STATE_EXITING || sc.state == XR_SESSION_STATE_LOSS_PENDING)
                        return;
                }
                else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING)
                    return;
                ev = { XR_TYPE_EVENT_DATA_BUFFER };
            }
            if (!running)
            {
                Sleep(50);
                continue;
            }
            XrFrameState fs{ XR_TYPE_FRAME_STATE };
            if (XR_FAILED(xrWaitFrame(session, nullptr, &fs)))
                return;
            g_predicted.store(fs.predictedDisplayTime, std::memory_order_relaxed);
            if (fs.predictedDisplayPeriod > 0)
            {
                const uint32_t hz = uint32_t(std::lround(1e9 / double(fs.predictedDisplayPeriod)));
                if (g_headsetHz.exchange(hz) != hz)
                    Log("vr: headset at %u Hz%s", hz, PaceOn() ? " (the game's frame cap follows it: REGPU_VR_PACE)" : "");
            }
            {
                // F8 in the game window (edge), or regpu::vr::Recentre.
                static bool wasDown = false;
                const bool down = (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
                DWORD pid = 0;
                GetWindowThreadProcessId(GetForegroundWindow(), &pid);
                if (down && !wasDown && pid == GetCurrentProcessId())
                    g_recentre = true;
                wasDown = down;
                if (g_recentre.exchange(false))
                    Recentre(fs.predictedDisplayTime);
            }
            {
                XrViewLocateInfo vli{ XR_TYPE_VIEW_LOCATE_INFO };
                vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                vli.displayTime = fs.predictedDisplayTime;
                vli.space = local;
                XrViewState vs{ XR_TYPE_VIEW_STATE };
                XrView views[2] = { { XR_TYPE_VIEW }, { XR_TYPE_VIEW } };
                uint32_t n = 0;
                if (XR_SUCCEEDED(xrLocateViews(session, &vli, &vs, 2, &n, views)) && n >= 1)
                {
                    std::lock_guard lock(g_poseMutex);
                    g_eyeFov = views[0].fov;
                    g_eyeFovs[0] = views[0].fov;
                    g_eyeFovs[1] = n >= 2 ? views[1].fov : views[0].fov;
                    if (n >= 2)
                    {
                        const float dx = views[1].pose.position.x - views[0].pose.position.x, dy = views[1].pose.position.y - views[0].pose.position.y,
                                    dz = views[1].pose.position.z - views[0].pose.position.z;
                        const float ipd = sqrtf(dx * dx + dy * dy + dz * dz);
                        if (ipd > 0.04f && ipd < 0.09f)
                            g_ipd.store(ipd, std::memory_order_relaxed);
                    }
                }
            }
            xrBeginFrame(session, nullptr);
            XrCompositionLayerQuad layer{ XR_TYPE_COMPOSITION_LAYER_QUAD };
            XrCompositionLayerProjection projection{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
            XrCompositionLayerProjectionView eyes[2] = { { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW }, { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW } };
            const XrCompositionLayerBaseHeader* layers[2] = { reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer), nullptr };
            XrCompositionLayerQuad hudLayer{ XR_TYPE_COMPOSITION_LAYER_QUAD };
            uint32_t layerCount = 0;
            FramePose framePose;
            bool stereo = false, withHud = false;
            if (PaceOn() && fs.shouldRender)
            {
                // Pacing: no new game frame since the last headset frame: wait a little for one (REGPU_VR_FRAME_WAIT_US,
                // default 1500), rather than show the same frame twice. The copy is all this thread draws, so the wait
                // comes out of the time the runtime gave it.
                static const auto wait = std::chrono::microseconds(int(EnvFloat("REGPU_VR_FRAME_WAIT_US", 1500.0f)));
                static uint64_t lastShown = 0;
                std::unique_lock lock(g_capturedMutex);
                g_capturedCv.wait_for(lock, wait, [] { return g_captured.load() != lastShown; });
                lastShown = g_captured.load();
            }
            const bool drawn = fs.shouldRender && DrawQuad(framePose, stereo, withHud);
            // The HUD on a panel in front of the seat, fixed in the room (REGPU_VR_HUD_WIDTH / _DISTANCE / _HEIGHT,
            // metres), its own alpha over the view (premultiplied, as the title drew it): stereo and head-tracked frames.
            auto addHud = [&] {
                if (!withHud)
                    return;
                hudLayer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
                hudLayer.space = local;
                hudLayer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                hudLayer.subImage.swapchain = hud;
                hudLayer.subImage.imageRect = { { 0, 0 }, { int32_t(hudW), int32_t(hudH) } };
                hudLayer.pose.orientation.w = 1;
                hudLayer.pose.position = { 0, hudHeight, -hudDistance };
                hudLayer.size = { hudWidth, hudWidth * float(hudH) / float(std::max(hudW, 1u)) };
                layers[1] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&hudLayer);
                layerCount = 2;
            };
            if (drawn && stereo)
            {
                // A stereo frame: each eye from the head pose moved half the eye distance along the head's x axis (as
                // it was drawn: gpu.cpp stereo::), across the field of view it was drawn with.
                const float tanY = tanf(framePose.fovY * 0.5f), tanX = tanY * float(pairW) / float(std::max(pairH, 1u));
                const auto& p = framePose.pose;
                const float x = p.orientation[0], y = p.orientation[1], z = p.orientation[2], w = p.orientation[3];
                // The head's x axis in the local space: the rotation's first column.
                const float ax[3] = { 1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w) };
                const float half = HalfIpd();
                XrFovf lens[2];
                {
                    std::lock_guard lock(g_poseMutex);
                    lens[0] = g_eyeFovs[0], lens[1] = g_eyeFovs[1];
                }
                for (int e = 0; e < 2; e++)
                {
                    const float sgn = e ? half : -half;
                    eyes[e].pose.orientation = { x, y, z, w };
                    eyes[e].pose.position = { p.position[0] + ax[0] * sgn, p.position[1] + ax[1] * sgn, p.position[2] + ax[2] * sgn };
                    // Only the part each lens shows (LensCrop), at that part's angles.
                    XrRect2Di rect;
                    LensCrop(lens[e], tanX, tanY, pairW, pairH, rect, eyes[e].fov);
                    rect.offset.x += int32_t(e ? pairW : 0);
                    eyes[e].subImage.swapchain = pair;
                    eyes[e].subImage.imageRect = rect;
                }
                if (!densityLogged && recommendedH)
                {
                    // Pixels across the lens vs the headset's own wish: the game's resolution for a 1:1 match.
                    densityLogged = true;
                    const XrRect2Di& r = eyes[0].subImage.imageRect;
                    const float scale = float(recommendedH) / float(std::max(r.extent.height, 1));
                    Log("vr: each eye shows %dx%d of the %ux%u frame across its lens (the headset recommends %ux%u): a resolution of "
                        "about %ux%u matches it",
                        r.extent.width, r.extent.height, pairW, pairH, recommendedW, recommendedH, uint32_t(float(pairW) * scale) & ~1u,
                        uint32_t(float(pairH) * scale) & ~1u);
                }
                projection.space = local;
                projection.viewCount = 2;
                projection.views = eyes;
                layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection);
                layerCount = 1;
                addHud();
                shown++;
            }
            else if (drawn && framePose.valid)
            {
                // A head-tracked frame: both eyes see it from the pose it was rendered with (mono), across the field
                // of view it was rendered with; the runtime turns it for the head's movement since.
                const float tanY = tanf(framePose.fovY * 0.5f), tanX = tanY * float(quadW) / float(std::max(quadH, 1u));
                for (XrCompositionLayerProjectionView& e : eyes)
                {
                    const auto& p = framePose.pose;
                    e.pose.orientation = { p.orientation[0], p.orientation[1], p.orientation[2], p.orientation[3] };
                    e.pose.position = { p.position[0], p.position[1], p.position[2] };
                    e.fov = { -atanf(tanX), atanf(tanX), atanf(tanY), -atanf(tanY) };
                    e.subImage.swapchain = quad;
                    e.subImage.imageRect = { { 0, 0 }, { int32_t(quadW), int32_t(quadH) } };
                }
                projection.space = local;
                projection.viewCount = 2;
                projection.views = eyes;
                layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection);
                layerCount = 1;
                addHud();
                shown++;
            }
            else if (drawn)
            {
                layer.space = local;
                layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                layer.subImage.swapchain = quad;
                layer.subImage.imageRect = { { 0, 0 }, { int32_t(quadW), int32_t(quadH) } };
                layer.pose.orientation.w = 1;
                layer.pose.position = { 0, height, -distance };
                layer.size = { width, width * float(quadH) / float(std::max(quadW, 1u)) };
                layerCount = 1;
                shown++;
            }
            XrFrameEndInfo end{ XR_TYPE_FRAME_END_INFO };
            end.displayTime = fs.predictedDisplayTime;
            end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            end.layerCount = layerCount;
            end.layers = layers;
            if (XrResult r = xrEndFrame(session, &end); XR_FAILED(r))
                Log("vr: xrEndFrame failed: %s", ResultText(instance, r).c_str());
            frames++;
            if (std::chrono::steady_clock::now() - lastLog > std::chrono::seconds(30))
            {
                lastLog = std::chrono::steady_clock::now();
                Log("vr: %llu headset frames, %llu with the screen, %llu game frames captured, %llu tags without a pose", (unsigned long long)frames,
                    (unsigned long long)shown, (unsigned long long)g_captured.load(), (unsigned long long)g_posesMissing.load());
            }
        }
    }
};

void Thread()
{
    SetThreadDescription(GetCurrentThread(), L"regpu vr");
    if (!LoadLoader())
        return;
    bool quiet = false;
    for (;;)
    {
        {
            Session s;
            if (s.Create(quiet))
            {
                quiet = false;
                s.Run();
                Log("vr: headset session ended");
            }
            else
                quiet = true;
        }
        Sleep(3000);
    }
}
} // namespace

namespace regpu::xr
{
bool Wanted()
{
    const char* v = getenv("REGPU_VR");
    return v && *v && strcmp(v, "0") != 0 && strcmp(v, "off") != 0;
}

void Start(ID3D12Device* device, ID3D12CommandQueue* queue)
{
    static bool started = false;
    if (started || !Wanted())
        return;
    started = true;
    g_device = device;
    g_queue = queue;
    Log("vr: on (%s): waiting for a headset", getenv("REGPU_VR"));
    std::thread(Thread).detach();
}

bool StereoFrame(float& clipPerMetre, float& halfIpd)
{
    if (!StereoMode())
        return false;
    // REGPU_VR_STEREO_TEST=1 (live): stereo without a headset, for screenshots (the desktop shows the eye drawn second;
    // REGPU_VR_SHOW=left (gpu.cpp) swaps the order).
    static LiveSwitch test("REGPU_VR_STEREO_TEST");
    if (!(g_capturing.load(std::memory_order_relaxed) || test) || !g_lastPosed.load(std::memory_order_relaxed))
        return false;
    const float fovY = g_lastFovY.load(std::memory_order_relaxed);
    if (fovY < 0.1f || fovY > 3.0f)
        return false;
    // P00 of a 16:9 symmetric projection with that vertical field of view (the game's main view).
    clipPerMetre = 1.0f / (tanf(fovY * 0.5f) * (16.0f / 9.0f));
    halfIpd = HalfIpd();
    return true;
}

ID3D12Resource* StereoTarget(int eye, uint32_t w, uint32_t h)
{
    static LiveSwitch test("REGPU_VR_STEREO_TEST"); // (the images are made without a headset too)
    if (eye < 0 || eye > 2 || !(g_capturing.load(std::memory_order_relaxed) || test) || !g_device)
        return nullptr;
    std::lock_guard lock(g_screenMutex);
    ComPtr<ID3D12Resource>& t = g_eyes[eye];
    if (!t || t->GetDesc().Width != w || t->GetDesc().Height != h)
    {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
        ComPtr<ID3D12Resource> made;
        if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&made))))
            return nullptr;
        if (t)
            g_oldScreens.push_back(t);
        while (g_oldScreens.size() > 8)
            g_oldScreens.erase(g_oldScreens.begin());
        t = made;
    }
    if (eye == 1)
        g_eyesThisFrame = true;
    return t.Get();
}

void StereoHudCaptured() { g_hudThisFrame = true; }

void SetCaptureTag(uint32_t tag)
{
    g_captureTag = tag;
    // Every 600th frame end: the frame's tag beside the newest one handed in (how many logic frames the drawing trails).
    static uint32_t calls = 0;
    if (tag && ++calls % 600 == 0)
        Log("vr: frame drawn with pose tag %u (newest handed in %u)", tag, g_newestTag.load(std::memory_order_relaxed));
}

bool MonoHudPanel()
{
    static LiveSwitch test("REGPU_VR_STEREO_TEST");
    return CockpitMode() && (g_capturing.load(std::memory_order_relaxed) || test) && g_lastPosed.load(std::memory_order_relaxed);
}

uint32_t PaceCap()
{
    if (!PaceOn() || !(CockpitMode() || StereoMode()) || !g_capturing.load(std::memory_order_relaxed))
        return 0;
    static const float mult = std::clamp(EnvFloat("REGPU_VR_PACE_MULT", 1.0f), 0.25f, 4.0f);
    const uint32_t hz = g_headsetHz.load(std::memory_order_relaxed);
    return hz >= 30 ? uint32_t(std::lround(float(hz) * mult)) : 0;
}

void CaptureFrame(ID3D12GraphicsCommandList* list, ID3D12Resource* backBuffer)
{
    if (!g_capturing.load(std::memory_order_relaxed))
        return;
    D3D12_RESOURCE_DESC d = backBuffer->GetDesc();
    ComPtr<ID3D12Resource> screen;
    {
        std::lock_guard lock(g_screenMutex);
        if (!g_screen || g_screen->GetDesc().Width != d.Width || g_screen->GetDesc().Height != d.Height)
        {
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
            ComPtr<ID3D12Resource> made;
            if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&made))))
                return;
            if (g_screen)
                g_oldScreens.push_back(g_screen);
            if (g_oldScreens.size() > 8)
                g_oldScreens.erase(g_oldScreens.begin());
            g_screen = made;
        }
        screen = g_screen;
        // This captured frame's pose: the one its main view was drawn with, by the tag the draw thread read from it
        // (SetCaptureTag). No tag: not head-tracked (the virtual screen).
        g_screenPose.valid = false;
        if (const uint32_t tag = g_captureTag.exchange(0))
        {
            std::lock_guard tl(g_taggedMutex);
            const TaggedPose& t = g_tagged[tag & 1023];
            if (t.tag == tag)
                g_screenPose = t.pose;
            else
                g_posesMissing++;
        }
        g_screenStereo = g_eyesThisFrame.exchange(false);
        g_screenHud = g_hudThisFrame.exchange(false);
    }
    Barrier(list, backBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(list, screen.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(screen.Get(), backBuffer);
    Barrier(list, screen.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Barrier(list, backBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    {
        std::lock_guard lock(g_capturedMutex);
        g_captured++;
    }
    g_capturedCv.notify_all();
}
}

namespace regpu::vr
{
bool GetHeadPose(HeadPose& out)
{
    std::lock_guard lock(g_poseMutex);
    if (!g_poseSession)
        return false;
    XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
    const XrTime time = g_predicted.load(std::memory_order_relaxed);
    if (!time || XR_FAILED(xrLocateSpace(g_poseView, g_poseLocal, time, &loc)))
        return false;
    if (!(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
        return false;
    out.orientation[0] = loc.pose.orientation.x, out.orientation[1] = loc.pose.orientation.y;
    out.orientation[2] = loc.pose.orientation.z, out.orientation[3] = loc.pose.orientation.w;
    const bool hasPosition = loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT;
    out.position[0] = hasPosition ? loc.pose.position.x : 0.0f;
    out.position[1] = hasPosition ? loc.pose.position.y : 0.0f;
    out.position[2] = hasPosition ? loc.pose.position.z : 0.0f;
    out.fovLeft = g_eyeFov.angleLeft, out.fovRight = g_eyeFov.angleRight, out.fovUp = g_eyeFov.angleUp, out.fovDown = g_eyeFov.angleDown;
    return true;
}

void Recentre() { g_recentre = true; }

void SubmitFramePose(const HeadPose* pose, float renderFovY)
{
    g_lastPosed.store(pose != nullptr, std::memory_order_relaxed);
    g_lastFovY.store(renderFovY, std::memory_order_relaxed);
}

void SubmitTaggedPose(uint32_t tag, const HeadPose& pose, float renderFovY)
{
    std::lock_guard lock(g_taggedMutex);
    TaggedPose& t = g_tagged[tag & 1023];
    g_newestTag.store(tag, std::memory_order_relaxed);
    t.tag = tag;
    t.pose.valid = true;
    t.pose.pose = pose;
    t.pose.fovY = renderFovY;
}
}

#endif
