// AMD FSR 3 / FSR 4 upscaling (FidelityFX API, MIT): the same inputs as DLSS (see dlss.h). Needs
// amd_fidelityfx_loader_dx12.dll and amd_fidelityfx_upscaler_dx12.dll next to the executable (copied there by the build).
// The upscaler DLL (4.0.x in SDK 2.1) holds several versions: FSR 4 (machine-learned, RDNA 4 GPUs) and FSR 3.1 (any
// D3D12 GPU). DlssSetup::fsr4 picks the newest 4.x the device offers, else the newest 3.x (FM3 port, session 36).
//
// Contexts are kept, one per setup (FM3 port, session 36): a setup change (the scene image switching between 8-bit and
// HDR formats while a track loads) used to destroy the context at once and create another, while the GPU had not yet
// run the work already recorded with the old one (this frame's list is still open, earlier frames may be in flight):
// its history and work textures were freed under the GPU's feet, and at 4K the AMD driver timed out (FM3 OPEN-ISSUES U,
// six re-creations in 23 s, then the display reset). A switch between kept contexts resets that context's history.
#include "gpu/fsr3.h"

#include <cstring>
#include <string>
#include <vector>

#include <ffx_api.h>
#include <ffx_upscale.h>
#include <dx12/ffx_api_dx12.h>

#include "log.h"

namespace
{
struct Fsr3Context
{
    DlssSetup setup{};
    ffxContext context = nullptr;
};
struct Fsr3
{
    bool failed = false;
    std::vector<Fsr3Context> contexts; // at most kMaxContexts, for the life of the device
    const void* last = nullptr;        // the context of the last dispatch (a different one starts without history)
    bool listed = false;
    uint64_t fsr4Version = 0, fsr3Version = 0;
    std::string fsr4Name, fsr3Name;
} g_fsr3;
constexpr size_t kMaxContexts = 4;

// The major version in a provider's display name ("FSR 4.0.3" -> 4; 0 if none).
int MajorOf(const char* name)
{
    for (const char* p = name; p && *p; p++)
        if (*p >= '0' && *p <= '9')
            return atoi(p);
    return 0;
}

// The upscaler versions this device offers, listed once: the newest 4.x and 3.x by their order (the API lists the
// default, newest, first).
void ListVersions(ID3D12Device* device)
{
    if (g_fsr3.listed)
        return;
    g_fsr3.listed = true;
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device = device;
    q.outputCount = &count;
    if (ffxQuery(nullptr, &q.header) != FFX_API_RETURN_OK || !count)
    {
        Log("fsr: no upscaler versions listed (the default one is used)");
        return;
    }
    std::vector<uint64_t> ids(count);
    std::vector<const char*> names(count);
    q.versionIds = ids.data();
    q.versionNames = names.data();
    if (ffxQuery(nullptr, &q.header) != FFX_API_RETURN_OK)
        return;
    std::string all;
    for (uint64_t i = 0; i < count; i++)
    {
        const char* name = names[i] ? names[i] : "?";
        all += (i ? ", " : "") + std::string(name);
        const int major = MajorOf(name);
        if (major >= 4 && !g_fsr3.fsr4Version)
            g_fsr3.fsr4Version = ids[i], g_fsr3.fsr4Name = name;
        if (major == 3 && !g_fsr3.fsr3Version)
            g_fsr3.fsr3Version = ids[i], g_fsr3.fsr3Name = name;
    }
    Log("fsr: upscaler versions on this GPU: %s", all.c_str());
}

Fsr3Context* ContextFor(ID3D12Device* device, const DlssSetup& setup)
{
    for (Fsr3Context& c : g_fsr3.contexts)
        if (c.setup == setup)
            return &c;
    if (g_fsr3.contexts.size() >= kMaxContexts)
    {
        static bool logged = false;
        if (!logged)
            Log("fsr: %zu contexts already: this setup falls back to FSR 1", kMaxContexts);
        logged = true;
        return nullptr;
    }
    ListVersions(device);
    g_fsr3.contexts.reserve(kMaxContexts); // (dispatches use the element's address)
    uint64_t version = setup.fsr4 ? g_fsr3.fsr4Version : g_fsr3.fsr3Version;
    const char* versionName = setup.fsr4 ? g_fsr3.fsr4Name.c_str() : g_fsr3.fsr3Name.c_str();
    if (setup.fsr4 && !version)
    {
        static bool logged = false;
        if (!logged)
            Log("fsr: FSR 4 is not offered on this GPU (RDNA 4 / RX 9000 series): FSR 3 instead");
        logged = true;
        version = g_fsr3.fsr3Version;
        versionName = g_fsr3.fsr3Name.c_str();
    }
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = device;
    // SDK 2.1: the API version the program was built against must be chained (super-resolution-ml.md, "New APIs").
    ffxCreateContextDescUpscaleVersion apiVersion{};
    apiVersion.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    apiVersion.version = FFX_UPSCALER_VERSION;
    apiVersion.header.pNext = &backend.header;
    ffxOverrideVersion pick{};
    pick.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    pick.versionId = version;
    if (version)
        backend.header.pNext = &pick.header;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &apiVersion.header;
    create.maxRenderSize = { setup.renderWidth, setup.renderHeight };
    create.maxUpscaleSize = { setup.outputWidth, setup.outputHeight };
    create.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | (setup.depthInverted ? FFX_UPSCALE_ENABLE_DEPTH_INVERTED : 0) |
        (setup.hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0);
    Fsr3Context c;
    c.setup = setup;
    const ffxReturnCode_t r = ffxCreateContext(&c.context, &create.header, nullptr);
    if (r != FFX_API_RETURN_OK)
    {
        Log("fsr: create failed (%d): amd_fidelityfx_*_dx12.dll missing?", int(r));
        if (g_fsr3.contexts.empty())
            g_fsr3.failed = true;
        return nullptr;
    }
    Log("fsr: %s, %ux%u -> %ux%u%s (context %zu)", *versionName ? versionName : "default version", setup.renderWidth, setup.renderHeight,
        setup.outputWidth, setup.outputHeight, setup.hdr ? ", HDR" : "", g_fsr3.contexts.size() + 1);
    g_fsr3.contexts.push_back(c);
    return &g_fsr3.contexts.back();
}
} // namespace

// On the draw thread (FM3 port, session 36): the context for `setup`, made on first use; *handle and *switched (a
// different context than the last dispatch's: no history) go to Fsr3Dispatch, which runs where the list replays.
bool Fsr3Prepare(ID3D12Device* device, const DlssSetup& setup, void** handle, bool* switched)
{
    if (g_fsr3.failed)
        return false;
    Fsr3Context* c = ContextFor(device, setup);
    if (!c)
        return false;
    *switched = g_fsr3.last != c->context;
    g_fsr3.last = c->context;
    *handle = c->context;
    return true;
}

bool Fsr3Dispatch(ID3D12GraphicsCommandList* list, void* handle, bool switched, const DlssSetup& setup, const DlssFrame& frame, float sharpness)
{
    ffxContext context = handle;
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = list;
    d.color = ffxApiGetResourceDX12(frame.color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = ffxApiGetResourceDX12(frame.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = ffxApiGetResourceDX12(frame.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = ffxApiGetResourceDX12(frame.output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = { frame.jitterX, frame.jitterY };
    d.motionVectorScale = { frame.motionScaleX, frame.motionScaleY };
    d.renderSize = { setup.renderWidth, setup.renderHeight };
    d.upscaleSize = { setup.outputWidth, setup.outputHeight };
    d.enableSharpening = sharpness > 0.0f;
    d.sharpness = sharpness;
    d.frameTimeDelta = frame.frameTimeMs;
    d.preExposure = 1.0f;
    d.reset = frame.reset || switched;
    // The game's camera is unknown here; typical values (used for depth-based heuristics).
    d.cameraNear = 0.1f;
    d.cameraFar = 10000.0f;
    d.cameraFovAngleVertical = 1.0f;
    d.viewSpaceToMetersFactor = 1.0f;
    const ffxReturnCode_t r = ffxDispatch(&context, &d.header);
    if (r != FFX_API_RETURN_OK)
    {
        static int logged = 0;
        if (logged++ < 3)
            Log("fsr: dispatch failed (%d)", int(r));
        return false;
    }
    return true;
}

bool Fsr3Evaluate(ID3D12Device* device, ID3D12GraphicsCommandList* list, const DlssSetup& setup, const DlssFrame& frame, float sharpness)
{
    void* handle = nullptr;
    bool switched = false;
    return Fsr3Prepare(device, setup, &handle, &switched) && Fsr3Dispatch(list, handle, switched, setup, frame, sharpness);
}
