// Intel XeSS Super Resolution (FM3 port session 43, the Intel iGPU work: N8). The renderer hands over the scene's
// colour, depth and motion vectors at render size and gets the upscaled image in the output texture, as for DLSS
// (TemporalEvaluate in renderer.cpp; the same DlssSetup/DlssFrame). Runs on any GPU with Shader Model 6.4 (the DP4a
// path; Intel Arc uses its XMX units). libxess.dll is loaded at run time from beside the executable, so a build or
// an install without it just reports XeSS unavailable (the renderer then upscales with FSR 1). Headers from Intel's
// XeSS SDK (REGPU_XESS_DIR at build time; the SDK's licence lets libxess.dll be redistributed unmodified).
#include "gpu/dlss.h"
#include "gpu/xess.h"

#include <windows.h>
#include <xess/xess.h>
#include <xess/xess_d3d12.h>

#include <string>

#include "log.h"

namespace
{
struct Api
{
    decltype(&xessGetVersion) getVersion = nullptr;
    decltype(&xessD3D12CreateContext) createContext = nullptr;
    decltype(&xessD3D12Init) init = nullptr;
    decltype(&xessD3D12Execute) execute = nullptr;
    decltype(&xessDestroyContext) destroyContext = nullptr;
    decltype(&xessSetVelocityScale) setVelocityScale = nullptr;
    decltype(&xessSetJitterScale) setJitterScale = nullptr;
};

// One context per setup (sizes, quality, HDR), kept: FM3 switches between an HDR scene and LDR menus several times
// while loading, and each re-initialisation is a blocking call (as the FSR contexts, ReGPU patch regpu-fsr4).
struct Context
{
    xess_context_handle_t handle = nullptr;
    DlssSetup setup{};
    uint64_t lastUse = 0;
};
constexpr size_t kContexts = 4;

struct Xess
{
    bool tried = false, available = false;
    Api api;
    ID3D12Device* device = nullptr;
    Context contexts[kContexts];
    uint64_t uses = 0;
    uint32_t version[3] = {};
} g_xess;

template <typename T>
bool Load(HMODULE m, const char* name, T& out)
{
    out = reinterpret_cast<T>(GetProcAddress(m, name));
    return out != nullptr;
}
} // namespace

bool XessInit(ID3D12Device* device)
{
    if (g_xess.tried)
        return g_xess.available;
    g_xess.tried = true;
    g_xess.device = device;
    wchar_t exe[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring path = exe;
    path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"libxess.dll";
    HMODULE m = LoadLibraryW(path.c_str());
    if (!m)
    {
        Log("xess: libxess.dll not found beside the executable: XeSS unavailable (FSR 1 instead)");
        return false;
    }
    Api& a = g_xess.api;
    if (!Load(m, "xessGetVersion", a.getVersion) || !Load(m, "xessD3D12CreateContext", a.createContext) || !Load(m, "xessD3D12Init", a.init) ||
        !Load(m, "xessD3D12Execute", a.execute) || !Load(m, "xessDestroyContext", a.destroyContext) ||
        !Load(m, "xessSetVelocityScale", a.setVelocityScale) || !Load(m, "xessSetJitterScale", a.setJitterScale))
    {
        Log("xess: libxess.dll lacks an expected export: XeSS unavailable");
        return false;
    }
    xess_version_t v{};
    a.getVersion(&v);
    if (a.createContext(device, &g_xess.contexts[0].handle) != XESS_RESULT_SUCCESS)
    {
        Log("xess: no context on this GPU (needs Shader Model 6.4): XeSS unavailable");
        return false;
    }
    g_xess.available = true;
    Log("xess: XeSS %u.%u.%u ready", v.major, v.minor, v.patch);
    return true;
}

bool XessEvaluate(ID3D12GraphicsCommandList* list, const DlssSetup& setup, const DlssFrame& frame)
{
    if (!g_xess.available)
        return false;
    const Api& a = g_xess.api;
    // This setup's context; else a new one (or the least recently used re-initialised). Init is blocking (it builds
    // XeSS's pipelines the first time); a context being re-initialised was last used frames ago.
    Context* ctx = nullptr;
    for (Context& c : g_xess.contexts)
        if (c.handle && c.setup.outputWidth && c.setup == setup)
            ctx = &c;
    if (!ctx)
    {
        Context* pick = nullptr;
        for (Context& c : g_xess.contexts)
            if (!c.setup.outputWidth && (c.handle || !pick || pick->handle == nullptr))
            {
                pick = &c; // a free slot (one with a handle but no setup first: the context XessInit made)
                if (c.handle)
                    break;
            }
        if (!pick)
        {
            pick = &g_xess.contexts[0];
            for (Context& c : g_xess.contexts)
                if (c.lastUse < pick->lastUse)
                    pick = &c;
        }
        if (!pick->handle && a.createContext(g_xess.device, &pick->handle) != XESS_RESULT_SUCCESS)
        {
            Log("xess: another context could not be made");
            return false;
        }
        // Our quality steps (0 native, 1 quality 1.5x, 2 balanced 1.7x, 3 performance 2x, 4 ultra performance 3x) are
        // XeSS's ratios.
        static const xess_quality_settings_t qualities[] = { XESS_QUALITY_SETTING_AA, XESS_QUALITY_SETTING_QUALITY, XESS_QUALITY_SETTING_BALANCED,
            XESS_QUALITY_SETTING_PERFORMANCE, XESS_QUALITY_SETTING_ULTRA_PERFORMANCE };
        xess_d3d12_init_params_t p{};
        p.outputResolution = { setup.outputWidth, setup.outputHeight };
        p.qualitySetting = qualities[setup.quality > 4 ? 1 : setup.quality];
        p.initFlags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE | (setup.depthInverted ? XESS_INIT_FLAG_INVERTED_DEPTH : 0) |
            (setup.hdr ? 0 : XESS_INIT_FLAG_LDR_INPUT_COLOR);
        const xess_result_t r = a.init(pick->handle, &p);
        if (r != XESS_RESULT_SUCCESS)
        {
            Log("xess: init failed (%d) for %ux%u -> %ux%u", int(r), setup.renderWidth, setup.renderHeight, setup.outputWidth, setup.outputHeight);
            pick->setup = {};
            return false;
        }
        pick->setup = setup;
        ctx = pick;
        Log("xess: %ux%u -> %ux%u, quality %u%s", setup.renderWidth, setup.renderHeight, setup.outputWidth, setup.outputHeight, setup.quality,
            setup.hdr ? ", HDR" : "");
    }
    // Motion in the texture's units, scaled to render pixels (current to previous, as DLSS); jitter in render pixels.
    a.setVelocityScale(ctx->handle, frame.motionScaleX, frame.motionScaleY);
    a.setJitterScale(ctx->handle, 1.0f, 1.0f);
    xess_d3d12_execute_params_t e{};
    e.pColorTexture = frame.color;
    e.pVelocityTexture = frame.motion;
    e.pDepthTexture = frame.depth;
    e.pOutputTexture = frame.output;
    e.jitterOffsetX = frame.jitterX;
    e.jitterOffsetY = frame.jitterY;
    e.exposureScale = 1.0f;
    e.resetHistory = frame.reset ? 1 : 0;
    e.inputWidth = setup.renderWidth;
    e.inputHeight = setup.renderHeight;
    ctx->lastUse = ++g_xess.uses;
    const xess_result_t r = a.execute(ctx->handle, list, &e);
    if (r != XESS_RESULT_SUCCESS)
    {
        static int logged = 0;
        if (logged++ < 3)
            Log("xess: execute failed (%d)", int(r));
        return false;
    }
    return true;
}
