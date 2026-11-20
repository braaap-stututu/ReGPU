// Hyper presets and cuts (include/regpu/hyper.h). Ported from the FM2 port's hyper-prototype branch
// (src/gpu/hyper.cpp there, MIT), adapted to this tree: cuts this tree already had under older names (fetch
// specialisation, the alpha-test-free shaders, 10-bit copies, helper scale, background-optimised shaders) are set
// through those switches; the rest are REGPU_HYPER_* switches read where the renderer and translator use them.
#include "regpu/hyper.h"

#include "gpu/vk/device.h"
#include "render/presenter.h"

#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <windows.h>

namespace regpu::hypermode
{
namespace
{
Settings g_settings;

// The environment, else the port's setting (when set: >= 0), else the preset's default.
int Value(const char* name, int setting, int fallback)
{
    const char* v = getenv(name);
    if (v && *v)
        return atoi(v);
    return setting >= 0 ? setting : fallback;
}

void Default(const char* name, const char* value)
{
    if (!getenv(name))
        _putenv_s(name, value);
}

void Default(const char* name, int value)
{
    Default(name, std::to_string(value).c_str());
}

struct GpuClass
{
    uint32_t vendor = 0;     // PCI vendor: 0x8086 Intel, 0x1002 AMD, 0x10DE NVIDIA
    bool integrated = false; // 512 MB or less of dedicated memory: shares system memory
};

// The GPU the renderer uses (REGPU_ADAPTER picks it: vk::CreateDevice).
GpuClass DetectGpu()
{
    GpuClass g;
    const vk::Device& d = RenderVkDevice();
    g.vendor = d.props.vendorID;
    g.integrated = d.uma;
    return g;
}

Preset Parse(const char* v)
{
    if (!v || !*v)
        return Preset::Off;
    if (!_stricmp(v, "mild") || !strcmp(v, "1"))
        return Preset::Mild;
    if (!_stricmp(v, "low") || !strcmp(v, "2"))
        return Preset::Low;
    if (!_stricmp(v, "extreme") || !strcmp(v, "3"))
        return Preset::Extreme;
    if (!_stricmp(v, "potato") || !strcmp(v, "4"))
        return Preset::Potato;
    return Preset::Off;
}

Preset g_level = Preset::Off;
bool g_configured = false;
} // namespace

void Configure(const Settings& settings)
{
    g_settings = settings;
    const char* env = getenv("REGPU_HYPER");
    const std::string pick = env && *env ? env : g_settings.preset;
    if (!_stricmp(pick.c_str(), "auto"))
    {
        const GpuClass gpu = DetectGpu();
        g_level = gpu.integrated ? Preset::Potato : Preset::Off;
        Log("hyper: auto picked %s (GPU vendor %04X, %s)", Name(g_level), gpu.vendor, gpu.integrated ? "integrated" : "dedicated");
    }
    else
        g_level = Parse(pick.c_str());
    g_configured = true;
    const Preset p = g_level;
    const int lvl = int(p);

    // Lossless cuts in every preset (FM2: 10:10:10:2 resolves -0.5 ms, no-discard +15% on Intel).
    if (Value("REGPU_HYPER_RESOLVE1010", g_settings.resolve1010, p != Preset::Off) != 0)
        Default("REGPU_COPIES_1010102", "1");
    if (Value("REGPU_HYPER_NODISCARD", g_settings.noDiscard, p != Preset::Off) != 0)
        Default("REGPU_SPLIT_ALPHA_TEST", "1");

    // Helper surfaces (reflection views, blur and bloom levels): Mild 75%, Low/Extreme 50%, Potato 25%.
    static const int helpersBy[] = { 100, 75, 50, 50, 25 };
    const int helpers = std::clamp(Value("REGPU_HYPER_HELPERS", g_settings.helpers, helpersBy[lvl]), 25, 100);
    if (helpers < 100)
        Default("REGPU_HELPER_SCALE", std::to_string(helpers / 100.0).c_str());

    // The lossless shader cuts as one group (FM2 Intel set: +213% on an HD 630 with the rest of Potato). This tree
    // already had fetch specialisation (REGPU_FETCH_SPECIALIZE) and optimised shaders (REGPU_OPTIMIZE_SHADERS, built
    // in the background instead of at first load, so no multi-minute first boot); the rest are new here.
    if (Value("REGPU_HYPER_SHADER_CUTS", g_settings.shaderCuts, p == Preset::Potato) != 0)
    {
        if (Value("REGPU_HYPER_SHADER_OPT", -1, 1) != 0)
            Default("REGPU_OPTIMIZE_SHADERS", "1");
        if (Value("REGPU_HYPER_FETCHSPEC", -1, 1) != 0)
            Default("REGPU_FETCH_SPECIALIZE", "1");
        Default("REGPU_HYPER_PLAIN_MUL", "5");
        Default("REGPU_HYPER_SWIZZLE_SEL", "1");
        Default("REGPU_HYPER_TAILSPEC", "1");
        Default("REGPU_HYPER_RESOLVE_BY_SIZE", "1");
        Default("REGPU_HYPER_VCACHE", "2"); // triangles reordered for the vertex cache and overdraw (lossless; meshoptimizer)
    }

    // Mesh decimation (meshoptimizer): big meshes keep this percent of their triangles. Extreme 25, Potato 10.
    const int decimate = Value("REGPU_HYPER_DECIMATE", g_settings.decimate, p == Preset::Extreme ? 25 : p == Preset::Potato ? 10 : 0);
    if (decimate > 0 && decimate < 100)
        Default("REGPU_HYPER_DECIMATE", decimate);

    // Draw culling by pixel count (Potato 256 px, with the probe, 200% hysteresis and the pin: the version that
    // gained on both Intel and AMD and does not flicker).
    const int cull = Value("REGPU_HYPER_CULL_PIXELS", g_settings.cullPixels, p == Preset::Potato ? 256 : -1);
    if (cull >= 0)
    {
        Default("REGPU_HYPER_CULL_PIXELS", cull);
        Default("REGPU_HYPER_CULL_PROBE", "1");
        Default("REGPU_HYPER_CULL_HYST", "200");
        Default("REGPU_HYPER_CULL_PASSES", "1");
        Default("REGPU_HYPER_CULL_ALL", "1");
    }

    // Frozen 3D helper views: drawn normally for 120 frames, then every 4th frame.
    const int freeze = Value("REGPU_HYPER_FREEZE", g_settings.freeze, p == Preset::Potato ? 120 : 0);
    if (freeze > 0)
    {
        Default("REGPU_HYPER_FREEZE", freeze);
        Default("REGPU_HYPER_FREEZE_REFRESH", "4");
    }
}

Preset Level()
{
    if (!g_configured)
        Configure({});
    return g_level;
}

const char* Name(Preset preset)
{
    switch (preset)
    {
    case Preset::Mild: return "mild";
    case Preset::Low: return "low";
    case Preset::Extreme: return "extreme";
    case Preset::Potato: return "potato";
    default: return "off";
    }
}

uint32_t SceneScalePercent()
{
    static const int byPreset[] = { 0, 50, 35, 25, 0 };
    const int v = Value("REGPU_HYPER_SCENE", g_settings.scenePercent, byPreset[int(Level())]);
    return v <= 0 || v >= 100 ? 0u : uint32_t(std::clamp(v, 25, 100));
}

bool CheapUpscale()
{
    const bool byPreset = Level() == Preset::Low || Level() == Preset::Extreme;
    return Value("REGPU_HYPER_UPSCALE", g_settings.cheapUpscale, byPreset ? 1 : 0) != 0;
}

uint32_t AnisotropyCap()
{
    static const int byPreset[] = { 16, 4, 2, 2, 0 };
    const int v = Value("REGPU_HYPER_ANISO", g_settings.aniso, byPreset[int(Level())]);
    return v >= 16 ? 16u : v >= 8 ? 8u : v >= 4 ? 4u : v >= 2 ? 2u : 0u;
}

uint32_t HelperScalePercent()
{
    static const int byPreset[] = { 100, 75, 50, 50, 25 };
    return uint32_t(std::clamp(Value("REGPU_HYPER_HELPERS", g_settings.helpers, byPreset[int(Level())]), 25, 100));
}

int CullPixels()
{
    const char* v = getenv("REGPU_HYPER_CULL_PIXELS");
    return v && *v ? atoi(v) : -1;
}

void LogSummary()
{
    const uint32_t scene = SceneScalePercent();
    Log("hyper: preset %s; scene scale %s; upscale %s", Name(Level()), scene ? (std::to_string(scene) + "%").c_str() : "as set",
        CheapUpscale() ? "bilinear, one pass" : "as set");
    auto on = [](const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; };
    if (on("REGPU_HYPER_PLAIN_MUL"))
        Log("hyper: shader cuts (plain multiplies where no inf/NaN can reach, swizzle selects, tail specialisation, fetch specialisation%s)",
            on("REGPU_OPTIMIZE_SHADERS") ? ", optimised builds" : "");
    if (CullPixels() >= 0)
        Log("hyper: draws of %d pixels or fewer culled (probe %s, hysteresis %s%%)", CullPixels(), on("REGPU_HYPER_CULL_PROBE") ? "on" : "off",
            getenv("REGPU_HYPER_CULL_HYST") ? getenv("REGPU_HYPER_CULL_HYST") : "100");
    if (on("REGPU_HYPER_FREEZE"))
        Log("hyper: 3D helper views frozen after %s frames, redrawn every %s", getenv("REGPU_HYPER_FREEZE"),
            getenv("REGPU_HYPER_FREEZE_REFRESH") ? getenv("REGPU_HYPER_FREEZE_REFRESH") : "never");
    if (on("REGPU_HYPER_DECIMATE"))
        Log("hyper: big meshes keep %s%% of their triangles", getenv("REGPU_HYPER_DECIMATE"));
    if (on("REGPU_HYPER_VCACHE"))
        Log("hyper: mesh triangles reordered for the vertex cache%s", atoi(getenv("REGPU_HYPER_VCACHE")) >= 2 ? " and overdraw" : "");
    if (on("REGPU_COPIES_1010102"))
        Log("hyper: 10:10:10:2 copies kept at 32 bits");
    if (HelperScalePercent() < 100)
        Log("hyper: offscreen helper surfaces at %u%% size", HelperScalePercent());
    if (AnisotropyCap() < 16)
        Log("hyper: anisotropic filtering at most %ux", AnisotropyCap());
}
} // namespace regpu::hypermode
