// Config and Host storage, and Start (include/regpu/regpu.h).
#include <cmath>
#include "settings.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "debug.h"
#include "memory.h"
#include "gpu/renderer.h"

void PresenterInit();
bool PresenterHasFocus();
void GpuLogState();
void GpuInit();

namespace regpu
{
Config& GetConfig()
{
    static Config config;
    return config;
}

Host& GetHost()
{
    static Host host;
    return host;
}

void Start()
{
    g_base = GetHost().base;
    DebugServerStart(); // REGPU_DEBUG=1: tools/regpuctl.py drives the running game
    PresenterInit();
    GpuInit();
}

// Supersampling (FM3 port, session 36): the game renders at ssaa x the shown size, and the frame is filtered down to it
// at the present (renderer.cpp, SsaaDownsample). The swap chain stays the shown size (ShownSize).
void Config::RenderSize(uint32_t& w, uint32_t& h) const
{
    ShownSize(w, h);
    // Triple screen (REGPU_TRIPLE, FM3 port session 39): the resolution is the three screens together; each screen's
    // pass renders at a third of its width (no supersampling).
    static const bool triple = [] {
        const char* v = getenv("REGPU_TRIPLE");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (triple)
    {
        // Each screen 16:9 (the title's frame): a third of the width, the height fitted to it.
        w = (w / 3) & ~1u;
        h = std::min<uint32_t>(h, w * 9 / 16) & ~1u;
        w = std::min<uint32_t>(w, h * 16 / 9) & ~1u;
        return;
    }
    if (ssaa > 1.0f)
    {
        w = uint32_t(std::lround(double(w) * ssaa)) & ~1u;
        h = uint32_t(std::lround(double(h) * ssaa)) & ~1u;
    }
}

void Config::ShownSize(uint32_t& w, uint32_t& h) const
{
    w = width;
    h = height;
    // Triple screen: the resolution is the three screens together and each is 16:9 (RenderSize); a forced aspect ratio
    // (a single screen's 16:9) would shrink all three into a third (FM3 port session 39: 853x480 per screen at 7680x1440).
    if (const char* t = getenv("REGPU_TRIPLE"); t && *t && strcmp(t, "0") != 0)
        return;
    // The forced aspect, or the resolution's own; never narrower than 16:9 (narrower views
    // are not supported yet: such resolutions get 16:9 with bars).
    uint64_t aw = aspectW, ah = aspectH;
    if (!aw || !ah)
        aw = width, ah = height;
    if (aw * 9 < ah * 16)
        aw = 16, ah = 9;
    // Fit that aspect inside the resolution.
    if (uint64_t(width) * ah > uint64_t(height) * aw)
        w = uint32_t(uint64_t(height) * aw / ah) & ~1u;
    else
        h = uint32_t(uint64_t(width) * ah / aw) & ~1u;
}

float Config::SceneScale() const
{
    if (upscaler == Upscaler::Off)
        return 1.0f;
    // The usual presets (native, 1.5x, 1.7x, 2x, 3x per axis).
    static const float scales[] = { 1.0f, 1.0f / 1.5f, 1.0f / 1.7f, 0.5f, 1.0f / 3.0f };
    return scales[std::min<uint32_t>(quality, 4)];
}
} // namespace regpu

void regpu::SaveCaches() { RendererSaveCaches(); }

bool regpu::WindowHasFocus() { return PresenterHasFocus(); }
void regpu::LogState() { GpuLogState(); }
void regpu::LogCrashContext() { RendererLogCrashContext(); }
