#pragma once
// Hyper: ReGPU's low-spec, high-performance mode. Ported from the Forza Motorsport 2 port's hyper-prototype
// branch (src/gpu/hyper.h there, MIT, zer0ducksgiven), where it was built and measured on a Vega laptop and an
// Intel HD 630, and meant to move into ReGPU. It degrades how the GPU work is done (resolution, precision,
// sampling, refresh) and never removes an effect. Strictly generic: decided from what the renderer sees, with no
// game knowledge.
//
// REGPU_HYPER=off|mild|low|extreme|potato|auto picks a preset (auto: potato on an integrated GPU, else off). Every
// cut has its own REGPU_HYPER_* switch for A/B tests; the order is the environment, then the port's settings
// (Configure), then the preset. The cuts and their measured gains: docs in the FM3 port's HYPER.md.
#include <cstdint>
#include <string>

namespace regpu::hypermode
{
// What a port's settings say; empty or negative means "not set" (the preset decides).
struct Settings
{
    std::string preset;     // off, mild, low, extreme, potato, auto
    int scenePercent = -1;  // 25-100: the 3D scene's render scale
    int cheapUpscale = -1;  // 0 or 1: the upscale as one bilinear pass
    int noDiscard = -1;     // 0 or 1
    int helpers = -1;       // percent, 25-100
    int resolve1010 = -1;   // 0 or 1
    int aniso = -1;         // 0, 2, 4, 8 or 16
    int decimate = -1;      // percent of triangles kept (needs meshoptimizer, REGPU_MESHOPT_DIR)
    int cullPixels = -1;    // -1 off, else the pixel threshold
    int freeze = -1;        // frames before a 3D helper view freezes, 0 off
    int shaderCuts = -1;    // 0 or 1: the lossless shader cuts as one group
};

enum class Preset
{
    Off,
    Mild,
    Low,
    Extreme,
    Potato, // native resolution: the strongest kept native cuts (no scene scale, no upscale)
};

// Call before the renderer starts. Sets every cut's default for the preset (as REGPU_* environment defaults the
// renderer reads), so an explicit environment variable still wins.
void Configure(const Settings& settings);

Preset Level();
const char* Name(Preset preset);

// The scene's render scale (percent of the output per axis) Hyper asks for, 0 when it leaves it alone.
uint32_t SceneScalePercent();
// The final upscale as one bilinear pass instead of FSR 1.
bool CheapUpscale();
// The most anisotropic filtering the samplers may use (the player's level when lower); 16 = no cap.
uint32_t AnisotropyCap();
// Offscreen helper surfaces' size in percent per axis; 100 = full.
uint32_t HelperScalePercent();
// Draw culling's pixel threshold, -1 off.
int CullPixels();

// One line per active cut, for the log.
void LogSummary();
} // namespace regpu::hypermode
