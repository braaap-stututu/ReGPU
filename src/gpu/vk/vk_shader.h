#pragma once
// HLSL to SPIR-V for the Vulkan backend, through dxc: dxcompiler.dll in process on Windows when it is available,
// otherwise the dxc executable (REGPU_DXC, the bundled dxc.exe, or PATH). Results are cached on disk under a name
// that includes a hash of the final HLSL.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vk
{
// The translator's HLSL as the Vulkan pipeline layout binds it (gpu/vk/bindings.h): the bindless arrays sized, ReGPU's
// eight samplers (s0-s7) as one array binding (gSamplers[8]), and REGPU_LEAN_FETCH's sampler table (gsamp, s0-s7 in
// space1, which would collide with gtex in set 1) as the same array.
std::string AdaptHlsl(const std::string& hlsl);

// pixel selects ps_6_0/vs_6_0. cacheDir may be empty to skip the disk cache.
bool CompileHlsl(const std::string& hlsl, bool pixel, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error = nullptr, bool disableOptimizations = false);
// Any stage and entry point (profile "vs_6_0", "ps_6_0", "cs_6_0"...): ReGPU's own passes (blits, resolves, post
// filters, upscalers). Registers are shifted as for the translated shaders (b 0, t 100, s 200, u 300; set 0).
bool CompileHlslStage(const std::string& hlsl, const char* profile, const char* entry, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error = nullptr);
// Whether CompileHlsl would find this shader in the disk cache (no read, no compile).
bool SpirvCached(const std::string& hlsl, uint64_t key, const std::filesystem::path& cacheDir);

// Same, but for a shader that must always be compiled (the renderer's own blit passes).
inline uint64_t HashSource(const std::string& text)
{
    uint64_t h = 0xCBF29CE484222325ull;
    for (unsigned char c : text)
        h = (h ^ c) * 0x100000001B3ull;
    return h;
}
}
