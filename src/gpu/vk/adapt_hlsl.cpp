// The translated shaders' HLSL as the Vulkan pipeline layout binds it (gpu/vk/bindings.h): see vk_shader.h.
#include "gpu/vk/vk_shader.h"

#include <cstring>
#include <string>
#include <utility>

namespace vk
{
// The translator's HLSL as the Vulkan pipeline layout binds it (see bindings.h).
std::string AdaptHlsl(const std::string& hlsl)
{
    std::string source = hlsl;
    for (const auto& [from, to] : { std::pair<const char*, const char*>{ "gtex[] :", "gtex[65536] :" }, { "gcube[] :", "gcube[65536] :" },
             { "gvol[] :", "gvol[65536] :" } })
        if (const size_t at = source.find(from); at != std::string::npos)
            source.replace(at, strlen(from), to);
    // ReGPU declares its eight samplers one by one (s0-s7); the Vulkan layout has them as one array binding.
    {
        static const char* const kSamplers[8] = { "sLinearWrap", "sLinearClamp", "sPointClamp", "sPointWrap", "sAniso2Wrap", "sAniso4Wrap",
            "sAniso8Wrap", "sAniso16Wrap" };
        const std::string first = std::string("SamplerState ") + kSamplers[0] + " : register(s0);";
        if (const size_t at = source.find(first); at != std::string::npos)
        {
            const std::string last = std::string("SamplerState ") + kSamplers[7] + " : register(s7);";
            const size_t end = source.find(last, at);
            if (end != std::string::npos)
            {
                std::string decl = "SamplerState gSamplers[8] : register(s0);\n";
                for (int i = 0; i < 8; i++)
                    decl += std::string("#define ") + kSamplers[i] + " gSamplers[" + std::to_string(i) + "]\n";
                source.replace(at, end + last.size() - at, decl);
            }
        }
    }
    {
        const char* const lean = "SamplerState gsamp[8] : register(s0, space1);";
        if (const size_t at = source.find(lean); at != std::string::npos)
            source.replace(at, strlen(lean), "#define gsamp gSamplers");
    }
    return source;
}
}
