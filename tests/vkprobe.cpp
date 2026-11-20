// The Vulkan backend's smoke test (REGPU_VULKAN builds): opens the Vulkan device the backend would use, then compiles
// the translator's shared HLSL preamble to SPIR-V the way the backend does (vk::AdaptHlsl, dxc), with a pixel shader
// that reads every kind of resource the translated shaders declare. Needs dxcompiler.dll beside the executable.
// Exit code 0: both worked.
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/hlsl.h"
#include "gpu/vk/vk_shader.h"
#include "gpu/vk/vk_shaders.h"

bool VulkanProbe();

int main()
{
    const bool device = VulkanProbe();
    const std::string source = vk::AdaptHlsl(std::string(HlslPreamble()) + R"(
float4 main(float4 pos : SV_Position) : SV_Target
{
    const uint i = asuint(tbias[0].y);
    float4 c = gtex[i].Sample(sLinearWrap, pos.xy) + gcube[i].Sample(sPointClamp, pos.xyz) + gvol[i].Sample(sAniso16Wrap, pos.xyz);
    return c * pc[0] + float4(asfloat(gmem.Load(0)), asfloat(gphys.Load(4)), float(pinfo.z), tinfo[1].x);
}
)");
    std::vector<uint32_t> spirv;
    std::string error;
    const bool compiled = vk::CompileHlsl(source, true, vk::HashSource(source), {}, spirv, &error);
    printf("vulkan: preamble to SPIR-V %s (%zu words)%s%s\n", compiled ? "ok" : "FAILED", spirv.size(), error.empty() ? "" : ": ", error.c_str());
    return device && compiled ? 0 : 1;
}
