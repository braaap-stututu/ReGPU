// The Vulkan renderer's smoke test: opens the Vulkan device the renderer uses, then compiles
// the translator's shared HLSL preamble to SPIR-V the way the backend does (vk::AdaptHlsl, dxc), with a pixel shader
// that reads every kind of resource the translated shaders declare. Needs dxcompiler.dll beside the executable.
// Exit code 0: both worked.
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/hlsl.h"
#include "gpu/vk/vk_shader.h"
#include "gpu/vk/device.h"

int main()
{
    // The renderer's device (gpu/vk/device.h): the Vulkan 1.3 baseline, VMA and the queue timeline. FATAL ends the
    // process when the GPU lacks the baseline.
    vk::Device rd;
    vk::CreateDevice(rd, {});
    VkCommandPool pool;
    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = rd.queueFamily;
    vk::Check(vkCreateCommandPool(rd.device, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vk::Check(vkAllocateCommandBuffers(rd.device, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cb, &bi);
    vkEndCommandBuffer(cb);
    const uint64_t value = vk::Submit(rd, &cb, 1);
    const bool timeline = vk::WaitFor(rd, value, 5'000'000'000ull) && vk::Completed(rd) >= value;
    printf("vulkan: renderer device %s, timeline submit %s\n", rd.name.c_str(), timeline ? "ok" : "FAILED");
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
    return compiled && timeline ? 0 : 1;
}
