// gpu/vk/passes.h on the GPU: ReGPU's own passes as the renderer makes them. A compute pass from HLSL (a constant buffer,
// a sampled image, a storage image and a sampler left to the fallback) and a graphics pass (a full-screen triangle
// reading a sampled image into a colour attachment), their layouts read from the SPIR-V, their descriptor sets from
// vk::TransientSets, recorded by vk::RecordedList and read back. Exit code 0: both outputs are right.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "gpu/vk/passes.h"
#include "gpu/vk/resources.h"

namespace
{
const char kCompute[] = R"(
cbuffer C : register(b0) { float4 k; };
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState s : register(s0);
[numthreads(8, 8, 1)] void main(uint2 id : SV_DispatchThreadID)
{
    dst[id] = src.SampleLevel(s, (float2(id) + 0.5) / 8.0, 0) * k;
}
)";
const char kGraphics[] = R"(
Texture2D<float4> src : register(t0);
float4 vs(uint id : SV_VertexID) : SV_Position { return float4(float(id == 1) * 4.0 - 1.0, float(id == 2) * -4.0 + 1.0, 0, 1); }
float4 ps(float4 pos : SV_Position) : SV_Target { return src.Load(int3(pos.xy, 0)).bgra; }
)";

bool Near(float a, float b) { return std::fabs(a - b) < 0.01f; }

float Half(uint16_t h)
{
    const uint32_t e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? m / 1024.0f * 6.1035e-5f : (1.0f + m / 1024.0f) * std::ldexp(1.0f, int(e) - 15);
    return h & 0x8000 ? -v : v;
}
}

int main()
{
    vk::Device d;
    vk::CreateDevice(d, {});
    bool ok = true;

    vk::Pass compute = vk::CreateComputePass(d, kCompute, "main", {}, "test compute pass");
    vk::PipelineDesc pd;
    pd.colorFormats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    vk::Pass graphics = vk::CreateGraphicsPass(d, kGraphics, "vs", "ps", pd, {}, "test graphics pass");
    printf("vkpasses: compute pass %s (%zu bindings), graphics pass %s (%zu bindings)\n", compute ? "made" : "FAILED", compute.bindings.size(),
        graphics ? "made" : "FAILED", graphics.bindings.size());
    if (!compute || !graphics)
        return 1;
    // The compute pass declares b0, t0, u0 and s0 (bindings 0, 100, 300, 200).
    for (uint32_t b : { 0u, 100u, 200u, 300u })
    {
        bool found = false;
        for (const auto& x : compute.bindings)
            found = found || x.binding == b;
        if (!found)
        {
            printf("vkpasses: binding %u missing from the compute pass's layout\n", b);
            ok = false;
        }
    }

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    vk::Check(vkCreateSampler(d.device, &si, nullptr, &sampler), "vkCreateSampler");

    // The source: 8x8 RGBA8 of (64, 128, 192, 255).
    vk::ImageDesc sd;
    sd.width = sd.height = 8;
    sd.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sd.name = "source";
    vk::Image src = vk::CreateImage(d, sd);
    vk::ImageDesc cd;
    cd.width = cd.height = 8;
    cd.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    cd.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    cd.name = "compute output";
    vk::Image out = vk::CreateImage(d, cd);
    vk::ImageDesc gd;
    gd.width = gd.height = 8;
    gd.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    gd.name = "graphics output";
    vk::Image target = vk::CreateImage(d, gd);
    vk::Buffer upload = vk::CreateBuffer(d, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, vk::Memory::Upload, "upload");
    vk::Buffer readback = vk::CreateBuffer(d, 8 * 8 * 8 + 8 * 8 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::Memory::Readback, "readback");
    for (uint32_t i = 0; i < 64; i++)
    {
        const uint8_t texel[4] = { 64, 128, 192, 255 };
        memcpy(upload.mapped + i * 4, texel, 4);
    }
    const float k[4] = { 2.0f, 1.0f, 0.5f, 1.0f };
    memcpy(upload.mapped + 1024, k, sizeof(k));

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = d.queueFamily;
    vk::Check(vkCreateCommandPool(d.device, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vk::Check(vkAllocateCommandBuffers(d.device, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cb, &bi);
    vk::RecordedList list;
    list.Init(cb, true);

    vk::Transition(list, src, vk::Use::CopyDest);
    VkBufferImageCopy c{};
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    c.imageExtent = { 8, 8, 1 };
    list.CopyBufferToImage(upload.buffer, src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    vk::Barriers bars;
    bars.Add(src, vk::Use::Sampled);
    bars.Add(out, vk::Use::Storage);
    bars.Add(target, vk::Use::ColorTarget);
    bars.Record(list);

    vk::TransientSets sets;
    // Compute: (64, 128, 192, 255) / 255 * (2, 1, 0.5, 1).
    {
        vk::PassArgs args;
        args.Buffer(0, upload.buffer, 1024, 16).Image(100, src.view).Storage(300, out.view);
        const VkDescriptorSet set = sets.Write(d, compute, args, sampler);
        list.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
        list.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, compute.layout, 0, 1, &set);
        list.Dispatch(1, 1, 1);
    }
    // Graphics: the source swizzled bgra into the target.
    {
        vk::PassArgs args;
        args.Image(100, src.view);
        const VkDescriptorSet set = sets.Write(d, graphics, args, sampler);
        vk::Targets t{};
        t.colorCount = 1;
        t.color[0] = target.target;
        t.extent = { 8, 8 };
        list.SetTargets(t);
        list.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
        list.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.layout, 0, 1, &set);
        const VkViewport v{ 0, 0, 8, 8, 0, 1 };
        const VkRect2D r{ { 0, 0 }, { 8, 8 } };
        list.SetViewports(1, &v);
        list.SetScissors(1, &r);
        list.SetStencilReference(0);
        const float zero[4] = {};
        list.SetBlendConstants(zero);
        list.SetPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        list.Draw(3, 1, 0, 0);
    }
    vk::Barriers back;
    back.Add(out, vk::Use::CopySource);
    back.Add(target, vk::Use::CopySource);
    back.Record(list);
    VkBufferImageCopy rc{};
    rc.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    rc.imageExtent = { 8, 8, 1 };
    list.CopyImageToBuffer(out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &rc);
    rc.bufferOffset = 8 * 8 * 8;
    list.CopyImageToBuffer(target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &rc);
    vk::Barriers host;
    host.AddGlobal(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    host.Record(list);
    vkEndCommandBuffer(cb);
    vk::WaitFor(d, vk::Submit(d, &cb, 1));
    vmaInvalidateAllocation(d.allocator, readback.allocation, 0, VK_WHOLE_SIZE);

    const uint16_t* h = reinterpret_cast<const uint16_t*>(readback.mapped) + (3 * 8 + 5) * 4;
    const float cr = Half(h[0]), cg = Half(h[1]), cbv = Half(h[2]), ca = Half(h[3]);
    const bool computeOk = Near(cr, 2 * 64 / 255.0f) && Near(cg, 128 / 255.0f) && Near(cbv, 0.5f * 192 / 255.0f) && Near(ca, 1.0f);
    const uint8_t* p = readback.mapped + 8 * 8 * 8 + (6 * 8 + 2) * 4;
    const bool graphicsOk = p[0] == 192 && p[1] == 128 && p[2] == 64 && p[3] == 255;
    printf("vkpasses: compute (%.3f, %.3f, %.3f, %.3f) %s; graphics (%u, %u, %u, %u) %s\n", cr, cg, cbv, ca, computeOk ? "ok" : "WRONG", p[0], p[1], p[2], p[3],
        graphicsOk ? "ok" : "WRONG");
    ok = ok && computeOk && graphicsOk;

    sets.Destroy(d);
    vk::Destroy(d, compute);
    vk::Destroy(d, graphics);
    vk::Destroy(d, src);
    vk::Destroy(d, out);
    vk::Destroy(d, target);
    vk::Destroy(d, upload);
    vk::Destroy(d, readback);
    vkDestroySampler(d.device, sampler, nullptr);
    vkDestroyCommandPool(d.device, pool, nullptr);
    printf("vkpasses: validation messages %llu\n", (unsigned long long)vk::g_validationMessages.load());
    return ok && vk::g_validationMessages.load() == 0 ? 0 : 1;
}
