// The draw path's pieces together on the GPU: shaders written against the translator's preamble (hlsl.cpp, through
// vk::AdaptHlsl and DXC to SPIR-V) in a pipeline built with gpu/vk/pipelines.h on the layout of gpu/vk/bindings.h, fed by
// set 0 (the constant blocks as dynamic uniform ranges of an upload ring, gmem, gphys, the samplers) and a bindless 2D
// texture in set 1, drawn by vk::RecordedList into an RGBA16F target and read back. Every term of the output comes from a
// different binding, so a wrong binding shows as a wrong channel. Exit code 0: the pixel is right.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include "gpu/constants.h"
#include "gpu/hlsl.h"
#include "gpu/vk/bindings.h"
#include "gpu/vk/pipelines.h"
#include "gpu/vk/resources.h"
#include "gpu/vk/vk_shader.h"

namespace
{
std::vector<uint32_t> Compile(const std::string& body, bool pixel)
{
    const std::string source = vk::AdaptHlsl(std::string(HlslPreamble()) + body);
    std::vector<uint32_t> spirv;
    std::string error;
    if (!vk::CompileHlsl(source, pixel, vk::HashSource(source), {}, spirv, &error))
        printf("vkdraw: %s shader: %s\n", pixel ? "pixel" : "vertex", error.c_str());
    return spirv;
}
}

int main()
{
    vk::Device d;
    vk::CreateDevice(d, {});
    vk::DrawBindings bindings;
    if (!vk::CreateDrawBindings(d, bindings, sizeof(gpu::VSConstants), sizeof(gpu::PSConstants), 256))
        return 1;
    VkSampler samplers[vk::kSamplerCount];
    const int aniso[vk::kSamplerCount] = { 16, 0, 0, 0, 2, 4, 8, 16 };
    vk::CreateSamplers(d, aniso, samplers);

    // The ring: VS constants at 0, PS constants at 64 KB, gmem data at 128 KB. Guest memory: a buffer standing in.
    const VkDeviceSize ringBytes = 1 << 20;
    vk::Buffer ring = vk::CreateBuffer(d, ringBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::Memory::UploadVram, "ring");
    vk::Buffer guest = vk::CreateBuffer(d, 1 << 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::Memory::UploadVram, "guest");
    memset(ring.mapped, 0, ringBytes);
    auto* ps = reinterpret_cast<gpu::PSConstants*>(ring.mapped + (64 << 10));
    const uint32_t textureIndex = 7;
    memcpy(&ps->tbias[2][1], &textureIndex, 4); // tbias[2].y: the bindless index (as the renderer writes it)
    ps->c[5][3] = 0.5f;                          // pc[5].w -> alpha
    const float gmemValue = 0.25f, gphysValue = 0.125f;
    memcpy(ring.mapped + (128 << 10) + 16, &gmemValue, 4);
    memcpy(guest.mapped + 32, &gphysValue, 4);
    VkDescriptorSet set0 = vk::CreateSet0(d, bindings, ring.buffer, ringBytes, guest.buffer, guest.size, samplers);

    // The texture: 4x4 RGBA8 of (0, 0, 0.75, 1), at bindless index 7.
    vk::ImageDesc td;
    td.width = td.height = 4;
    td.name = "texture";
    vk::Image texture = vk::CreateImage(d, td);
    vk::Buffer texels = vk::CreateBuffer(d, 64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::Memory::Upload, "texels");
    for (int i = 0; i < 16; i++)
        reinterpret_cast<uint32_t*>(texels.mapped)[i] = 0xFF000000u | (191u << 16); // A=255, B=191 (0.749)
    vk::WriteTexture(bindings, vk::TextureKind::Tex2D, textureIndex, texture.view);
    vk::FlushTextureWrites(d, bindings);

    // The target.
    vk::ImageDesc rd;
    rd.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    rd.width = rd.height = 16;
    rd.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    rd.name = "target";
    vk::Image target = vk::CreateImage(d, rd);
    vk::Buffer readback = vk::CreateBuffer(d, 16 * 16 * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::Memory::Readback, "readback");
    // (REGPU_VKDRAW_GPL=2: drawn into a 4x target with a D32S8 depth target, resolved into `target`.)
    const bool full = getenv("REGPU_VKDRAW_GPL") && atoi(getenv("REGPU_VKDRAW_GPL")) >= 2 && d.pipelineLibrary;
    vk::Image msTarget, depthTarget;
    if (full)
    {
        vk::ImageDesc md = rd;
        md.samples = VK_SAMPLE_COUNT_4_BIT;
        md.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        md.name = "ms target";
        msTarget = vk::CreateImage(d, md);
        vk::ImageDesc dd = md;
        dd.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        dd.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dd.name = "depth target";
        depthTarget = vk::CreateImage(d, dd);
    }

    // Shaders: a full-screen triangle, and a pixel shader of one term per binding.
    // REGPU_VKDRAW_VSOUT=1: the translated shaders' interface instead (VSOut, the pixel shader reading o[5] too). On the
    // RX 7900 XTX (driver 26.11.2, LLPC) that pixel shader writes zeros when it reads an interpolant and all four bindings
    // (constants, gmem, gphys, a bindless texture); any one fewer works, and the iGPU's compiler gets it right. Kept as
    // a reproducer: see docs/vulkan.md.
    const bool vsOut = getenv("REGPU_VKDRAW_VSOUT") && atoi(getenv("REGPU_VKDRAW_VSOUT")) != 0;
    const std::vector<uint32_t> vsCode = Compile(vsOut ? R"(
VSOut main(uint id : SV_VertexID)
{
    VSOut r;
    const float2 uv = float2((id << 1) & 2, id & 2);
    r.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    [unroll] for (int i = 0; i < 16; i++) r.o[i] = float4(i, 0, 0, 1);
    r.pcoord = uv;
    return r;
}
)" : R"(
float4 main(uint id : SV_VertexID) : SV_Position
{
    const float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
)", false);
    const std::vector<uint32_t> psCode = Compile(vsOut ? R"(
float4 main(VSOut input) : SV_Target
{
    const float4 t = gtex[asuint(tbias[2].y)].SampleLevel(sPointClamp, float2(0.5, 0.5), 0);
    return float4(asfloat(gmem.Load(131072 + 16)), asfloat(gphys.Load(32)), t.b, pc[5].w + (input.o[5].x - 5.0));
}
)" : R"(
float4 main(float4 pos : SV_Position) : SV_Target
{
    const float4 t = gtex[asuint(tbias[2].y)].SampleLevel(sPointClamp, float2(0.5, 0.5), 0);
    return float4(asfloat(gmem.Load(131072 + 16)), asfloat(gphys.Load(32)), t.b, pc[5].w);
}
)", true);
    // REGPU_VKDRAW_DUMP=1: the two shaders' SPIR-V to vkdraw_vs.spv / vkdraw_ps.spv (for spirv-dis).
    if (getenv("REGPU_VKDRAW_DUMP"))
        for (const auto& [file, code] : { std::pair{ "vkdraw_vs.spv", &vsCode }, std::pair{ "vkdraw_ps.spv", &psCode } })
            if (FILE* f = fopen(file, "wb"))
            {
                fwrite(code->data(), 4, code->size(), f);
                fclose(f);
            }
    if (vsCode.empty() || psCode.empty())
        return 1;
    vk::ShaderModule vs = vk::CreateShaderModule(d, vsCode, "vs"), psm = vk::CreateShaderModule(d, psCode, "ps");
    vk::PipelineDesc pd;
    pd.layout = bindings.layout;
    pd.vs = vs.module;
    pd.ps = psm.module;
    pd.colorFormats[0] = rd.format;
    // REGPU_VKDRAW_GPL=1: the pipeline from graphics pipeline library parts, fast-linked, depth/stencil state dynamic (as
    // the renderer's draws with REGPU_VK_PIPELINE_LIBRARY).
    const bool gpl = getenv("REGPU_VKDRAW_GPL") && d.pipelineLibrary;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (gpl)
    {
        pd.dynamicDepthStencil = true;
        // REGPU_VKDRAW_GPL=2: the renderer's draw state too (depth target D32S8, depth clamp, list restart, 4x MSAA).
        if (atoi(getenv("REGPU_VKDRAW_GPL")) >= 2)
        {
            pd.depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
            pd.depthClip = false;
            pd.primitiveRestart = true;
            pd.samples = VK_SAMPLE_COUNT_4_BIT;
        }
        VkPipeline parts[4] = {};
        const VkGraphicsPipelineLibraryFlagsEXT kinds[4] = { VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
            VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
            VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT };
        for (int i = 0; i < 4; i++)
            if (!(parts[i] = vk::CreateGraphicsLibrary(d, pd, kinds[i], false, "vkdraw part")))
                return 1;
        pipeline = vk::LinkGraphicsPipeline(d, parts, 4, pd.layout, false, "vkdraw linked");
        printf("vkdraw: pipeline from libraries (fast link)\n");
    }
    else
        pipeline = vk::CreateGraphicsPipeline(d, pd, "vkdraw");
    if (!pipeline)
        return 1;

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = d.queueFamily;
    VkCommandPool pool;
    vk::Check(vkCreateCommandPool(d.device, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vk::Check(vkAllocateCommandBuffers(d.device, &cai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cb, &bi);
    static vk::RecordedList list;
    list.Init(cb);

    vk::Barriers barriers;
    barriers.Add(texture, vk::Use::CopyDest);
    barriers.Add(target, full ? vk::Use::CopyDest : vk::Use::ColorTarget);
    if (full)
    {
        barriers.Add(msTarget, vk::Use::ColorTarget);
        barriers.Add(depthTarget, vk::Use::DepthTarget);
    }
    barriers.Record(list);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { 4, 4, 1 };
    list.CopyBufferToImage(texels.buffer, texture.image, texture.state.layout, 1, &region);
    // Host writes to the ring and guest memory before the shaders read them (submission covers them; a barrier
    // documents it for the validation layer's sake).
    vk::Transition(list, texture, vk::Use::Sampled);

    vk::Targets t{};
    t.colorCount = 1;
    t.color[0] = full ? msTarget.target : target.target;
    t.extent = target.Extent();
    if (full)
    {
        t.depth = depthTarget.target;
        t.depthHasStencil = true;
    }
    list.SetTargets(t);
    const float clear[4] = { 1, 1, 1, 1 };
    list.ClearColor(t.color[0], target.Extent(), clear, 0, nullptr);
    list.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // Sets 1-3 once, set 0 per draw with the draw's constant offsets.
    list.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, bindings.layout, 1, 3, bindings.textureSets);
    const uint32_t offsets[3] = { 0, 64 << 10, 0 };
    list.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, bindings.layout, 0, 1, &set0, 3, offsets);
    const VkViewport vp{ 0, 0, 16, 16, 0, 1 };
    const VkRect2D sc{ { 0, 0 }, { 16, 16 } };
    list.SetViewports(1, &vp);
    list.SetScissors(1, &sc);
    list.SetPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    if (gpl)
    {
        vk::DrawState st{};
        st.depthTest = full; // (depth 0 against the clear's: ALWAYS)
        st.depthWrite = full;
        list.SetDrawState(st);
    }
    list.Draw(3, 1, 0, 0);
    if (full)
    {
        vk::Transition(list, msTarget, vk::Use::CopySource);
        VkImageResolve r{};
        r.srcSubresource = r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.extent = { 16, 16, 1 };
        list.ResolveImage(msTarget.image, msTarget.state.layout, target.image, target.state.layout, r);
    }
    vk::Transition(list, target, vk::Use::CopySource);
    VkBufferImageCopy back{};
    back.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    back.imageExtent = { 16, 16, 1 };
    list.CopyImageToBuffer(target.image, target.state.layout, readback.buffer, 1, &back);
    barriers.AddGlobal(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    barriers.Record(list);

    VkCommandBuffer done = list.Real();
    vkEndCommandBuffer(done);
    const uint64_t value = vk::Submit(d, &done, 1);
    bool ok = vk::WaitFor(d, value, 5'000'000'000ull);
    if (!ok)
        printf("vkdraw: the GPU did not finish in 5 s\n");
    vmaInvalidateAllocation(d.allocator, readback.allocation, 0, VK_WHOLE_SIZE);

    // RGBA16F at (8, 8).
    auto half = [](uint16_t h) {
        const uint32_t e = (h >> 10) & 31, m = h & 1023;
        const float v = e ? std::ldexp(1.0f + m / 1024.0f, int(e) - 15) : std::ldexp(m / 1024.0f, -14);
        return h & 0x8000 ? -v : v;
    };
    const uint16_t* px = reinterpret_cast<const uint16_t*>(readback.mapped) + (8 * 16 + 8) * 4;
    const float r = half(px[0]), g = half(px[1]), b = half(px[2]), a = half(px[3]);
    auto close = [](float x, float want) { return std::fabs(x - want) < 0.01f; };
    const bool right = close(r, gmemValue) && close(g, gphysValue) && close(b, 0.749f) && close(a, 0.5f);
    printf("vkdraw: pixel (%.3f, %.3f, %.3f, %.3f): gmem %s, gphys %s, bindless texture %s, PS constants %s\n", r, g, b, a,
        close(r, gmemValue) ? "ok" : "WRONG", close(g, gphysValue) ? "ok" : "WRONG", close(b, 0.749f) ? "ok" : "WRONG", close(a, 0.5f) ? "ok" : "WRONG");
    ok &= right;
    if (d.validation)
    {
        printf("vkdraw: validation messages %llu\n", (unsigned long long)vk::g_validationMessages.load());
        ok &= vk::g_validationMessages.load() == 0;
    }
    return ok ? 0 : 1;
}
