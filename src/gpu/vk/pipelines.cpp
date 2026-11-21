#include "gpu/vk/pipelines.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "log.h"

namespace vk
{
ShaderModule CreateShaderModule(Device& d, const std::vector<uint32_t>& spirv, const char* name)
{
    ShaderModule m;
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = spirv.size() * 4;
    ci.pCode = spirv.data();
    const VkResult result = vkCreateShaderModule(d.device, &ci, nullptr, &m.module);
    if (result != VK_SUCCESS)
        Log("vk: shader module %s not created: %s", name ? name : "?", ResultName(result));
    else if (name)
        SetName(d, VK_OBJECT_TYPE_SHADER_MODULE, uint64_t(m.module), name);
    return m;
}

void Destroy(Device& d, ShaderModule& m)
{
    if (m.module)
        vkDestroyShaderModule(d.device, m.module, nullptr);
    m = {};
}

// REGPU_SHADER_STATS (see below), for a complete or linked pipeline.
static void LogShaderStats(Device& d, VkPipeline pipeline, const char* name)
{
    // REGPU_SHADER_STATS=1 (AMD, VK_AMD_shader_info): each stage's registers, scratch and code size, to compare shader
    // translations (gpu/spirv against HLSL): "shader stats: <name> vs vgpr V sgpr S scratch B isa N | ps ...".
    if (d.shaderInfo && vkGetShaderInfoAMD)
    {
        std::string line;
        for (const VkShaderStageFlagBits stage : { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT })
        {
            VkShaderStatisticsInfoAMD st{};
            size_t size = sizeof(st);
            if (vkGetShaderInfoAMD(d.device, pipeline, stage, VK_SHADER_INFO_TYPE_STATISTICS_AMD, &size, &st) != VK_SUCCESS)
                continue;
            size_t isa = 0;
            vkGetShaderInfoAMD(d.device, pipeline, stage, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &isa, nullptr);
            char part[160];
            snprintf(part, sizeof(part), "%s%s vgpr %u sgpr %u scratch %zu isa %zu", line.empty() ? "" : " | ", stage == VK_SHADER_STAGE_VERTEX_BIT ? "vs" : "ps",
                st.resourceUsage.numUsedVgprs, st.resourceUsage.numUsedSgprs, st.resourceUsage.scratchMemUsageInBytes, isa);
            line += part;
        }
        Log("shader stats: %s %s", name ? name : "?", line.c_str());
        // REGPU_SHADER_ISA=<part of a pipeline name>: its pixel shader's disassembly to out/isa/<name>.txt.
        static const char* isaWant = getenv("REGPU_SHADER_ISA");
        if (isaWant && name && strstr(name, isaWant))
        {
            size_t size = 0;
            if (vkGetShaderInfoAMD(d.device, pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &size, nullptr) == VK_SUCCESS && size)
            {
                std::string text(size, '\0');
                vkGetShaderInfoAMD(d.device, pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &size, text.data());
                std::string file = std::string("out/isa_") + name + ".txt";
                for (char& ch : file)
                    if (ch == '/')
                        ch = '_';
                if (FILE* f = fopen(file.c_str(), "wb"))
                {
                    fwrite(text.data(), 1, strlen(text.c_str()), f);
                    fclose(f);
                }
            }
        }
    }
}

// A complete pipeline (parts 0) or the parts of a library (with flags VK_PIPELINE_CREATE_LIBRARY_BIT_KHR ...).
static VkPipeline BuildGraphics(Device& d, const PipelineDesc& p, const char* name, VkGraphicsPipelineLibraryFlagsEXT parts, VkPipelineCreateFlags flags)
{
    const bool all = parts == 0;
    const bool vi = all || (parts & VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT);
    const bool pre = all || (parts & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT);
    const bool frag = all || (parts & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT);
    const bool out = all || (parts & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT);
    VkPipelineShaderStageCreateInfo stages[2]{};
    uint32_t stageCount = 0;
    if (pre)
        stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, p.vs, p.vsEntry, nullptr };
    if (p.ps && frag)
        stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, p.ps, p.psEntry, nullptr };

    // No vertex buffers: the translated shaders fetch their vertices from gmem / gphys (as on D3D12, no input layout).
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = p.topology;
    // (Restart on a list-class pipeline needs VK_EXT_primitive_topology_list_restart; without it strips with cut indices
    // draw the cut as a vertex, which no title seen so far relies on.)
    ia.primitiveRestartEnable = p.primitiveRestart && (d.listRestart || (p.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
        p.topology != VK_PRIMITIVE_TOPOLOGY_LINE_LIST && p.topology != VK_PRIMITIVE_TOPOLOGY_POINT_LIST));
    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = p.wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = p.cull;
    rs.frontFace = p.frontFace;
    rs.depthClampEnable = !p.depthClip;
    rs.depthBiasEnable = p.depthBiasConstant != 0 || p.depthBiasSlope != 0;
    rs.depthBiasConstantFactor = p.depthBiasConstant;
    rs.depthBiasClamp = p.depthBiasClamp;
    rs.depthBiasSlopeFactor = p.depthBiasSlope;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = p.samples;
    ms.alphaToCoverageEnable = p.alphaToCoverage;

    auto face = [&](const StencilFace& f) {
        VkStencilOpState s{};
        s.failOp = f.fail;
        s.passOp = f.pass;
        s.depthFailOp = f.depthFail;
        s.compareOp = f.compare;
        s.compareMask = p.stencilReadMask;
        s.writeMask = p.stencilWriteMask;
        return s;
    };
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = p.depthTest;
    ds.depthWriteEnable = p.depthWrite;
    ds.depthCompareOp = p.depthTest ? p.depthCompare : VK_COMPARE_OP_ALWAYS;
    ds.stencilTestEnable = p.stencilTest;
    ds.front = face(p.front);
    ds.back = face(p.back);

    VkPipelineColorBlendAttachmentState blend[8]{};
    for (uint32_t i = 0; i < p.colorCount; i++)
    {
        const BlendTarget& t = p.blend[p.independentBlend ? i : 0];
        blend[i].blendEnable = t.enable;
        blend[i].srcColorBlendFactor = t.srcColor;
        blend[i].dstColorBlendFactor = t.dstColor;
        blend[i].colorBlendOp = t.colorOp;
        blend[i].srcAlphaBlendFactor = t.srcAlpha;
        blend[i].dstAlphaBlendFactor = t.dstAlpha;
        blend[i].alphaBlendOp = t.alphaOp;
        blend[i].colorWriteMask = t.writeMask;
    }
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = p.colorCount;
    cb.pAttachments = blend;

    // Each dynamic state in the part that owns it (a library lists only its own). A complete pipeline lists them in the
    // order it always has: the driver's pipeline cache keys on the create info as given, so a new order misses every
    // cached pipeline (FM3, 21 Nov 2026: ~3,700 rebuilt in play, 100+ ms each on an RX 7900 XTX).
    std::vector<VkDynamicState> dynamic;
    if (all)
    {
        dynamic = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY };
        if (p.dynamicDepthStencil)
            dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                                              VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                                              VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                              VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE });
        if (p.dynamicBlend)
            dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
                                              VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT });
    }
    if (!all && vi)
        dynamic.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
    if (!all && pre)
    {
        dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR });
        if (p.dynamicDepthStencil)
            dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS,
                                              VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE });
    }
    if (!all && frag)
    {
        dynamic.push_back(VK_DYNAMIC_STATE_STENCIL_REFERENCE);
        if (p.dynamicDepthStencil)
            dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                              VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                              VK_DYNAMIC_STATE_STENCIL_WRITE_MASK });
    }
    if (!all && out)
    {
        dynamic.push_back(VK_DYNAMIC_STATE_BLEND_CONSTANTS);
        if (p.dynamicBlend)
            dynamic.insert(dynamic.end(), { VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
                                              VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT });
    }
    VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dyn.dynamicStateCount = uint32_t(dynamic.size());
    dyn.pDynamicStates = dynamic.data();

    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = p.colorCount;
    rendering.pColorAttachmentFormats = p.colorFormats;
    const bool depth = p.depthFormat != VK_FORMAT_UNDEFINED;
    const bool stencil = p.depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || p.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
        p.depthFormat == VK_FORMAT_D16_UNORM_S8_UINT;
    rendering.depthAttachmentFormat = depth ? p.depthFormat : VK_FORMAT_UNDEFINED;
    rendering.stencilAttachmentFormat = stencil ? p.depthFormat : VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineLibraryCreateInfoEXT library{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT };
    library.flags = parts;
    if (!all)
        rendering.pNext = &library;
    VkGraphicsPipelineCreateInfo gi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gi.flags = flags;
    gi.pNext = &rendering;
    gi.stageCount = stageCount;
    gi.pStages = stageCount ? stages : nullptr;
    gi.pVertexInputState = vi ? &vertexInput : nullptr;
    gi.pInputAssemblyState = vi ? &ia : nullptr;
    gi.pViewportState = pre ? &viewport : nullptr;
    gi.pRasterizationState = pre ? &rs : nullptr;
    gi.pMultisampleState = frag || out ? &ms : nullptr;
    // (A fragment-shader library takes the depth/stencil state even without a depth target: VUID-...-09035.)
    gi.pDepthStencilState = frag && (depth || !all) ? &ds : nullptr;
    gi.pColorBlendState = out ? &cb : nullptr;
    gi.pDynamicState = dynamic.empty() ? nullptr : &dyn;
    gi.layout = pre || frag ? p.layout : VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateGraphicsPipelines(d.device, d.pipelineCache, 1, &gi, nullptr, &pipeline);
    if (result != VK_SUCCESS)
    {
        Log("vk: pipeline %s not created: %s", name ? name : "?", ResultName(result));
        return VK_NULL_HANDLE;
    }
    if (name)
        SetName(d, VK_OBJECT_TYPE_PIPELINE, uint64_t(pipeline), name);
    if (all)
        LogShaderStats(d, pipeline, name);
    return pipeline;
}

VkPipeline CreateGraphicsPipeline(Device& d, const PipelineDesc& p, const char* name)
{
    return BuildGraphics(d, p, name, 0, 0);
}

VkPipeline CreateGraphicsLibrary(Device& d, const PipelineDesc& p, VkGraphicsPipelineLibraryFlagsEXT parts, bool retain, const char* name)
{
    return BuildGraphics(d, p, name, parts,
        VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | (retain ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0));
}

VkPipeline LinkGraphicsPipeline(Device& d, const VkPipeline* libraries, uint32_t count, VkPipelineLayout layout, bool optimize, const char* name)
{
    VkPipelineLibraryCreateInfoKHR li{ VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR };
    li.libraryCount = count;
    li.pLibraries = libraries;
    VkGraphicsPipelineCreateInfo gi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gi.pNext = &li;
    gi.flags = optimize ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0;
    gi.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateGraphicsPipelines(d.device, d.pipelineCache, 1, &gi, nullptr, &pipeline);
    if (result != VK_SUCCESS)
    {
        Log("vk: pipeline %s not linked: %s", name ? name : "?", ResultName(result));
        return VK_NULL_HANDLE;
    }
    if (name)
        SetName(d, VK_OBJECT_TYPE_PIPELINE, uint64_t(pipeline), name);
    LogShaderStats(d, pipeline, name);
    return pipeline;
}

VkPipeline CreateComputePipeline(Device& d, VkPipelineLayout layout, VkShaderModule cs, const char* name)
{
    VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs, "main", nullptr };
    ci.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateComputePipelines(d.device, d.pipelineCache, 1, &ci, nullptr, &pipeline);
    if (result != VK_SUCCESS)
    {
        Log("vk: compute pipeline %s not created: %s", name ? name : "?", ResultName(result));
        return VK_NULL_HANDLE;
    }
    if (name)
        SetName(d, VK_OBJECT_TYPE_PIPELINE, uint64_t(pipeline), name);
    return pipeline;
}
}
