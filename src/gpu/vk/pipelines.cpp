#include "gpu/vk/pipelines.h"

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

VkPipeline CreateGraphicsPipeline(Device& d, const PipelineDesc& p, const char* name)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    uint32_t stageCount = 0;
    stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, p.vs, p.vsEntry, nullptr };
    if (p.ps)
        stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, p.ps, p.psEntry, nullptr };

    // No vertex buffers: the translated shaders fetch their vertices from gmem / gphys (as on D3D12, no input layout).
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = p.topology;
    ia.primitiveRestartEnable = p.primitiveRestart;
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

    const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY };
    VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dyn.dynamicStateCount = uint32_t(sizeof(dynamic) / sizeof(dynamic[0]));
    dyn.pDynamicStates = dynamic;

    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = p.colorCount;
    rendering.pColorAttachmentFormats = p.colorFormats;
    const bool depth = p.depthFormat != VK_FORMAT_UNDEFINED;
    const bool stencil = p.depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || p.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
        p.depthFormat == VK_FORMAT_D16_UNORM_S8_UINT;
    rendering.depthAttachmentFormat = depth ? p.depthFormat : VK_FORMAT_UNDEFINED;
    rendering.stencilAttachmentFormat = stencil ? p.depthFormat : VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo gi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gi.pNext = &rendering;
    gi.stageCount = stageCount;
    gi.pStages = stages;
    gi.pVertexInputState = &vertexInput;
    gi.pInputAssemblyState = &ia;
    gi.pViewportState = &viewport;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pDepthStencilState = depth ? &ds : nullptr;
    gi.pColorBlendState = &cb;
    gi.pDynamicState = &dyn;
    gi.layout = p.layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateGraphicsPipelines(d.device, d.pipelineCache, 1, &gi, nullptr, &pipeline);
    if (result != VK_SUCCESS)
    {
        Log("vk: pipeline %s not created: %s", name ? name : "?", ResultName(result));
        return VK_NULL_HANDLE;
    }
    if (name)
        SetName(d, VK_OBJECT_TYPE_PIPELINE, uint64_t(pipeline), name);
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
