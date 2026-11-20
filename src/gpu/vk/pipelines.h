#pragma once
// Graphics and compute pipelines (docs/vulkan-conversion-plan.md, step 5). PipelineDesc holds what renderer.cpp's
// D3D12_GRAPHICS_PIPELINE_STATE_DESCs set (EnsurePipeline, the blits, the post passes), in Vulkan terms, so each D3D12
// desc converts field by field. Viewport, scissor, stencil reference, blend constants and the primitive topology are
// dynamic (RecordedList sets them, as the D3D12 list did); the topology class is fixed per pipeline.
//
// Pipelines are built with dynamic rendering (no render pass objects): the attachment formats and sample count are part
// of the desc. Every build goes through the device's VkPipelineCache (kept on disk by vk::SavePipelineCache), which
// replaces the D3D12 pipeline library.
#include <cstdint>
#include <vector>

#include "gpu/vk/device.h"

namespace vk
{
struct ShaderModule
{
    VkShaderModule module = VK_NULL_HANDLE;
    explicit operator bool() const { return module != VK_NULL_HANDLE; }
};
ShaderModule CreateShaderModule(Device& d, const std::vector<uint32_t>& spirv, const char* name = nullptr);
void Destroy(Device& d, ShaderModule& m);

struct BlendTarget
{
    bool enable = false;
    VkBlendFactor srcColor = VK_BLEND_FACTOR_ONE, dstColor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp colorOp = VK_BLEND_OP_ADD;
    VkBlendFactor srcAlpha = VK_BLEND_FACTOR_ONE, dstAlpha = VK_BLEND_FACTOR_ZERO;
    VkBlendOp alphaOp = VK_BLEND_OP_ADD;
    VkColorComponentFlags writeMask = 0xF;
};

struct StencilFace
{
    VkStencilOp fail = VK_STENCIL_OP_KEEP, depthFail = VK_STENCIL_OP_KEEP, pass = VK_STENCIL_OP_KEEP;
    VkCompareOp compare = VK_COMPARE_OP_ALWAYS;
};

struct PipelineDesc
{
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE; // ps may be null (depth-only)
    const char* vsEntry = "main";
    const char* psEntry = "main";

    // Attachments.
    uint32_t colorCount = 1;
    VkFormat colorFormats[8] = { VK_FORMAT_R8G8B8A8_UNORM };
    VkFormat depthFormat = VK_FORMAT_UNDEFINED; // with its stencil when the format has one
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    // Blend (one per colour attachment; D3D12's IndependentBlendEnable false = target 0's for all).
    BlendTarget blend[8]{};
    bool independentBlend = false;
    bool alphaToCoverage = false;

    // Rasteriser.
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE; // D3D12's FrontCounterClockwise false
    bool depthClip = true;                           // false: depth clamp (D3D12's DepthClipEnable false)
    bool wireframe = false;
    // Depth bias (D3D12's DepthBias, DepthBiasClamp, SlopeScaledDepthBias; the same units on float depth).
    float depthBiasConstant = 0, depthBiasClamp = 0, depthBiasSlope = 0;

    // Depth and stencil.
    bool depthTest = false, depthWrite = false;
    VkCompareOp depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    bool stencilTest = false;
    uint8_t stencilReadMask = 0xFF, stencilWriteMask = 0xFF;
    StencilFace front{}, back{};

    // Input assembly: the class of primitive (the exact topology is dynamic) and the strip cut (D3D12's
    // IBStripCutValue: 0xFFFF / 0xFFFFFFFF ends a strip).
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool primitiveRestart = false;
};

// Builds the pipeline (through the device's pipeline cache). VK_NULL_HANDLE and a log line on failure.
VkPipeline CreateGraphicsPipeline(Device& d, const PipelineDesc& desc, const char* name = nullptr);
VkPipeline CreateComputePipeline(Device& d, VkPipelineLayout layout, VkShaderModule cs, const char* name = nullptr);
}
