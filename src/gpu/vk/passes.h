#pragma once
// ReGPU's own GPU passes on Vulkan (docs/vulkan-conversion-plan.md, step 6): the resolve blits, the post filters, the
// upscalers' helpers, the front-buffer passes. On D3D12 each had a root signature of its own (a descriptor table of
// SRVs/UAVs and root constants) and fxc bytecode. Here a pass is HLSL compiled to SPIR-V by DXC with the translator's
// register shifts (b 0, t 100, s 200, u 300; set 0), and its descriptor set layout is read from the SPIR-V itself, so
// the HLSL stays as it was. A use of a pass writes a descriptor set from the frame slot's transient pools (reset when
// the slot comes round again) and binds it; constant buffers (b registers) come from the upload ring as uniform
// buffer ranges, where D3D12 had root constants.
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "gpu/vk/device.h"
#include "gpu/vk/pipelines.h"

namespace vk
{
// The descriptor bindings of set 0 that a SPIR-V module declares (its variables' Binding and DescriptorSet decorations,
// and their types). Other sets are ignored.
std::vector<VkDescriptorSetLayoutBinding> ReflectBindings(const std::vector<uint32_t>& spirv, VkShaderStageFlags stage);

struct Pass
{
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineBindPoint point = VK_PIPELINE_BIND_POINT_COMPUTE;
    std::vector<VkDescriptorSetLayoutBinding> bindings; // every stage's, merged
    explicit operator bool() const { return pipeline != VK_NULL_HANDLE; }
};

// A compute pass from SPIR-V (an embedded, pre-compiled shader) or HLSL (compiled now, cached in cacheDir).
Pass CreateComputePass(Device& d, const std::vector<uint32_t>& spirv, const char* name);
Pass CreateComputePass(Device& d, const std::string& hlsl, const char* entry, const std::filesystem::path& cacheDir, const char* name);

// A graphics pass (a full-screen triangle or a few quads): the vertex and pixel entries of one HLSL source, and the
// attachments, blend and rasteriser state of `state` (its layout and shader fields are filled in here). Viewport,
// scissor, stencil reference, blend constants and topology are dynamic, as for the draws.
Pass CreateGraphicsPass(Device& d, const std::string& hlsl, const char* vsEntry, const char* psEntry, PipelineDesc state,
    const std::filesystem::path& cacheDir, const char* name);
void Destroy(Device& d, Pass& pass);

// What one use of a pass binds, by binding number (the HLSL register plus its shift: t3 = 103, u0 = 300, b0 = 0).
struct PassArgs
{
    struct Entry
    {
        uint32_t binding;
        VkDescriptorImageInfo image{};
        VkDescriptorBufferInfo buffer{};
        bool isBuffer = false;
    };
    std::vector<Entry> entries;
    // A sampled image (t), in `layout` (SHADER_READ_ONLY_OPTIMAL unless given).
    PassArgs& Image(uint32_t binding, VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // A storage image (u), in GENERAL.
    PassArgs& Storage(uint32_t binding, VkImageView view);
    PassArgs& Sampler(uint32_t binding, VkSampler sampler);
    // A uniform or storage buffer range (b, t buffers, u buffers).
    PassArgs& Buffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
};

// Descriptor sets that last one frame: allocated from pools of a frame slot, reset when the slot is reused.
class TransientSets
{
public:
    // A set for `pass` with `args` written. Bindings the pass declares that args leaves out are null descriptors
    // (robustness2) or, for samplers, `fallbackSampler`.
    VkDescriptorSet Write(Device& d, const Pass& pass, const PassArgs& args, VkSampler fallbackSampler);
    void Reset(Device& d); // the slot's frame has completed
    void Destroy(Device& d);

private:
    VkDescriptorSet Allocate(Device& d, VkDescriptorSetLayout layout);
    std::mutex m_mutex;
    std::vector<VkDescriptorPool> m_pools;
    size_t m_current = 0;
};
}
