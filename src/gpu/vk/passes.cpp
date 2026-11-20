#include "gpu/vk/passes.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <unordered_map>

#include "gpu/vk/vk_shader.h"
#include "log.h"

namespace vk
{
namespace
{
// SPIR-V opcodes and enumerants read here.
enum : uint32_t
{
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpConstant = 43,
    OpVariable = 59,
    OpDecorate = 71,
    DecorationBlock = 2,
    DecorationBufferBlock = 3,
    DecorationBinding = 33,
    DecorationDescriptorSet = 34,
    StorageUniformConstant = 0,
    StorageUniform = 2,
    StorageStorageBuffer = 12,
    DimBuffer = 5,
};
}

std::vector<VkDescriptorSetLayoutBinding> ReflectBindings(const std::vector<uint32_t>& spirv, VkShaderStageFlags stage)
{
    std::vector<VkDescriptorSetLayoutBinding> out;
    if (spirv.size() < 5)
        return out;
    struct Type
    {
        uint32_t op = 0;
        uint32_t a = 0, b = 0, c = 0; // image: dim, sampled; pointer: storage class, pointee; array: element, length id
    };
    std::unordered_map<uint32_t, Type> types;
    std::unordered_map<uint32_t, uint32_t> constants, binding, set;
    std::unordered_map<uint32_t, uint32_t> blockKind; // struct id -> DecorationBlock / DecorationBufferBlock
    struct Var
    {
        uint32_t id, pointer, storage;
    };
    std::vector<Var> vars;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word = spirv[i], op = word & 0xFFFF, count = word >> 16;
        if (!count || i + count > spirv.size())
            break;
        const uint32_t* w = &spirv[i];
        switch (op)
        {
        case OpTypeImage: types[w[1]] = { op, w[3], w[7] }; break;
        case OpTypeSampler:
        case OpTypeSampledImage:
        case OpTypeStruct: types[w[1]] = { op }; break;
        case OpTypeArray: types[w[1]] = { op, w[2], w[3] }; break;
        case OpTypeRuntimeArray: types[w[1]] = { op, w[2], 0 }; break;
        case OpTypePointer: types[w[1]] = { op, w[2], w[3] }; break;
        case OpConstant:
            if (count >= 4)
                constants[w[2]] = w[3];
            break;
        case OpVariable: vars.push_back({ w[2], w[1], w[3] }); break;
        case OpDecorate:
            if (count >= 4 && w[2] == DecorationBinding)
                binding[w[1]] = w[3];
            else if (count >= 4 && w[2] == DecorationDescriptorSet)
                set[w[1]] = w[3];
            else if (w[2] == DecorationBlock || w[2] == DecorationBufferBlock)
                blockKind[w[1]] = w[2];
            break;
        }
        i += count;
    }
    for (const Var& v : vars)
    {
        if (!binding.count(v.id) || (set.count(v.id) && set[v.id] != 0))
            continue;
        if (v.storage != StorageUniformConstant && v.storage != StorageUniform && v.storage != StorageStorageBuffer)
            continue;
        uint32_t t = types[v.pointer].b, arrayCount = 1;
        while (types[t].op == OpTypeArray || types[t].op == OpTypeRuntimeArray)
        {
            arrayCount *= types[t].op == OpTypeArray ? std::max(constants[types[t].b], 1u) : 1u;
            t = types[t].a;
        }
        const Type& ty = types[t];
        VkDescriptorType kind;
        if (v.storage == StorageStorageBuffer)
            kind = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else if (v.storage == StorageUniform)
            kind = blockKind[t] == DecorationBufferBlock ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (ty.op == OpTypeSampler)
            kind = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (ty.op == OpTypeSampledImage)
            kind = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (ty.op == OpTypeImage)
            kind = ty.a == DimBuffer ? (ty.b == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                                     : (ty.b == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        else
            continue;
        const uint32_t b = binding[v.id];
        auto same = std::find_if(out.begin(), out.end(), [&](const VkDescriptorSetLayoutBinding& x) { return x.binding == b; });
        if (same != out.end())
        {
            // Two variables at one register (the blits declare Texture2D<float> and Texture2D<float4> at t0): one binding.
            same->descriptorCount = std::max(same->descriptorCount, arrayCount);
            continue;
        }
        out.push_back({ b, kind, arrayCount, stage, nullptr });
    }
    return out;
}

namespace
{
// The set layout and pipeline layout of a pass from its stages' bindings.
bool MakeLayouts(Device& d, Pass& p, const char* name)
{
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = uint32_t(p.bindings.size());
    li.pBindings = p.bindings.data();
    if (vkCreateDescriptorSetLayout(d.device, &li, nullptr, &p.setLayout) != VK_SUCCESS)
        return false;
    VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.setLayout;
    if (vkCreatePipelineLayout(d.device, &pl, nullptr, &p.layout) != VK_SUCCESS)
        return false;
    SetName(d, VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(p.layout), name);
    return true;
}

void Merge(std::vector<VkDescriptorSetLayoutBinding>& into, const std::vector<VkDescriptorSetLayoutBinding>& more)
{
    for (const VkDescriptorSetLayoutBinding& b : more)
    {
        auto it = std::find_if(into.begin(), into.end(), [&](const VkDescriptorSetLayoutBinding& x) { return x.binding == b.binding; });
        if (it == into.end())
            into.push_back(b);
        else
        {
            it->stageFlags |= b.stageFlags;
            it->descriptorCount = std::max(it->descriptorCount, b.descriptorCount);
        }
    }
}
}

Pass CreateComputePass(Device& d, const std::vector<uint32_t>& spirv, const char* name)
{
    Pass p;
    p.point = VK_PIPELINE_BIND_POINT_COMPUTE;
    p.bindings = ReflectBindings(spirv, VK_SHADER_STAGE_COMPUTE_BIT);
    ShaderModule cs = CreateShaderModule(d, spirv, name);
    if (!cs || !MakeLayouts(d, p, name))
    {
        Destroy(d, cs);
        Destroy(d, p);
        return {};
    }
    p.pipeline = CreateComputePipeline(d, p.layout, cs.module, name);
    Destroy(d, cs);
    if (!p.pipeline)
        Destroy(d, p);
    return p;
}

Pass CreateComputePass(Device& d, const std::string& hlsl, const char* entry, const std::filesystem::path& cacheDir, const char* name)
{
    std::vector<uint32_t> spirv;
    std::string error;
    if (!CompileHlslStage(hlsl, "cs_6_0", entry, cacheDir, spirv, &error))
    {
        Log("vk: pass %s (%s) did not compile: %s", name, entry, error.c_str());
        return {};
    }
    return CreateComputePass(d, spirv, name);
}

Pass CreateGraphicsPass(Device& d, const std::string& hlsl, const char* vsEntry, const char* psEntry, PipelineDesc state,
    const std::filesystem::path& cacheDir, const char* name)
{
    std::vector<uint32_t> vsCode, psCode;
    std::string error;
    if (!CompileHlslStage(hlsl, "vs_6_0", vsEntry, cacheDir, vsCode, &error) ||
        (psEntry && !CompileHlslStage(hlsl, "ps_6_0", psEntry, cacheDir, psCode, &error)))
    {
        Log("vk: pass %s (%s, %s) did not compile: %s", name, vsEntry, psEntry ? psEntry : "-", error.c_str());
        return {};
    }
    Pass p;
    p.point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    p.bindings = ReflectBindings(vsCode, VK_SHADER_STAGE_VERTEX_BIT);
    if (psEntry)
        Merge(p.bindings, ReflectBindings(psCode, VK_SHADER_STAGE_FRAGMENT_BIT));
    ShaderModule vs = CreateShaderModule(d, vsCode, name), ps = psEntry ? CreateShaderModule(d, psCode, name) : ShaderModule{};
    if (!vs || (psEntry && !ps) || !MakeLayouts(d, p, name))
    {
        Destroy(d, vs);
        Destroy(d, ps);
        Destroy(d, p);
        return {};
    }
    state.layout = p.layout;
    state.vs = vs.module;
    state.ps = ps.module;
    state.vsEntry = "main";
    state.psEntry = "main";
    p.pipeline = CreateGraphicsPipeline(d, state, name);
    Destroy(d, vs);
    Destroy(d, ps);
    if (!p.pipeline)
        Destroy(d, p);
    return p;
}

void Destroy(Device& d, Pass& p)
{
    if (p.pipeline)
        vkDestroyPipeline(d.device, p.pipeline, nullptr);
    if (p.layout)
        vkDestroyPipelineLayout(d.device, p.layout, nullptr);
    if (p.setLayout)
        vkDestroyDescriptorSetLayout(d.device, p.setLayout, nullptr);
    p = {};
}

PassArgs& PassArgs::Image(uint32_t binding, VkImageView view, VkImageLayout layout)
{
    Entry e{ binding };
    e.image = { VK_NULL_HANDLE, view, layout };
    entries.push_back(e);
    return *this;
}
PassArgs& PassArgs::Storage(uint32_t binding, VkImageView view)
{
    return Image(binding, view, VK_IMAGE_LAYOUT_GENERAL);
}
PassArgs& PassArgs::Sampler(uint32_t binding, VkSampler sampler)
{
    Entry e{ binding };
    e.image = { sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    entries.push_back(e);
    return *this;
}
PassArgs& PassArgs::Buffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
{
    Entry e{ binding };
    e.buffer = { buffer, offset, range };
    e.isBuffer = true;
    entries.push_back(e);
    return *this;
}

VkDescriptorSet TransientSets::Allocate(Device& d, VkDescriptorSetLayout layout)
{
    for (;;)
    {
        if (m_current == m_pools.size())
        {
            const VkDescriptorPoolSize sizes[] = {
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2048 },
                { VK_DESCRIPTOR_TYPE_SAMPLER, 1024 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256 },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256 },
            };
            VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pi.maxSets = 2048;
            pi.poolSizeCount = uint32_t(std::size(sizes));
            pi.pPoolSizes = sizes;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            Check(vkCreateDescriptorPool(d.device, &pi, nullptr, &pool), "vkCreateDescriptorPool(passes)");
            m_pools.push_back(pool);
        }
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = m_pools[m_current];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(d.device, &ai, &set);
        if (r == VK_SUCCESS)
            return set;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)
            Check(r, "vkAllocateDescriptorSets(passes)");
        m_current++; // full: the next pool (made above when there is none)
    }
}

VkDescriptorSet TransientSets::Write(Device& d, const Pass& pass, const PassArgs& args, VkSampler fallbackSampler)
{
    std::lock_guard lock(m_mutex);
    const VkDescriptorSet set = Allocate(d, pass.setLayout);
    std::vector<VkWriteDescriptorSet> writes;
    std::deque<VkDescriptorImageInfo> images; // (stable addresses for the writes)
    std::deque<VkDescriptorBufferInfo> buffers;
    for (const VkDescriptorSetLayoutBinding& b : pass.bindings)
        for (uint32_t element = 0; element < b.descriptorCount; element++)
        {
            const PassArgs::Entry* given = nullptr;
            for (const PassArgs::Entry& e : args.entries)
                if (e.binding == b.binding && element == 0) // (arrays: their first element only; the rest null)
                {
                    given = &e;
                    break;
                }
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet = set;
            w.dstBinding = b.binding;
            w.dstArrayElement = element;
            w.descriptorCount = 1;
            w.descriptorType = b.descriptorType;
            switch (b.descriptorType)
            {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                buffers.push_back(given && given->isBuffer ? given->buffer : VkDescriptorBufferInfo{ VK_NULL_HANDLE, 0, VK_WHOLE_SIZE });
                w.pBufferInfo = &buffers.back();
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                continue; // (no pass uses typed buffers yet: left unwritten)
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                images.push_back({ given && given->image.sampler ? given->image.sampler : fallbackSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
                w.pImageInfo = &images.back();
                break;
            default:
            {
                VkDescriptorImageInfo info = given && !given->isBuffer ? given->image : VkDescriptorImageInfo{};
                if (b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                else if (info.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                if (b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && !info.sampler)
                    info.sampler = fallbackSampler;
                images.push_back(info);
                w.pImageInfo = &images.back();
                break;
            }
            }
            writes.push_back(w);
        }
    if (!writes.empty())
        vkUpdateDescriptorSets(d.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    return set;
}

void TransientSets::Reset(Device& d)
{
    std::lock_guard lock(m_mutex);
    for (VkDescriptorPool pool : m_pools)
        vkResetDescriptorPool(d.device, pool, 0);
    m_current = 0;
}

void TransientSets::Destroy(Device& d)
{
    std::lock_guard lock(m_mutex);
    for (VkDescriptorPool pool : m_pools)
        vkDestroyDescriptorPool(d.device, pool, nullptr);
    m_pools.clear();
    m_current = 0;
}
}
