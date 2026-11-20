#include "gpu/vk/bindings.h"

#include <algorithm>

#include "log.h"

namespace vk
{
namespace
{
constexpr uint32_t kSet0Bindings[] = { 0, 1, 2, 100, 101, 200 };
constexpr uint32_t kMaxSet0 = 256; // frame slots x sampler banks x 2 (b2 over this or the last slot's ring) x ring regrowths
}

bool CreateDrawBindings(Device& d, DrawBindings& b, uint32_t vsBytes, uint32_t psBytes, uint32_t motionBytes)
{
    b.constantBytes[0] = vsBytes;
    b.constantBytes[1] = psBytes;
    b.constantBytes[2] = motionBytes;

    VkPhysicalDeviceDescriptorIndexingProperties indexing{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES };
    VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    p2.pNext = &indexing;
    vkGetPhysicalDeviceProperties2(d.physical, &p2);
    if (indexing.maxDescriptorSetUpdateAfterBindSampledImages < 3 * kBindlessCount ||
        indexing.maxPerStageDescriptorUpdateAfterBindSampledImages < 3 * kBindlessCount)
    {
        Log("vk: the GPU holds %u update-after-bind images per stage, ReGPU needs %u", indexing.maxPerStageDescriptorUpdateAfterBindSampledImages,
            3 * kBindlessCount);
        return false;
    }

    // Set 0.
    const VkShaderStageFlags all = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding set0[6]{};
    for (int i = 0; i < 3; i++)
        set0[i] = { uint32_t(i), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, all, nullptr };
    set0[3] = { 100, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, nullptr };
    set0[4] = { 101, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, nullptr };
    set0[5] = { 200, VK_DESCRIPTOR_TYPE_SAMPLER, kSamplerCount, all, nullptr };
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 6;
    lci.pBindings = set0;
    Check(vkCreateDescriptorSetLayout(d.device, &lci, nullptr, &b.set0), "vkCreateDescriptorSetLayout (set 0)");

    // Sets 1-3: one bindless array each.
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        (d.updateUnusedWhilePending ? VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT : 0);
    for (int i = 0; i < 3; i++)
    {
        const VkDescriptorSetLayoutBinding binding{ 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kBindlessCount, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr };
        VkDescriptorSetLayoutBindingFlagsCreateInfo fci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        fci.bindingCount = 1;
        fci.pBindingFlags = &flags;
        VkDescriptorSetLayoutCreateInfo tci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        tci.pNext = &fci;
        tci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        tci.bindingCount = 1;
        tci.pBindings = &binding;
        Check(vkCreateDescriptorSetLayout(d.device, &tci, nullptr, &b.textures[i]), "vkCreateDescriptorSetLayout (bindless)");
    }

    const VkDescriptorSetLayout sets[4] = { b.set0, b.textures[0], b.textures[1], b.textures[2] };
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 4;
    pli.pSetLayouts = sets;
    Check(vkCreatePipelineLayout(d.device, &pli, nullptr, &b.layout), "vkCreatePipelineLayout (draws)");

    // Pools.
    const VkDescriptorPoolSize set0Sizes[] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 3 * kMaxSet0 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMaxSet0 }, { VK_DESCRIPTOR_TYPE_SAMPLER, kSamplerCount * kMaxSet0 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = kMaxSet0;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = set0Sizes;
    Check(vkCreateDescriptorPool(d.device, &pci, nullptr, &b.pool), "vkCreateDescriptorPool (set 0)");
    const VkDescriptorPoolSize textureSizes{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3 * kBindlessCount };
    VkDescriptorPoolCreateInfo tpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    tpci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    tpci.maxSets = 3;
    tpci.poolSizeCount = 1;
    tpci.pPoolSizes = &textureSizes;
    Check(vkCreateDescriptorPool(d.device, &tpci, nullptr, &b.texturePool), "vkCreateDescriptorPool (bindless)");
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = b.texturePool;
    ai.descriptorSetCount = 3;
    ai.pSetLayouts = b.textures;
    Check(vkAllocateDescriptorSets(d.device, &ai, b.textureSets), "vkAllocateDescriptorSets (bindless)");
    SetName(d, VK_OBJECT_TYPE_DESCRIPTOR_SET, uint64_t(b.textureSets[0]), "gtex");
    SetName(d, VK_OBJECT_TYPE_DESCRIPTOR_SET, uint64_t(b.textureSets[1]), "gcube");
    SetName(d, VK_OBJECT_TYPE_DESCRIPTOR_SET, uint64_t(b.textureSets[2]), "gvol");
    // Indices 0-2 of every array are the null textures (renderer.cpp's 0 null 2D, 1 null cube, 2 null 3D; a full table
    // draws with index = kind): reads give 0, as D3D12's null views did. Other unwritten entries are never read.
    for (uint32_t kind = 0; kind < 3; kind++)
        for (uint32_t index = 0; index < 3; index++)
            WriteTexture(b, TextureKind(kind), index, VK_NULL_HANDLE);
    FlushTextureWrites(d, b);
    return true;
}

VkDescriptorSet CreateSet0(Device& d, DrawBindings& b, VkBuffer ring, VkDeviceSize ringBytes, VkBuffer guest, VkDeviceSize guestBytes,
    const VkSampler (&samplers)[kSamplerCount], VkBuffer motionRing, VkDeviceSize motionRingBytes)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = b.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &b.set0;
    Check(vkAllocateDescriptorSets(d.device, &ai, &set), "vkAllocateDescriptorSets (set 0)");
    VkDescriptorBufferInfo constants[3];
    for (int i = 0; i < 3; i++)
    {
        const bool other = i == 2 && motionRing; // (b2 over another ring: last frame's)
        constants[i] = { other ? motionRing : ring, 0,
            std::min<VkDeviceSize>(b.constantBytes[i] ? b.constantBytes[i] : 16, other ? motionRingBytes : ringBytes) };
    }
    const VkDescriptorBufferInfo ringInfo{ ring, 0, ringBytes };
    const VkDescriptorBufferInfo guestInfo{ guest, 0, guestBytes };
    VkDescriptorImageInfo samplerInfo[kSamplerCount];
    for (uint32_t i = 0; i < kSamplerCount; i++)
        samplerInfo[i] = { samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++)
    {
        w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[i].dstSet = set;
        w[i].dstBinding = kSet0Bindings[i];
        w[i].descriptorCount = 1;
    }
    for (int i = 0; i < 3; i++)
    {
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        w[i].pBufferInfo = &constants[i];
    }
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo = &ringInfo;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[4].pBufferInfo = &guestInfo;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[5].descriptorCount = kSamplerCount;
    w[5].pImageInfo = samplerInfo;
    vkUpdateDescriptorSets(d.device, 6, w, 0, nullptr);
    return set;
}

void FreeSet0(Device& d, DrawBindings& b, VkDescriptorSet set)
{
    if (set)
        vkFreeDescriptorSets(d.device, b.pool, 1, &set);
}

void WriteTexture(DrawBindings& b, TextureKind kind, uint32_t index, VkImageView view)
{
    std::lock_guard lock(b.writeMutex);
    b.pending.push_back({ kind, index, view });
}

void FlushTextureWrites(Device& d, DrawBindings& b)
{
    std::vector<DrawBindings::PendingWrite> writes;
    {
        std::lock_guard lock(b.writeMutex);
        writes.swap(b.pending);
    }
    if (writes.empty())
        return;
    std::vector<VkDescriptorImageInfo> infos(writes.size());
    std::vector<VkWriteDescriptorSet> w(writes.size());
    for (size_t i = 0; i < writes.size(); i++)
    {
        infos[i] = { VK_NULL_HANDLE, writes[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[i].dstSet = b.textureSets[uint32_t(writes[i].kind)];
        w[i].dstBinding = 0;
        w[i].dstArrayElement = writes[i].index;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d.device, uint32_t(w.size()), w.data(), 0, nullptr);
}

void CreateSamplers(Device& d, const int (&anisotropy)[kSamplerCount], VkSampler (&out)[kSamplerCount])
{
    for (uint32_t i = 0; i < kSamplerCount; i++)
    {
        const bool linear = i == 0 || i == 1 || i >= 4;
        const bool wrap = i == 0 || i == 3 || i >= 4;
        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter = si.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.mipmapMode = linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = wrap ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        if (linear && anisotropy[i] > 1)
        {
            si.anisotropyEnable = VK_TRUE;
            si.maxAnisotropy = std::min(float(anisotropy[i]), d.maxAnisotropy);
        }
        Check(vkCreateSampler(d.device, &si, nullptr, &out[i]), "vkCreateSampler");
    }
}
}
