// gpu/vk/resources.h on the GPU: images of the formats renderer.cpp uses are created with their views, filled from an
// upload buffer and read back through the tracked transitions (Transition / Barriers), a depth-stencil image is cleared
// and its depth read back, and destroyed objects go through the Graveyard. Exit code 0: all as expected (run with
// REGPU_VK_VALIDATION=2 to check the barriers too).
#include <cstdio>
#include <cstring>

#include <dxgiformat.h>

#include "gpu/vk/resources.h"

int main()
{
    vk::Device d;
    vk::CreateDevice(d, {});
    bool ok = true;

    // Every DXGI format renderer.cpp names maps, and an image of it can be made (sampled, copied to and from).
    const DXGI_FORMAT formats[] = { DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC3_UNORM,
        DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_D32_FLOAT_S8X24_UINT };
    std::vector<vk::Image> images;
    for (DXGI_FORMAT f : formats)
    {
        vk::ImageDesc desc;
        desc.format = vk::ToVkFormat(f);
        desc.width = desc.height = 64;
        desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (vk::IsDepthFormat(desc.format))
            desc.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        desc.name = "test image";
        vk::Image im = desc.format ? vk::CreateImage(d, desc) : vk::Image{};
        if (!im || !im.view)
        {
            printf("vkresources: DXGI format %d -> %d: no image\n", int(f), int(desc.format));
            ok = false;
        }
        images.push_back(im);
    }

    // A round trip: upload -> RGBA8 image -> readback, and a depth clear read back.
    vk::Buffer upload = vk::CreateBuffer(d, 64 * 64 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::Memory::UploadVram, "upload");
    vk::Buffer readback = vk::CreateBuffer(d, 64 * 64 * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::Memory::Readback, "readback");
    for (uint32_t i = 0; i < 64 * 64; i++)
        reinterpret_cast<uint32_t*>(upload.mapped)[i] = i * 2654435761u;

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

    vk::Image& rgba = images[0];
    vk::Image& depth = images.back();
    vk::Transition(list, rgba, vk::Use::CopyDest);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { 64, 64, 1 };
    list.CopyBufferToImage(upload.buffer, rgba.image, rgba.state.layout, 1, &region);
    vk::Barriers barriers;
    barriers.Add(rgba, vk::Use::CopySource);
    barriers.Add(depth, vk::Use::DepthTarget);
    barriers.Record(list);
    list.CopyImageToBuffer(rgba.image, rgba.state.layout, readback.buffer, 1, &region);
    list.ClearDepthStencil(depth.target, depth.Extent(), true, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0.25f, 7, 0, nullptr);
    vk::Transition(list, depth, vk::Use::CopySource);
    VkBufferImageCopy depthRegion{};
    depthRegion.bufferOffset = 64 * 64 * 4;
    depthRegion.imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    depthRegion.imageExtent = { 64, 64, 1 };
    list.CopyImageToBuffer(depth.image, depth.state.layout, readback.buffer, 1, &depthRegion);
    barriers.AddGlobal(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    barriers.Record(list);
    // A redundant transition records nothing (the image is already a copy source and only read).
    vk::Transition(list, rgba, vk::Use::CopySource);

    VkCommandBuffer done = list.Real();
    vkEndCommandBuffer(done);
    const uint64_t value = vk::Submit(d, &done, 1);
    ok &= vk::WaitFor(d, value, 5'000'000'000ull);
    vmaInvalidateAllocation(d.allocator, readback.allocation, 0, VK_WHOLE_SIZE);
    bool same = true;
    for (uint32_t i = 0; i < 64 * 64 && same; i++)
        same = reinterpret_cast<const uint32_t*>(readback.mapped)[i] == i * 2654435761u;
    float depthValue = 0;
    memcpy(&depthValue, readback.mapped + 64 * 64 * 4 + 4 * 100, 4);
    printf("vkresources: %zu formats, upload %s (%s), round trip %s, depth clear %.3f %s\n", images.size(),
        upload.inVram ? "in VRAM" : "in system memory", upload.mapped ? "mapped" : "NOT MAPPED", same ? "ok" : "WRONG", depthValue,
        depthValue == 0.25f ? "ok" : "WRONG");
    ok &= same && depthValue == 0.25f;

    // Everything goes through the graveyard: due once the GPU passed the submission after it was buried.
    vk::Graveyard graveyard;
    for (vk::Image& im : images)
        graveyard.Bury(d, im);
    graveyard.Bury(d, upload);
    graveyard.Bury(d, readback);
    const size_t buried = graveyard.Pending();
    graveyard.Collect(d, true);
    printf("vkresources: graveyard %zu buried, %zu left\n", buried, graveyard.Pending());
    ok &= graveyard.Pending() == 0;
    if (d.validation)
    {
        printf("vkresources: validation messages %llu\n", (unsigned long long)vk::g_validationMessages.load());
        ok &= vk::g_validationMessages.load() == 0;
    }
    return ok ? 0 : 1;
}
