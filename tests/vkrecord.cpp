// vk::RecordedList on the GPU (gpu/vk/recorded.h): records on this thread, replays on the recorder's thread, then
// checks the pixels. Covers the lazy render passes (a clear of a bound target inside rendering, a clear of an unbound
// image in a pass of its own), barriers, an image-to-buffer copy, an occlusion query with no draws (0 samples) and the
// hand-off that ends a command buffer for submission. Exit code 0: all as expected.
#include <cstdio>
#include <cstring>

#include "gpu/vk/device.h"
#include "gpu/vk/recorded.h"

namespace
{
struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

Image MakeImage(vk::Device& d, uint32_t w, uint32_t h)
{
    Image im;
    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk::Check(vmaCreateImage(d.allocator, &ci, &ai, &im.image, &im.memory, nullptr), "vmaCreateImage");
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk::Check(vkCreateImageView(d.device, &vi, nullptr, &im.view), "vkCreateImageView");
    return im;
}

VkImageMemoryBarrier2 Transition(VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return b;
}

struct HandOffState
{
    vk::Device* d;
    uint64_t value = 0;
};
}

int main()
{
    vk::Device d;
    vk::CreateDevice(d, {});
    constexpr uint32_t W = 64, H = 32;
    Image a = MakeImage(d, W, H), b = MakeImage(d, W, H);

    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = W * H * 4 * 2 + 64;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer readback;
    VmaAllocation readbackMemory;
    VmaAllocationInfo readbackInfo;
    vk::Check(vmaCreateBuffer(d.allocator, &bci, &bai, &readback, &readbackMemory, &readbackInfo), "vmaCreateBuffer");

    VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qci.queryType = VK_QUERY_TYPE_OCCLUSION;
    qci.queryCount = 1;
    VkQueryPool queries;
    vk::Check(vkCreateQueryPool(d.device, &qci, nullptr, &queries), "vkCreateQueryPool");

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
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    static vk::RecordedList list; // the replay thread is detached: the list lives as long as the process
    list.Init(cb);

    VkImageMemoryBarrier2 toTarget[2] = {
        Transition(a.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
        Transition(b.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
    };
    list.Barrier(2, toTarget);
    list.ResetQueries(queries, 0, 1);

    // a: bound as the target; cleared whole to red, then its right half to green (inside the same rendering).
    vk::Targets t{};
    t.colorCount = 1;
    t.color[0] = a.view;
    t.extent = { W, H };
    list.SetTargets(t);
    const float red[4] = { 1, 0, 0, 1 }, green[4] = { 0, 1, 0, 1 }, blue[4] = { 0, 0, 1, 1 };
    list.ClearColor(a.view, { W, H }, red, 0, nullptr);
    const VkRect2D right{ { int32_t(W / 2), 0 }, { W / 2, H } };
    list.ClearColor(a.view, { W, H }, green, 1, &right);
    // b: not bound: cleared to blue in a pass of its own.
    list.ClearColor(b.view, { W, H }, blue, 0, nullptr);
    // An occlusion query around no draws: 0 samples.
    list.BeginQuery(queries, 0);
    list.EndQuery(queries, 0);

    VkImageMemoryBarrier2 toCopy[2] = {
        Transition(a.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT),
        Transition(b.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT),
    };
    list.Barrier(2, toCopy);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { W, H, 1 };
    list.CopyImageToBuffer(a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &region);
    region.bufferOffset = W * H * 4;
    list.CopyImageToBuffer(b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &region);
    list.CopyQueryResults(queries, 0, 1, readback, W * H * 8);
    VkMemoryBarrier2 toHost{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    toHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    list.Barrier(0, nullptr, 0, nullptr, &toHost);

    // The hand-off ends the buffer and submits it on the replay thread.
    static HandOffState state{ &d };
    list.HandOff(VK_NULL_HANDLE,
        [](VkCommandBuffer closed, void* ctx) {
            auto* s = static_cast<HandOffState*>(ctx);
            vk::Check(vkEndCommandBuffer(closed), "vkEndCommandBuffer");
            s->value = vk::Submit(*s->d, &closed, 1);
        },
        &state);
    list.Real(); // waits for the replay (and so for the hand-off) to finish
    const bool waited = vk::WaitFor(d, state.value, 5'000'000'000ull);

    vmaInvalidateAllocation(d.allocator, readbackMemory, 0, VK_WHOLE_SIZE);
    const uint8_t* px = static_cast<const uint8_t*>(readbackInfo.pMappedData);
    auto at = [&](const uint8_t* base, uint32_t x, uint32_t y) { return base + (y * W + x) * 4; };
    auto is = [](const uint8_t* p, uint8_t r, uint8_t g, uint8_t bl) { return p[0] == r && p[1] == g && p[2] == bl && p[3] == 255; };
    const bool left = is(at(px, 3, 5), 255, 0, 0), rightOk = is(at(px, W - 3, H - 5), 0, 255, 0);
    const bool other = is(at(px + W * H * 4, 10, 10), 0, 0, 255);
    uint64_t samples = ~0ull;
    memcpy(&samples, px + W * H * 8, 8);
    printf("vkrecord: submitted %s; bound clear %s, rect clear %s, own-pass clear %s, empty query %llu samples %s\n", waited ? "ok" : "TIMED OUT",
        left ? "ok" : "WRONG", rightOk ? "ok" : "WRONG", other ? "ok" : "WRONG", (unsigned long long)samples, samples == 0 ? "ok" : "WRONG");
    return waited && left && rightOk && other && samples == 0 ? 0 : 1;
}
