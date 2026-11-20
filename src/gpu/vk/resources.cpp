#include "gpu/vk/resources.h"

#include <algorithm>

#if defined(_WIN32)
#include <dxgiformat.h>
#endif

#include "log.h"

namespace vk
{
Buffer CreateBuffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage, Memory memory, const char* name)
{
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = size;
    ci.usage = usage;
    VmaAllocationCreateInfo ai{};
    switch (memory)
    {
    case Memory::Device:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case Memory::Upload:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case Memory::UploadVram:
        // Device-local and mappable where the GPU has it (resizable BAR, UMA); VMA falls back to host memory.
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case Memory::Readback:
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    }
    VmaAllocationInfo info{};
    const VkResult result = vmaCreateBuffer(d.allocator, &ci, &ai, &b.buffer, &b.allocation, &info);
    if (result != VK_SUCCESS)
    {
        Log("vk: buffer of %llu KB (%s) not created: %s", (unsigned long long)(size >> 10), name ? name : "?", ResultName(result));
        return {};
    }
    b.mapped = static_cast<uint8_t*>(info.pMappedData);
    VkMemoryPropertyFlags flags = 0;
    vmaGetAllocationMemoryProperties(d.allocator, b.allocation, &flags);
    b.inVram = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    // ALLOW_TRANSFER_INSTEAD: VMA may hand out unmappable VRAM; the renderer's rings must be written by the CPU.
    if ((memory == Memory::UploadVram || memory == Memory::Upload) && !b.mapped)
    {
        vmaDestroyBuffer(d.allocator, b.buffer, b.allocation);
        return memory == Memory::UploadVram ? CreateBuffer(d, size, usage, Memory::Upload, name) : Buffer{};
    }
    if (name)
    {
        SetName(d, VK_OBJECT_TYPE_BUFFER, uint64_t(b.buffer), name);
        vmaSetAllocationName(d.allocator, b.allocation, name);
    }
    return b;
}

void Destroy(Device& d, Buffer& b)
{
    if (b.buffer)
        vmaDestroyBuffer(d.allocator, b.buffer, b.allocation);
    b = {};
}

namespace
{
VkImageAspectFlags AspectsOf(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}
}

VkImageView CreateView(Device& d, const Image& image, VkImageViewType type, VkFormat format, VkImageAspectFlags aspects, uint32_t baseMip,
    uint32_t mips, uint32_t baseLayer, uint32_t layers)
{
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = image.image;
    vi.viewType = type;
    vi.format = format;
    vi.subresourceRange = { aspects, baseMip, mips, baseLayer, layers };
    VkImageView view = VK_NULL_HANDLE;
    const VkResult result = vkCreateImageView(d.device, &vi, nullptr, &view);
    if (result != VK_SUCCESS)
        Log("vk: image view (%s, format %d) not created: %s", image.desc.name ? image.desc.name : "?", int(format), ResultName(result));
    return view;
}

Image CreateImage(Device& d, const ImageDesc& desc)
{
    Image im;
    im.desc = desc;
    im.aspects = AspectsOf(desc.format);
    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = { desc.width, desc.height, desc.depth };
    ci.mipLevels = desc.mips;
    ci.arrayLayers = desc.layers;
    ci.samples = desc.samples;
    ci.usage = desc.usage;
    ci.flags = desc.cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    VkFormat formats[2] = { desc.format, desc.alsoViewedAs };
    VkImageFormatListCreateInfo list{ VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO };
    if (desc.alsoViewedAs != VK_FORMAT_UNDEFINED && desc.alsoViewedAs != desc.format)
    {
        ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        list.viewFormatCount = 2;
        list.pViewFormats = formats;
        ci.pNext = &list;
    }
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (desc.dedicated)
        ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    const VkResult result = vmaCreateImage(d.allocator, &ci, &ai, &im.image, &im.allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        Log("vk: image %ux%ux%u format %d (%s) not created: %s", desc.width, desc.height, desc.depth, int(desc.format),
            desc.name ? desc.name : "?", ResultName(result));
        return {};
    }
    if (desc.name)
    {
        SetName(d, VK_OBJECT_TYPE_IMAGE, uint64_t(im.image), desc.name);
        vmaSetAllocationName(d.allocator, im.allocation, desc.name);
    }
    if (desc.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT))
    {
        const VkImageViewType type = desc.type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
            : desc.cube                                            ? (desc.layers > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE)
            : desc.layers > 1                                      ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                   : VK_IMAGE_VIEW_TYPE_2D;
        // A depth-stencil image is sampled for its depth (D3D12's R32_FLOAT_X8X24 SRV of D32_S8).
        const VkImageAspectFlags sampled = im.aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : im.aspects;
        im.view = CreateView(d, im, type, desc.format, sampled, 0, desc.mips, 0, desc.layers);
    }
    if (desc.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
        im.target = CreateView(d, im, VK_IMAGE_VIEW_TYPE_2D, desc.format, im.aspects, 0, 1, 0, 1);
    return im;
}

void Destroy(Device& d, Image& image)
{
    if (image.view)
        vkDestroyImageView(d.device, image.view, nullptr);
    if (image.target)
        vkDestroyImageView(d.device, image.target, nullptr);
    if (image.image)
        vmaDestroyImage(d.allocator, image.image, image.allocation);
    image = {};
}

ImageState StateFor(Use use)
{
    constexpr VkPipelineStageFlags2 shaders =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    switch (use)
    {
    case Use::Sampled: return { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, shaders, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT };
    case Use::ColorTarget: return { VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, kColorTargetAccess };
    case Use::DepthTarget: return { VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, kDepthTargetStages, kDepthTargetAccess };
    case Use::DepthReadOnly:
        return { VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, kDepthTargetStages | shaders,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT };
    case Use::CopySource: return { VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT };
    case Use::CopyDest: return { VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
    case Use::Storage:
        return { VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
    case Use::Present: return { VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE };
    }
    return {};
}

namespace
{
constexpr VkAccessFlags2 kWrites = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

VkImageMemoryBarrier2 MakeBarrier(const Image& image, const ImageState& from, const ImageState& to, uint32_t baseMip, uint32_t mips,
    uint32_t baseLayer, uint32_t layers)
{
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = from.stages;
    // Only writes need making available; a read-after-read needs the execution dependency alone.
    b.srcAccessMask = from.access & kWrites;
    b.dstStageMask = to.stages;
    b.dstAccessMask = to.access;
    b.oldLayout = from.layout;
    b.newLayout = to.layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image.image;
    b.subresourceRange = { image.aspects, baseMip, mips, baseLayer, layers };
    return b;
}

// Whether going from `from` to `to` needs no barrier: same layout, and neither side writes (a read after a read).
bool NoBarrierNeeded(const ImageState& from, const ImageState& to)
{
    return from.layout == to.layout && !(from.access & kWrites) && !(to.access & kWrites) && (to.stages & ~from.stages) == 0;
}
}

void Barriers::Add(Image& image, const ImageState& to)
{
    if (NoBarrierNeeded(image.state, to))
        return;
    images.push_back(MakeBarrier(image, image.state, to, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS));
    image.state = to;
}

void Barriers::AddRange(const Image& image, const ImageState& from, const ImageState& to, uint32_t baseMip, uint32_t mips, uint32_t baseLayer,
    uint32_t layers)
{
    images.push_back(MakeBarrier(image, from, to, baseMip, mips, baseLayer, layers));
}

void Barriers::AddGlobal(VkPipelineStageFlags2 srcStages, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStages, VkAccessFlags2 dstAccess)
{
    global.srcStageMask |= srcStages;
    global.srcAccessMask |= srcAccess;
    global.dstStageMask |= dstStages;
    global.dstAccessMask |= dstAccess;
    hasGlobal = true;
}

void Barriers::Record(RecordedList& list)
{
    if (Empty())
        return;
    list.Barrier(uint32_t(images.size()), images.data(), uint32_t(buffers.size()), buffers.data(), hasGlobal ? &global : nullptr);
    images.clear();
    buffers.clear();
    global = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    hasGlobal = false;
}

void Transition(RecordedList& list, Image& image, const ImageState& to)
{
    if (NoBarrierNeeded(image.state, to))
        return;
    const VkImageMemoryBarrier2 b = MakeBarrier(image, image.state, to, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    list.Barrier(1, &b);
    image.state = to;
}

void Transition(RecordedList& list, Image& image, Use to)
{
    Transition(list, image, StateFor(to));
}

void Graveyard::Bury(uint64_t after, std::function<void()> destroy)
{
    std::lock_guard lock(m_mutex);
    m_dead.emplace_back(after, std::move(destroy));
}

void Graveyard::Bury(Device& d, Buffer b)
{
    if (b)
        Bury(d.submitted.load(std::memory_order_acquire) + 1, [&d, b]() mutable { Destroy(d, b); });
}

void Graveyard::Bury(Device& d, Image im)
{
    if (im)
        Bury(d.submitted.load(std::memory_order_acquire) + 1, [&d, im]() mutable { Destroy(d, im); });
}

void Graveyard::Bury(Device& d, VkImageView view)
{
    if (view)
        Bury(d.submitted.load(std::memory_order_acquire) + 1, [&d, view] { vkDestroyImageView(d.device, view, nullptr); });
}

void Graveyard::Collect(Device& d, bool waitAll)
{
    std::vector<std::function<void()>> due;
    {
        std::lock_guard lock(m_mutex);
        if (m_dead.empty())
            return;
        // waitAll (shutdown, device loss): everything goes once the GPU has finished all it was given, including what was
        // buried after the last submission.
        if (waitAll)
            WaitFor(d, d.submitted.load(std::memory_order_acquire));
        const uint64_t done = waitAll ? UINT64_MAX : Completed(d);
        auto keep = std::partition(m_dead.begin(), m_dead.end(), [&](const auto& e) { return e.first > done; });
        for (auto it = keep; it != m_dead.end(); ++it)
            due.push_back(std::move(it->second));
        m_dead.erase(keep, m_dead.end());
    }
    for (auto& fn : due)
        fn();
}

size_t Graveyard::Pending()
{
    std::lock_guard lock(m_mutex);
    return m_dead.size();
}

VkFormat ToVkFormat(uint32_t dxgiFormat)
{
#if defined(_WIN32)
    switch (DXGI_FORMAT(dxgiFormat))
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
    case DXGI_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
    case DXGI_FORMAT_R32_TYPELESS: return VK_FORMAT_D32_SFLOAT; // a depth buffer also viewed as R32_FLOAT (alsoViewedAs)
    case DXGI_FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return VK_FORMAT_D32_SFLOAT_S8_UINT; // its depth view (aspect DEPTH)
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;   // its stencil view (aspect STENCIL)
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DXGI_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case DXGI_FORMAT_R16G16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case DXGI_FORMAT_R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
    case DXGI_FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16_UINT: return VK_FORMAT_R16_UINT;
    case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R8G8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R8_UINT: return VK_FORMAT_R8_UINT;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case DXGI_FORMAT_B5G6R5_UNORM: return VK_FORMAT_R5G6B5_UNORM_PACK16;
    case DXGI_FORMAT_B5G5R5A1_UNORM: return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
    case DXGI_FORMAT_B4G4R4A4_UNORM: return VK_FORMAT_A4R4G4B4_UNORM_PACK16;
    case DXGI_FORMAT_BC1_UNORM: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case DXGI_FORMAT_BC2_UNORM: return VK_FORMAT_BC2_UNORM_BLOCK;
    case DXGI_FORMAT_BC3_UNORM: return VK_FORMAT_BC3_UNORM_BLOCK;
    case DXGI_FORMAT_BC4_UNORM: return VK_FORMAT_BC4_UNORM_BLOCK;
    case DXGI_FORMAT_BC5_UNORM: return VK_FORMAT_BC5_UNORM_BLOCK;
    default: return VK_FORMAT_UNDEFINED;
    }
#else
    (void)dxgiFormat;
    return VK_FORMAT_UNDEFINED;
#endif
}

bool IsDepthFormat(VkFormat format)
{
    return (AspectsOf(format) & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
}

bool HasStencil(VkFormat format)
{
    return (AspectsOf(format) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
}

bool IsBlockCompressed(VkFormat format)
{
    return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

uint32_t FormatBytes(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_S8_UINT:
        return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_A4R4G4B4_UNORM_PACK16:
        return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: // 4 depth + 1 stencil, copied per aspect
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
        return 16;
    default:
        return 4;
    }
}
}
