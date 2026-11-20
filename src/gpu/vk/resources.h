#pragma once
// Vulkan resources for the renderer (docs/vulkan-conversion-plan.md, step 4): buffers and images in VMA memory, the
// per-image state that replaces D3D12's resource states, deferred destruction on the queue timeline, and the format
// mapping from the DXGI formats renderer.cpp names today.
//
// Image state. D3D12 resources carried a state the renderer tracked by hand (Surface::state, StateBefore/After in
// each barrier) and promoted or decayed implicitly at command list boundaries. Vulkan does neither: every Image keeps
// its layout and the stages and accesses of its last use, and Transition() records the sync2 barrier from there.
// State is tracked on the recording thread (as D3D12's states were), in recording order.
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include "gpu/vk/device.h"
#include "gpu/vk/recorded.h"



namespace vk
{
// Where a buffer lives.
enum class Memory
{
    Device,     // GPU-only (copies in and out)
    Upload,     // host-visible, written sequentially by the CPU, read by the GPU (the upload ring: D3D12's UPLOAD heap)
    UploadVram, // device-local and host-visible when the GPU offers it (resizable BAR / UMA: D3D12's GPU upload heap),
                // else as Upload
    Readback,   // host-visible and cached, written by the GPU and read by the CPU (D3D12's READBACK heap)
};

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint8_t* mapped = nullptr; // host-visible memories stay mapped
    VkDeviceSize size = 0;
    bool inVram = false;       // UploadVram that got device-local memory
    VkDeviceMemory imported = VK_NULL_HANDLE; // ImportHostMemory's memory (allocation is null then)
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};

Buffer CreateBuffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage, Memory memory, const char* name = nullptr);
void Destroy(Device& d, Buffer& b);
// A buffer over existing host memory (VK_EXT_external_memory_host; D3D12's OpenExistingHeapFromAddress): the GPU reads
// and writes `host` in place. `host` and `size` must be aligned to Device::hostPointerAlignment. An empty Buffer when the
// device can't import it (the caller copies instead). Not owned: Destroy frees the import, not the memory.
Buffer ImportHostMemory(Device& d, void* host, VkDeviceSize size, VkBufferUsageFlags usage, const char* name = nullptr);

// The last use of an image (or of the subresources Transition covered): what the next barrier waits on.
struct ImageState
{
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
};

struct ImageDesc
{
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t width = 1, height = 1, depth = 1;
    uint32_t mips = 1, layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    bool cube = false;   // layers is a multiple of 6 and the sampled view is a cube
    // A second format views of the image may use (VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT with a format list): D3D12's
    // typeless resources viewed as two formats (R32_TYPELESS as depth and as R32_FLOAT).
    VkFormat alsoViewedAs = VK_FORMAT_UNDEFINED;
    bool dedicated = false; // its own memory block (large render targets)
    const char* name = nullptr;
};

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;   // sampled: every mip and layer (cube when desc.cube; depth aspect only for
                                         // depth-stencil formats)
    VkImageView target = VK_NULL_HANDLE; // mip 0, layer 0 as an attachment (colour or depth-stencil), when the usage has one
    ImageDesc desc{};
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT; // every aspect of the format
    ImageState state{};
    explicit operator bool() const { return image != VK_NULL_HANDLE; }
    VkExtent2D Extent() const { return { desc.width, desc.height }; }
};

Image CreateImage(Device& d, const ImageDesc& desc);
void Destroy(Device& d, Image& image);
// A view of other mips, layers, format or aspect (the renderer's per-mip resolve targets, R32 views of depth).
VkImageView CreateView(Device& d, const Image& image, VkImageViewType type, VkFormat format, VkImageAspectFlags aspects, uint32_t baseMip,
    uint32_t mips, uint32_t baseLayer, uint32_t layers);

// Typical uses: the layout, stages and accesses that go with them.
enum class Use
{
    Sampled,        // read by shaders (any stage): SHADER_READ_ONLY_OPTIMAL
    ColorTarget,    // COLOR_ATTACHMENT_OPTIMAL, read and written (render passes load)
    DepthTarget,    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL, read and written
    DepthReadOnly,  // DEPTH_STENCIL_READ_ONLY_OPTIMAL: depth-tested and sampled at once
    CopySource,     // TRANSFER_SRC_OPTIMAL
    CopyDest,       // TRANSFER_DST_OPTIMAL
    Storage,        // GENERAL, read and written by compute shaders
    Present,        // PRESENT_SRC_KHR
};
ImageState StateFor(Use use);

// Records the barrier from the image's state to `to` (nothing when it is already there and only read), and makes `to`
// its state. Several images: batch them with a Barriers.
void Transition(RecordedList& list, Image& image, Use to);
void Transition(RecordedList& list, Image& image, const ImageState& to);

// Collects barriers and records them as one vkCmdPipelineBarrier2.
struct Barriers
{
    std::vector<VkImageMemoryBarrier2> images;
    std::vector<VkBufferMemoryBarrier2> buffers;
    VkMemoryBarrier2 global{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    bool hasGlobal = false;
    // Adds a transition of the image (all subresources) and makes `to` its state.
    void Add(Image& image, const ImageState& to);
    void Add(Image& image, Use to) { Add(image, StateFor(to)); }
    // Adds a transition of some mips/layers only; the image's tracked state is not changed (the caller tracks it).
    void AddRange(const Image& image, const ImageState& from, const ImageState& to, uint32_t baseMip, uint32_t mips, uint32_t baseLayer,
        uint32_t layers);
    // A memory dependency (buffer writes before reads: copies, the upload ring, write-backs).
    void AddGlobal(VkPipelineStageFlags2 srcStages, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStages, VkAccessFlags2 dstAccess);
    void Record(RecordedList& list);
    bool Empty() const { return images.empty() && buffers.empty() && !hasGlobal; }
};

// Deferred destruction (D3D12's FrameResources::garbage of ComPtrs): run `destroy` once the GPU has finished the
// queue timeline value `after` (Device::submitted at the time of release: everything recorded so far). Collect runs
// what is due; Collect(d, true) waits for all.
class Graveyard
{
public:
    void Bury(uint64_t after, std::function<void()> destroy);
    void Bury(Device& d, Buffer b);
    void Bury(Device& d, Image im);
    void Bury(Device& d, VkImageView view);
    void Collect(Device& d, bool waitAll = false);
    size_t Pending();

private:
    std::mutex m_mutex;
    std::vector<std::pair<uint64_t, std::function<void()>>> m_dead;
};

// DXGI format -> Vulkan format, for the formats renderer.cpp uses (VK_FORMAT_UNDEFINED for others). Typeless depth
// formats map to their depth format.
VkFormat ToVkFormat(uint32_t dxgiFormat); // a DXGI_FORMAT value (removed with D3D12)
bool IsDepthFormat(VkFormat format);
bool HasStencil(VkFormat format);
// Bytes per texel, or per 4x4 block for block-compressed formats.
uint32_t FormatBytes(VkFormat format);
bool IsBlockCompressed(VkFormat format);
}
