#pragma once
// The Vulkan device ReGPU renders on (docs/vulkan-conversion-plan.md, step 2): instance, physical device, one
// graphics+compute queue, the VMA allocator, the pipeline cache and the queue's timeline semaphore. The presenter
// creates it at start; RenderDevice() hands it out, as it handed out the D3D12 device before.
//
// Baseline: Vulkan 1.3 with dynamic rendering, synchronization2, maintenance4 and descriptor indexing
// (update-after-bind, partially bound, runtime arrays), timeline semaphores and robustness2 null descriptors. A GPU
// without them is refused with a FATAL line naming what is missing. Optional, used when offered: conditional
// rendering (SetPredication), external host memory (guest memory read in place), push descriptors, the swap chain
// and the memory budget.
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>

#include "gpu/vk/vk_loader.h"

#include <vk_mem_alloc.h>

namespace vk
{
struct Device
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath;

    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceDriverProperties driver{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    VkPhysicalDeviceMemoryProperties memory{};
    std::string name;      // "AMD Radeon RX 7900 XTX"
    uint8_t luid[8] = {};  // Windows: the adapter's LUID (deviceLUIDValid), for matching DXGI and the upscaler SDKs
    bool luidValid = false;

    // The GPU shares system memory (an integrated GPU: every device-local heap is also host-visible, or the device type
    // says so). The D3D12 renderer read this from D3D12_FEATURE_DATA_ARCHITECTURE1::UMA (g_uma).
    bool uma = false;
    // A device-local memory type that is also host-visible with more than 256 MB behind it (resizable BAR, or UMA):
    // what D3D12's GPU upload heap was (REGPU_GPU_UPLOAD_RING).
    bool largeBar = false;

    // Optional extensions in use.
    bool conditionalRendering = false; // VK_EXT_conditional_rendering
    bool externalMemoryHost = false;   // VK_EXT_external_memory_host
    uint32_t hostPointerAlignment = 0; // minImportedHostPointerAlignment
    bool pushDescriptor = false;       // VK_KHR_push_descriptor
    bool swapchain = false;            // VK_KHR_swapchain (+ the instance's surface extensions)
    bool memoryBudget = false;         // VK_EXT_memory_budget
    bool validation = false;           // REGPU_VK_VALIDATION: VK_LAYER_KHRONOS_validation is on
    float maxAnisotropy = 1.0f;

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

    // The queue's timeline: each submission through Submit signals the next value. Replaces the D3D12 renderer's
    // fences (Signal / GetCompletedValue / SetEventOnCompletion).
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<uint64_t> submitted{ 0 };
    // vkQueueSubmit and vkQueuePresentKHR need the queue externally synchronised (a D3D12 queue was free-threaded).
    std::mutex queueLock;
};

// Creates the device. cacheDir: where the pipeline cache is kept (Config::userDir). Ends the process with a FATAL
// line when no GPU meets the baseline. REGPU_ADAPTER picks the GPU, as it did for D3D12: part of its name ("7900"),
// an index in Vulkan's enumeration order ("1"), or "high" / "low" (a discrete / an integrated GPU).
void CreateDevice(Device& d, const std::filesystem::path& cacheDir);
// Writes the pipeline cache to disk. Safe while other threads create pipelines with it. Returns the bytes written.
size_t SavePipelineCache(Device& d);

// Submits command buffers on the device's queue (locked). Waits on `waits` (binary or timeline, with their values and
// stages) and signals the queue's timeline with the next value, which it returns, plus `signals`. Optionally a fence.
struct SemaphoreUse
{
    VkSemaphore semaphore = VK_NULL_HANDLE;
    uint64_t value = 0; // timeline value (0 for a binary semaphore)
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
};
uint64_t Submit(Device& d, const VkCommandBuffer* buffers, uint32_t count, const SemaphoreUse* waits = nullptr, uint32_t waitCount = 0,
    const SemaphoreUse* signals = nullptr, uint32_t signalCount = 0, VkFence fence = VK_NULL_HANDLE);
// The last timeline value the GPU has finished.
uint64_t Completed(const Device& d);
// Blocks until the GPU has finished `value` (true), or `timeoutNs` passed (false).
bool WaitFor(const Device& d, uint64_t value, uint64_t timeoutNs = UINT64_MAX);

// The result's name, for logs ("VK_ERROR_DEVICE_LOST").
const char* ResultName(VkResult result);
// Ends the process with a FATAL line if `result` is an error (the D3D12 renderer's Check).
void Check(VkResult result, const char* what);
// Names an object for validation messages, RenderDoc and RGP (only with REGPU_VK_VALIDATION or a debugger's layer).
void SetName(const Device& d, VkObjectType type, uint64_t handle, const char* name);
}
