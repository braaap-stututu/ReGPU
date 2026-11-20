#define VOLK_IMPLEMENTATION
#include "gpu/vk/vk_loader.h"

#include <cstdio>

namespace vkldr
{
bool Load()
{
    static const VkResult result = volkInitialize();
    if (result != VK_SUCCESS)
        fprintf(stderr, "vk: no Vulkan loader (vulkan-1.dll / libvulkan.so.1): is a Vulkan driver installed?\n");
    return result == VK_SUCCESS;
}

bool ResolveInstance(VkInstance instance)
{
    volkLoadInstance(instance);
    return vkCreateDevice != nullptr && vkQueueSubmit != nullptr;
}

void ResolveDevice(VkDevice device)
{
    volkLoadDevice(device);
}
}
