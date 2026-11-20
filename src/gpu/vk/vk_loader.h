#pragma once
// Vulkan entry points, loaded at run time through volk (thirdparty/volk): no link to vulkan-1.lib or libvulkan.so,
// and a machine without Vulkan fails with a message rather than a missing-DLL error. Every Vulkan call site uses the
// usual names; volk's function pointers stand behind them. Include this instead of <vulkan/vulkan.h>.
#include <volk.h>

namespace vkldr
{
// Loads the Vulkan loader (vulkan-1.dll / libvulkan.so.1) and the global entry points.
bool Load();
// The instance's entry points, and every device entry point through the instance's dispatch.
bool ResolveInstance(VkInstance instance);
// The device's entry points straight from the driver (no loader trampoline): call once the device exists. Only valid
// with one VkDevice per process, which is ReGPU's case.
void ResolveDevice(VkDevice device);
inline PFN_vkVoidFunction GetInstanceProcAddr(VkInstance instance, const char* name)
{
    return vkGetInstanceProcAddr(instance, name);
}
}
