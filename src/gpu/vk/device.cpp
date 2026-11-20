#include "gpu/vk/device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "log.h"
#include "regpu/regpu.h"

namespace vk
{
std::atomic<uint64_t> g_validationMessages{ 0 };
namespace
{
std::string Lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(tolower(c)); });
    return s;
}

bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name)
{
    for (const auto& e : list)
        if (!strcmp(e.extensionName, name))
            return true;
    return false;
}

VkBool32 VKAPI_PTR OnValidationMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        return VK_FALSE;
    g_validationMessages.fetch_add(1, std::memory_order_relaxed);
    // The first 3 of each message id, with the stack of the call that made it (as REGPU_D3D12_STACKS did).
    static std::mutex m;
    static std::unordered_map<int32_t, int> seen;
    {
        std::lock_guard lock(m);
        if (seen[data->messageIdNumber]++ >= 3)
            return VK_FALSE;
    }
    const auto stack = regpu::GetHost().currentStack;
    Log("vk %s %s: %s\n  at %s", severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning",
        data->pMessageIdName ? data->pMessageIdName : "?", data->pMessage ? data->pMessage : "", stack ? stack(16).c_str() : "?");
    return VK_FALSE;
}

// What a physical device lacks of the baseline ("" when nothing).
std::string MissingBaseline(VkPhysicalDevice p)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(p, &props);
    if (props.apiVersion < VK_API_VERSION_1_3)
        return "Vulkan 1.3";
    VkPhysicalDeviceVulkan13Features v13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features v12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceRobustness2FeaturesEXT robust{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT };
    v13.pNext = &v12;
    v12.pNext = &robust;
    VkPhysicalDeviceFeatures2 f{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f.pNext = &v13;
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> ext(count);
    vkEnumerateDeviceExtensionProperties(p, nullptr, &count, ext.data());
    if (!HasExtension(ext, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME))
        return "VK_EXT_robustness2";
    vkGetPhysicalDeviceFeatures2(p, &f);
    std::string missing;
    auto need = [&](VkBool32 have, const char* what) {
        if (!have)
            missing += std::string(missing.empty() ? "" : ", ") + what;
    };
    need(v13.dynamicRendering, "dynamicRendering");
    need(v13.synchronization2, "synchronization2");
    need(v13.maintenance4, "maintenance4");
    need(v12.descriptorIndexing, "descriptorIndexing");
    need(v12.runtimeDescriptorArray, "runtimeDescriptorArray");
    need(v12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound");
    need(v12.descriptorBindingSampledImageUpdateAfterBind, "descriptorBindingSampledImageUpdateAfterBind");
    need(v12.shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing");
    need(v12.timelineSemaphore, "timelineSemaphore");
    need(robust.nullDescriptor, "nullDescriptor");
    need(f.features.textureCompressionBC, "textureCompressionBC");
    need(f.features.independentBlend, "independentBlend");
    need(f.features.samplerAnisotropy, "samplerAnisotropy");
    need(f.features.imageCubeArray, "imageCubeArray");
    return missing;
}

// How much a device is preferred when REGPU_ADAPTER names none: discrete over integrated over the rest, then by its
// device-local memory.
uint64_t Preference(VkPhysicalDevice p)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(p, &props);
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(p, &mem);
    uint64_t local = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; i++)
        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            local = std::max<uint64_t>(local, mem.memoryHeaps[i].size);
    const uint64_t type = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
        : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU            ? 2
        : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU               ? 1
                                                                                : 0;
    return type << 56 | std::min<uint64_t>(local >> 20, (1ull << 56) - 1);
}

VkPhysicalDevice PickPhysicalDevice(VkInstance instance)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> all(count);
    vkEnumeratePhysicalDevices(instance, &count, all.data());
    std::vector<VkPhysicalDevice> usable;
    for (uint32_t i = 0; i < count; i++)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(all[i], &props);
        const std::string missing = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "a GPU (software renderer)" : MissingBaseline(all[i]);
        Log("presenter: Vulkan device %u: %s (API %u.%u.%u)%s%s", i, props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
            VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), missing.empty() ? "" : ": lacks ", missing.c_str());
        if (missing.empty())
            usable.push_back(all[i]);
    }
    if (usable.empty())
        Fatal("presenter: no graphics card offers Vulkan 1.3 with the features ReGPU needs (listed above in regpu.log): update the "
              "graphics driver");

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    if (const char* want = getenv("REGPU_ADAPTER"); want && *want)
    {
        const std::string w = Lower(want);
        if (w == "high" || w == "low")
        {
            const VkPhysicalDeviceType type = w == "high" ? VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU : VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            for (VkPhysicalDevice p : usable)
            {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(p, &props);
                if (!chosen && props.deviceType == type)
                    chosen = p;
            }
        }
        else if (w.find_first_not_of("0123456789") == std::string::npos)
        {
            const uint32_t index = uint32_t(atoi(want));
            if (index < count && std::find(usable.begin(), usable.end(), all[index]) != usable.end())
                chosen = all[index];
        }
        else
            for (VkPhysicalDevice p : usable)
            {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(p, &props);
                if (!chosen && Lower(props.deviceName).find(w) != std::string::npos)
                    chosen = p;
            }
        if (!chosen)
            Log("presenter: REGPU_ADAPTER=%s matches no usable graphics card: using the default one", want);
    }
    if (!chosen)
        chosen = *std::max_element(usable.begin(), usable.end(), [](VkPhysicalDevice a, VkPhysicalDevice b) { return Preference(a) < Preference(b); });
    return chosen;
}

void LoadPipelineCache(Device& d, const std::filesystem::path& cacheDir)
{
    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    std::vector<uint8_t> data;
    if (!cacheDir.empty())
    {
        char file[64];
        snprintf(file, sizeof(file), "vk-pipelines-%04x-%04x.bin", d.props.vendorID, d.props.deviceID);
        d.pipelineCachePath = cacheDir / file;
        if (FILE* f = fopen(d.pipelineCachePath.string().c_str(), "rb"))
        {
            fseek(f, 0, SEEK_END);
            const long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (size > 0)
            {
                data.resize(size_t(size));
                if (fread(data.data(), 1, data.size(), f) != data.size())
                    data.clear();
            }
            fclose(f);
        }
        // The header must name this device and driver (VkPipelineCacheHeaderVersionOne): a cache from another driver
        // version is ignored rather than handed over (some drivers crash on one).
        if (data.size() >= 32)
        {
            uint32_t vendor, device;
            memcpy(&vendor, data.data() + 8, 4);
            memcpy(&device, data.data() + 12, 4);
            if (vendor != d.props.vendorID || device != d.props.deviceID || memcmp(data.data() + 16, d.props.pipelineCacheUUID, VK_UUID_SIZE) != 0)
            {
                Log("presenter: the Vulkan pipeline cache is from another driver: starting a new one");
                data.clear();
            }
        }
        else
            data.clear();
    }
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    Check(vkCreatePipelineCache(d.device, &ci, nullptr, &d.pipelineCache), "vkCreatePipelineCache");
    if (!data.empty())
        Log("presenter: Vulkan pipeline cache %zu KB", data.size() >> 10);
}
}

const char* ResultName(VkResult result)
{
    switch (result)
    {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    default: return "VK_ERROR_?";
    }
}

void Check(VkResult result, const char* what)
{
    if (result < 0)
        Fatal("%s failed: %s (%d)", what, ResultName(result), int(result));
}

void SetName(const Device& d, VkObjectType type, uint64_t handle, const char* name)
{
    if (!vkSetDebugUtilsObjectNameEXT || !handle)
        return;
    VkDebugUtilsObjectNameInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    vkSetDebugUtilsObjectNameEXT(d.device, &info);
}

void CreateDevice(Device& d, const std::filesystem::path& cacheDir, const HostDevice* host)
{
    if (!vkldr::Load())
        Fatal("presenter: no Vulkan loader (vulkan-1.dll): install or update the graphics driver");
    if (volkGetInstanceVersion() < VK_API_VERSION_1_3)
        Fatal("presenter: the Vulkan loader is older than 1.3: update the graphics driver");

    // The instance.
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> instanceExt(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, instanceExt.data());
    std::vector<const char*> extensions;
    const bool surface = HasExtension(instanceExt, VK_KHR_SURFACE_EXTENSION_NAME)
#if defined(_WIN32)
        && HasExtension(instanceExt, VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
        ;
#endif
    if (surface)
    {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(_WIN32)
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
    }
    const bool debugUtils = HasExtension(instanceExt, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtils)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    // REGPU_VK_VALIDATION=1: the Khronos validation layer (installed with the Vulkan SDK), its messages in regpu.log.
    // 2: also synchronization validation (missing or wrong barriers; slower).
    std::vector<const char*> layers;
    if (const char* v = getenv("REGPU_VK_VALIDATION"); v && atoi(v) != 0)
    {
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (const auto& l : available)
            if (!strcmp(l.layerName, "VK_LAYER_KHRONOS_validation"))
                layers.push_back("VK_LAYER_KHRONOS_validation");
        d.validation = !layers.empty();
        Log("presenter: REGPU_VK_VALIDATION: %s", d.validation ? "validation layer on" : "no validation layer installed (the Vulkan SDK has it)");
    }
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "ReGPU";
    app.pEngineName = "ReGPU";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = uint32_t(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();
    ici.enabledLayerCount = uint32_t(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    const VkValidationFeatureEnableEXT syncValidation = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT features{ VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
    features.enabledValidationFeatureCount = 1;
    features.pEnabledValidationFeatures = &syncValidation;
    if (d.validation && atoi(getenv("REGPU_VK_VALIDATION")) >= 2)
    {
        ici.pNext = &features;
        Log("presenter: REGPU_VK_VALIDATION=2: synchronization validation on");
    }
    if (host)
    {
        // The host's instance (Config::hostInstance): its extensions and layers are the host's choice.
        d.instance = host->instance;
        d.validation = false;
        Log("presenter: using the host's Vulkan instance and device");
    }
    else
        Check(vkCreateInstance(&ici, nullptr, &d.instance), "vkCreateInstance");
    vkldr::ResolveInstance(d.instance);
    if (d.validation && debugUtils)
    {
        VkDebugUtilsMessengerCreateInfoEXT mci{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mci.pfnUserCallback = OnValidationMessage;
        vkCreateDebugUtilsMessengerEXT(d.instance, &mci, nullptr, &d.messenger);
    }

    // The physical device.
    d.physical = host ? host->physical : PickPhysicalDevice(d.instance);
    VkPhysicalDeviceIDProperties ids{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
    d.driver.pNext = &ids;
    ids.pNext = &hostProps;
    VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props2.pNext = &d.driver;
    vkGetPhysicalDeviceProperties2(d.physical, &props2);
    d.driver.pNext = nullptr;
    d.props = props2.properties;
    d.name = d.props.deviceName;
    d.luidValid = ids.deviceLUIDValid;
    memcpy(d.luid, ids.deviceLUID, sizeof(d.luid));
    d.maxAnisotropy = d.props.limits.maxSamplerAnisotropy;
    vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory);

    // UMA: an integrated GPU, or every device-local heap host-visible too. Large BAR: a device-local, host-visible
    // memory type on a heap over 256 MB.
    d.uma = d.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++)
    {
        const auto& t = d.memory.memoryTypes[i];
        const VkMemoryPropertyFlags both = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if ((t.propertyFlags & both) == both && d.memory.memoryHeaps[t.heapIndex].size > (256ull << 20))
            d.largeBar = true;
    }

    // The queue: the first family with graphics and compute (it can present on every desktop driver).
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &count, families.data());
    d.queueFamily = UINT32_MAX;
    if (host)
        d.queueFamily = host->queueFamily;
    for (uint32_t i = 0; i < count && d.queueFamily == UINT32_MAX; i++)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            d.queueFamily = i;
    if (d.queueFamily == UINT32_MAX)
        Fatal("presenter: %s has no graphics queue", d.name.c_str());

    // Device extensions.
    vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> deviceExt(count);
    vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &count, deviceExt.data());
    std::vector<const char*> enabled = { VK_EXT_ROBUSTNESS_2_EXTENSION_NAME };
    auto optional = [&](const char* name) {
        if (!HasExtension(deviceExt, name))
            return false;
        // (A host's device: only what the host enabled.)
        if (host && std::find(host->extensions.begin(), host->extensions.end(), std::string(name)) == host->extensions.end())
            return false;
        enabled.push_back(name);
        return true;
    };
    d.swapchain = surface && optional(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    d.conditionalRendering = optional(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
    d.externalMemoryHost = optional(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    d.hostPointerAlignment = d.externalMemoryHost ? uint32_t(hostProps.minImportedHostPointerAlignment) : 0;
    d.pushDescriptor = optional(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    d.memoryBudget = optional(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    // NVIDIA's NGX (DLSS on Vulkan, dlss.cpp) needs these two where the driver has them.
    d.ngx = optional("VK_NVX_binary_import") && optional("VK_NVX_image_view_handle");

    // Features: the baseline (checked by MissingBaseline), plus what the translated shaders and the renderer use when
    // the GPU has it.
    VkPhysicalDeviceFeatures2 have{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceVulkan11Features have11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceVulkan12Features have12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceConditionalRenderingFeaturesEXT haveCond{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT };
    have.pNext = &have11;
    have11.pNext = &have12;
    have12.pNext = &haveCond;
    vkGetPhysicalDeviceFeatures2(d.physical, &have);
    const VkPhysicalDeviceFeatures& h = have.features;

    VkPhysicalDeviceFeatures2 want{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceFeatures& w = want.features;
    w.textureCompressionBC = VK_TRUE;
    w.independentBlend = VK_TRUE;
    w.samplerAnisotropy = VK_TRUE;
    w.imageCubeArray = VK_TRUE;
    w.dualSrcBlend = h.dualSrcBlend;
    w.logicOp = h.logicOp;
    w.depthClamp = h.depthClamp;
    w.depthBiasClamp = h.depthBiasClamp;
    w.depthBounds = h.depthBounds;
    w.fillModeNonSolid = h.fillModeNonSolid;
    w.geometryShader = h.geometryShader;
    w.sampleRateShading = h.sampleRateShading;
    w.occlusionQueryPrecise = h.occlusionQueryPrecise;
    w.fragmentStoresAndAtomics = h.fragmentStoresAndAtomics;
    w.vertexPipelineStoresAndAtomics = h.vertexPipelineStoresAndAtomics;
    w.shaderClipDistance = h.shaderClipDistance;
    w.shaderInt16 = h.shaderInt16;
    w.shaderInt64 = h.shaderInt64;
    w.multiDrawIndirect = h.multiDrawIndirect;
    w.drawIndirectFirstInstance = h.drawIndirectFirstInstance;
    w.shaderStorageImageExtendedFormats = h.shaderStorageImageExtendedFormats;
    w.shaderStorageImageWriteWithoutFormat = h.shaderStorageImageWriteWithoutFormat;
    w.shaderStorageImageReadWithoutFormat = h.shaderStorageImageReadWithoutFormat; // (post filters read their storage images: FinishMain)
    d.storageReadWithoutFormat = h.shaderStorageImageReadWithoutFormat;
    w.shaderImageGatherExtended = h.shaderImageGatherExtended;
    w.largePoints = h.largePoints;
    w.wideLines = h.wideLines;

    VkPhysicalDeviceVulkan11Features w11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    w11.shaderDrawParameters = have11.shaderDrawParameters;
    w11.storageBuffer16BitAccess = have11.storageBuffer16BitAccess;
    VkPhysicalDeviceVulkan12Features w12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    w12.descriptorIndexing = VK_TRUE;
    w12.runtimeDescriptorArray = VK_TRUE;
    w12.descriptorBindingPartiallyBound = VK_TRUE;
    w12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    w12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    w12.descriptorBindingVariableDescriptorCount = have12.descriptorBindingVariableDescriptorCount;
    w12.descriptorBindingUpdateUnusedWhilePending = have12.descriptorBindingUpdateUnusedWhilePending;
    d.updateUnusedWhilePending = have12.descriptorBindingUpdateUnusedWhilePending;
    w12.descriptorBindingStorageBufferUpdateAfterBind = have12.descriptorBindingStorageBufferUpdateAfterBind;
    w12.timelineSemaphore = VK_TRUE;
    w12.hostQueryReset = have12.hostQueryReset;
    w12.scalarBlockLayout = have12.scalarBlockLayout;
    w12.shaderFloat16 = have12.shaderFloat16;
    w12.samplerMirrorClampToEdge = have12.samplerMirrorClampToEdge;
    w12.separateDepthStencilLayouts = have12.separateDepthStencilLayouts;
    w12.bufferDeviceAddress = have12.bufferDeviceAddress;
    w12.drawIndirectCount = have12.drawIndirectCount;
    VkPhysicalDeviceVulkan13Features w13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    w13.dynamicRendering = VK_TRUE;
    w13.synchronization2 = VK_TRUE;
    w13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceRobustness2FeaturesEXT robust{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT };
    robust.nullDescriptor = VK_TRUE;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT };
    cond.conditionalRendering = haveCond.conditionalRendering;
    d.conditionalRendering = d.conditionalRendering && haveCond.conditionalRendering;
    want.pNext = &w11;
    w11.pNext = &w12;
    w12.pNext = &w13;
    w13.pNext = &robust;
    robust.pNext = d.conditionalRendering ? &cond : nullptr;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = d.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &want;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enabled.size());
    dci.ppEnabledExtensionNames = enabled.data();
    if (host)
    {
        // The host's device and queue: it enabled ReGPU's baseline (regpu.h, Config::hostDevice); the optional
        // extensions are the ones it lists, its features are taken as the GPU's.
        d.device = host->device;
        vkldr::ResolveDevice(d.device);
        d.queue = host->queue;
    }
    else
    {
        Check(vkCreateDevice(d.physical, &dci, nullptr, &d.device), "vkCreateDevice");
        vkldr::ResolveDevice(d.device);
        vkGetDeviceQueue(d.device, d.queueFamily, 0, &d.queue);
    }

    // The allocator.
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.instance = d.instance;
    aci.physicalDevice = d.physical;
    aci.device = d.device;
    aci.pVulkanFunctions = &functions;
    aci.flags = (d.memoryBudget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0) |
        (w12.bufferDeviceAddress ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0);
    Check(vmaCreateAllocator(&aci, &d.allocator), "vmaCreateAllocator");

    // The queue's timeline.
    VkSemaphoreTypeCreateInfo tci{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &tci;
    Check(vkCreateSemaphore(d.device, &sci, nullptr, &d.timeline), "vkCreateSemaphore (timeline)");
    SetName(d, VK_OBJECT_TYPE_SEMAPHORE, uint64_t(d.timeline), "regpu queue timeline");

    LoadPipelineCache(d, cacheDir);

    uint64_t local = 0;
    for (uint32_t i = 0; i < d.memory.memoryHeapCount; i++)
        if (d.memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            local = std::max<uint64_t>(local, d.memory.memoryHeaps[i].size);
    Log("presenter: Vulkan on %s (%s %s), %llu MB video memory, %s%s; conditional rendering %s, host memory import %s, push descriptors %s",
        d.name.c_str(), d.driver.driverName, d.driver.driverInfo, (unsigned long long)(local >> 20),
        d.uma ? "UMA (shared system memory)" : "discrete", d.largeBar ? ", host-visible VRAM (resizable BAR)" : "",
        d.conditionalRendering ? "yes" : "no", d.externalMemoryHost ? "yes" : "no", d.pushDescriptor ? "yes" : "no");
}

size_t SavePipelineCache(Device& d)
{
    if (!d.device || !d.pipelineCache || d.pipelineCachePath.empty())
        return 0;
    size_t bytes = 0;
    if (vkGetPipelineCacheData(d.device, d.pipelineCache, &bytes, nullptr) != VK_SUCCESS || !bytes)
        return 0;
    std::vector<uint8_t> data(bytes);
    if (vkGetPipelineCacheData(d.device, d.pipelineCache, &bytes, data.data()) != VK_SUCCESS)
        return 0;
    std::error_code ec;
    std::filesystem::create_directories(d.pipelineCachePath.parent_path(), ec);
    // Written beside, then renamed over: a crash mid-write leaves the old cache, not half a new one.
    const std::filesystem::path temp = d.pipelineCachePath.string() + ".tmp";
    FILE* f = fopen(temp.string().c_str(), "wb");
    if (!f)
        return 0;
    const bool ok = fwrite(data.data(), 1, bytes, f) == bytes;
    fclose(f);
    if (!ok)
        return 0;
    std::filesystem::rename(temp, d.pipelineCachePath, ec);
    return ec ? 0 : bytes;
}

uint64_t Submit(Device& d, const VkCommandBuffer* buffers, uint32_t count, const SemaphoreUse* waits, uint32_t waitCount,
    const SemaphoreUse* signals, uint32_t signalCount, VkFence fence)
{
    VkCommandBufferSubmitInfo cbs[16];
    std::vector<VkCommandBufferSubmitInfo> cbsMore;
    VkCommandBufferSubmitInfo* cb = cbs;
    if (count > 16)
    {
        cbsMore.resize(count);
        cb = cbsMore.data();
    }
    for (uint32_t i = 0; i < count; i++)
        cb[i] = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, buffers[i], 0 };
    std::vector<VkSemaphoreSubmitInfo> wait(waitCount), signal(signalCount + 1);
    for (uint32_t i = 0; i < waitCount; i++)
        wait[i] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, waits[i].semaphore, waits[i].value, waits[i].stages, 0 };
    for (uint32_t i = 0; i < signalCount; i++)
        signal[i] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, signals[i].semaphore, signals[i].value, signals[i].stages, 0 };
    VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.commandBufferInfoCount = count;
    si.pCommandBufferInfos = cb;
    si.waitSemaphoreInfoCount = waitCount;
    si.pWaitSemaphoreInfos = wait.data();
    std::lock_guard lock(d.queueLock);
    const uint64_t value = d.submitted.load(std::memory_order_relaxed) + 1;
    signal[signalCount] = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, d.timeline, value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0 };
    si.signalSemaphoreInfoCount = signalCount + 1;
    si.pSignalSemaphoreInfos = signal.data();
    Check(vkQueueSubmit2(d.queue, 1, &si, fence), "vkQueueSubmit2");
    d.submitted.store(value, std::memory_order_release);
    return value;
}

uint64_t Completed(const Device& d)
{
    uint64_t value = 0;
    vkGetSemaphoreCounterValue(d.device, d.timeline, &value);
    return value;
}

bool WaitFor(const Device& d, uint64_t value, uint64_t timeoutNs)
{
    VkSemaphoreWaitInfo wi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wi.semaphoreCount = 1;
    wi.pSemaphores = &d.timeline;
    wi.pValues = &value;
    const VkResult result = vkWaitSemaphores(d.device, &wi, timeoutNs);
    if (result < 0)
        Check(result, "vkWaitSemaphores");
    return result == VK_SUCCESS;
}
}
