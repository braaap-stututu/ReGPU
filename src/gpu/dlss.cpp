// NVIDIA DLSS (Super Resolution, and DLAA at native scale) through NGX on Vulkan. The renderer hands over the scene's
// colour, depth and motion vectors at render size and gets the upscaled image in the output image (see
// TemporalEvaluate in renderer.cpp). Needs nvngx_dlss.dll next to the executable (copied there by the build), an RTX
// GPU, and the device extensions NGX asks for (VK_NVX_binary_import, VK_NVX_image_view_handle: vk::Device::ngx).
// Built only with the DLSS SDK (REGPU_DLSS); its Vulkan form has not been compiled on the machine that converted it
// (no SDK there): check it builds before relying on it.
#include "gpu/dlss.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include "log.h"

namespace
{
struct Dlss
{
    bool tried = false, available = false;
    vk::Device* device = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    DlssSetup setup{};
} g_dlss;

// Made up for this port (NGX wants a GUID-shaped project id).
const char kProjectId[] = "6b3a9c2e-4f1d-4e8a-9b7c-2d5f8e1a3c47";

NVSDK_NGX_Resource_VK Resource(const vk::Image& image, bool readWrite)
{
    const VkImageAspectFlags aspect = image.aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    const VkImageSubresourceRange range{ aspect, 0, 1, 0, 1 };
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, range, image.desc.format, image.desc.width, image.desc.height, readWrite);
}
}

bool DlssInit(vk::Device& device)
{
    if (g_dlss.tried)
        return g_dlss.available;
    g_dlss.tried = true;
    g_dlss.device = &device;
    if (!device.ngx)
    {
        Log("dlss: the device lacks VK_NVX_binary_import / VK_NVX_image_view_handle (not an NVIDIA GPU or driver)");
        return false;
    }
    wchar_t path[260] = L".";
#if defined(_WIN32)
    GetTempPathW(260, path);
#endif
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", path, device.instance, device.physical,
        device.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
    if (NVSDK_NGX_FAILED(r))
    {
        Log("dlss: NGX init failed (%08X): not an NVIDIA RTX GPU or driver, or nvngx_dlss.dll missing", unsigned(r));
        return false;
    }
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_dlss.params);
    int supported = 0, needsUpdate = 0;
    if (NVSDK_NGX_SUCCEED(r))
    {
        NVSDK_NGX_Parameter_GetI(g_dlss.params, NVSDK_NGX_Parameter_SuperSampling_Available, &supported);
        NVSDK_NGX_Parameter_GetI(g_dlss.params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdate);
    }
    if (!supported)
    {
        Log("dlss: not available on this GPU%s", needsUpdate ? " (needs a newer driver)" : "");
        return false;
    }
    g_dlss.available = true;
    Log("dlss: ready");
    return true;
}

bool DlssEvaluate(VkCommandBuffer cb, const DlssSetup& setup, const DlssFrame& frame)
{
    if (!g_dlss.available)
        return false;
    // (Re)create the feature when its sizes, quality or HDR mode change.
    if (!g_dlss.feature || !(setup == g_dlss.setup))
    {
        if (g_dlss.feature)
        {
            NVSDK_NGX_VULKAN_ReleaseFeature(g_dlss.feature);
            g_dlss.feature = nullptr;
        }
        static const NVSDK_NGX_PerfQuality_Value qualities[] = { NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
            NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf, NVSDK_NGX_PerfQuality_Value_UltraPerformance };
        NVSDK_NGX_DLSS_Create_Params create{};
        create.Feature.InWidth = setup.renderWidth;
        create.Feature.InHeight = setup.renderHeight;
        create.Feature.InTargetWidth = setup.outputWidth;
        create.Feature.InTargetHeight = setup.outputHeight;
        create.Feature.InPerfQualityValue = qualities[setup.quality > 4 ? 1 : setup.quality];
        create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
            (setup.depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) | (setup.hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0);
        const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(g_dlss.device->device, cb, 1, 1, &g_dlss.feature, g_dlss.params, &create);
        if (NVSDK_NGX_FAILED(r))
        {
            Log("dlss: create failed (%08X) for %ux%u -> %ux%u", unsigned(r), setup.renderWidth, setup.renderHeight, setup.outputWidth,
                setup.outputHeight);
            g_dlss.available = false;
            return false;
        }
        g_dlss.setup = setup;
        Log("dlss: %ux%u -> %ux%u, quality %u%s", setup.renderWidth, setup.renderHeight, setup.outputWidth, setup.outputHeight, setup.quality,
            setup.hdr ? ", HDR" : "");
    }
    NVSDK_NGX_Resource_VK color = Resource(*frame.color, false), output = Resource(*frame.output, true), depth = Resource(*frame.depth, false),
                          motion = Resource(*frame.motion, false);
    NVSDK_NGX_VK_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = &color;
    eval.Feature.pInOutput = &output;
    eval.pInDepth = &depth;
    eval.pInMotionVectors = &motion;
    eval.InJitterOffsetX = frame.jitterX;
    eval.InJitterOffsetY = frame.jitterY;
    eval.InRenderSubrectDimensions = { setup.renderWidth, setup.renderHeight };
    eval.InReset = frame.reset ? 1 : 0;
    eval.InMVScaleX = frame.motionScaleX;
    eval.InMVScaleY = frame.motionScaleY;
    eval.InFrameTimeDeltaInMsec = frame.frameTimeMs;
    const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, g_dlss.feature, g_dlss.params, &eval);
    if (NVSDK_NGX_FAILED(r))
    {
        static int logged = 0;
        if (logged++ < 3)
            Log("dlss: evaluate failed (%08X)", unsigned(r));
        return false;
    }
    return true;
}
