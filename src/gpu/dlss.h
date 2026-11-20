#pragma once
#include <cstdint>

#include "gpu/vk/device.h"
#include "gpu/vk/resources.h"

// The temporal upscalers' common inputs (DLSS in dlss.cpp, FSR 3/4 in fsr3.cpp, XeSS in xess.cpp), on Vulkan.
struct DlssSetup
{
    uint32_t renderWidth = 0, renderHeight = 0, outputWidth = 0, outputHeight = 0;
    uint32_t quality = 1; // settings.h: 0 native (DLAA) .. 4 ultra performance
    bool hdr = false, depthInverted = true;
    bool fsr4 = false; // FSR: the 4.x version where the device offers it (fsr3.cpp)
    bool operator==(const DlssSetup&) const = default;
};

struct DlssFrame
{
    // Inputs in SHADER_READ_ONLY_OPTIMAL (sampled), the output in GENERAL (a storage image). Each image's sampled view
    // is used (vk::Image::view).
    const vk::Image *color, *depth, *motion, *output;
    // FSR 3/4 only: the reactive mask (R8, sampled like the other inputs), or null. Where it is set the upscaler trusts
    // this frame's colour over its history (smoke, sparks, particles).
    const vk::Image* reactive = nullptr;
    float jitterX, jitterY;           // render pixels
    float motionScaleX, motionScaleY; // motion texture values to render pixels
    float frameTimeMs;
    bool reset;
};

// False without an RTX GPU, driver support, nvngx_dlss.dll, or the device extensions NGX needs (vk::Device::ngx).
bool DlssInit(vk::Device& device);
// Records the upscale into `cb` (the real command buffer, outside rendering).
bool DlssEvaluate(VkCommandBuffer cb, const DlssSetup& setup, const DlssFrame& frame);
