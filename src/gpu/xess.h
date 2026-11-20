#pragma once
// Intel XeSS Super Resolution (xess.cpp): the same inputs and contract as DLSS (dlss.h).
#include "gpu/dlss.h"

// False without libxess.dll beside the executable, or a device without what XeSS needs.
bool XessInit(vk::Device& device);
// Records the upscale into `cb` (the real command buffer).
bool XessEvaluate(VkCommandBuffer cb, const DlssSetup& setup, const DlssFrame& frame);
