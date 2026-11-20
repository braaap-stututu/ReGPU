// Builds without the DLSS, XeSS or FidelityFX SDK (REGPU_DLSS / REGPU_XESS_DIR / REGPU_FSR3 off): those upscalers report
// unavailable and the renderer falls back as it does on hardware without them.
#include "gpu/fsr3.h"

#include "gpu/xess.h"
#ifndef REGPU_XESS
bool XessInit(vk::Device&) { return false; }
bool XessEvaluate(VkCommandBuffer, const DlssSetup&, const DlssFrame&) { return false; }
#endif
#ifndef REGPU_DLSS
bool DlssInit(vk::Device&) { return false; }
bool DlssEvaluate(VkCommandBuffer, const DlssSetup&, const DlssFrame&) { return false; }
#endif
#ifndef REGPU_FSR3
bool Fsr3Evaluate(vk::Device&, VkCommandBuffer, const DlssSetup&, const DlssFrame&, float) { return false; }
bool Fsr4Evaluate(vk::Device&, VkCommandBuffer, const DlssSetup&, const DlssFrame&, float) { return false; }
bool Fsr3Prepare(vk::Device&, const DlssSetup&, void**, bool*) { return false; }
bool Fsr3Dispatch(VkCommandBuffer, void*, bool, const DlssSetup&, const DlssFrame&, float) { return false; }
#endif
