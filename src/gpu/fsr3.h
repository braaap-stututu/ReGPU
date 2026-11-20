#pragma once
#include "gpu/dlss.h"

// AMD FSR 3 upscaling (fsr3.cpp), with the inputs and layouts DLSS takes (dlss.h). Records into `cb` (the real command
// buffer). False if FidelityFX is unavailable.
bool Fsr3Evaluate(vk::Device& device, VkCommandBuffer cb, const DlssSetup& setup, const DlssFrame& frame, float sharpness);
// The same with the FidelityFX ML upscaler (FSR 4) where this GPU offers it; otherwise FSR 3.
bool Fsr4Evaluate(vk::Device& device, VkCommandBuffer cb, const DlssSetup& setup, const DlssFrame& frame, float sharpness);
// The same in two steps (FM3 port, session 36): Prepare on the recording thread (the context; false if FSR is
// unavailable), Dispatch where the buffer is replayed (vk::RecordedList::Call), so recording needn't wait for the replay.
bool Fsr3Prepare(vk::Device& device, const DlssSetup& setup, void** handle, bool* switched);
bool Fsr3Dispatch(VkCommandBuffer cb, void* handle, bool switched, const DlssSetup& setup, const DlssFrame& frame, float sharpness);
