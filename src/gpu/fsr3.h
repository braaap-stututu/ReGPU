#pragma once
#include "gpu/dlss.h"

// AMD FSR 3 upscaling (fsr3.cpp), with the inputs and states DLSS takes (dlss.h). Records into
// `list` (the real command list). False if FidelityFX is unavailable.
bool Fsr3Evaluate(ID3D12Device* device, ID3D12GraphicsCommandList* list, const DlssSetup& setup, const DlssFrame& frame, float sharpness);
// The same in two steps (FM3 port, session 36): Prepare on the recording thread (the context; false if FSR is
// unavailable), Dispatch where the list is replayed (RecordedList::Call), so recording needn't wait for the replay.
bool Fsr3Prepare(ID3D12Device* device, const DlssSetup& setup, void** handle, bool* switched);
bool Fsr3Dispatch(ID3D12GraphicsCommandList* list, void* handle, bool switched, const DlssSetup& setup, const DlssFrame& frame, float sharpness);
