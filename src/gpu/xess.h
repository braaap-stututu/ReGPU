#pragma once
// Intel XeSS Super Resolution (xess.cpp): the same inputs and contract as DLSS (dlss.h).
#include "gpu/dlss.h"

// False without libxess.dll beside the executable or a GPU with Shader Model 6.4.
bool XessInit(ID3D12Device* device);
// Records the upscale into `list` (the real command list).
bool XessEvaluate(ID3D12GraphicsCommandList* list, const DlssSetup& setup, const DlssFrame& frame);
