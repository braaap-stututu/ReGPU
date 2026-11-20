#pragma once
// VR through OpenXR (REGPU_VR; built with REGPU_OPENXR_DIR). Step 1, "screen": the presented picture on a virtual
// screen in the headset, redrawn at the headset's rate whatever the game's frame rate; the desktop window stays as it
// is. The OpenXR loader (openxr_loader.dll beside the exe) is loaded only when VR is asked for.
#include <cstdint>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

namespace regpu::xr
{
// REGPU_VR=screen (start-time): VR wanted.
bool Wanted();
// Starts the headset thread on ReGPU's device and queue (after they exist). It waits for a headset and keeps
// retrying, so the game runs normally without one.
void Start(ID3D12Device* device, ID3D12CommandQueue* queue);
// The back buffer holds the finished frame (state COPY_DEST, left so): copies it for the headset. No-op unless a
// headset session runs.
void CaptureFrame(ID3D12GraphicsCommandList* list, ID3D12Resource* backBuffer);
}
