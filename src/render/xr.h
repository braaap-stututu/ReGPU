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

// Step 3, true stereo (REGPU_VR=stereo, gpu.cpp stereo::): the frame's main view drawn twice, once per eye.
// StereoFrame: this frame is to be drawn in stereo (a head-tracked frame, from SubmitFramePose; or REGPU_VR_STEREO_TEST
// without a headset); `clipPerMetre` = P00 of the frame's projection (clip x per metre of sideways offset at w 1),
// `halfIpd` = half the eye distance (metres).
bool StereoFrame(float& clipPerMetre, float& halfIpd);
// The headset's image for eye 0 (left) or 1 (right) at w x h, state COMMON (to be left so), or null without a session.
// (eye 2: the HUD panel's image, premultiplied alpha; StereoHudCaptured once it is written this frame.)
ID3D12Resource* StereoTarget(int eye, uint32_t w, uint32_t h);
void StereoHudCaptured();
}
