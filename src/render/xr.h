#pragma once
// VR through OpenXR (REGPU_VR; built with REGPU_OPENXR_DIR). Step 1, "screen": the presented picture on a virtual
// screen in the headset, redrawn at the headset's rate whatever the game's frame rate; the desktop window stays as it
// is. The OpenXR loader (openxr_loader.dll beside the exe) is loaded only when VR is asked for.
#include <cstdint>

#include "gpu/vk/vk_loader.h"

namespace vk
{
struct Device;
struct Image;
}

namespace regpu::xr
{
// REGPU_VR=screen (start-time): VR wanted.
bool Wanted();
// Starts the headset thread on ReGPU's device and queue (after they exist). It waits for a headset and keeps
// retrying, so the game runs normally without one.
void Start(vk::Device& device);
// The back buffer holds the finished frame (TRANSFER_DST_OPTIMAL, left so): copies it for the headset. No-op unless a
// headset session runs.
void CaptureFrame(VkCommandBuffer cb, VkImage backBuffer, uint32_t width, uint32_t height, VkFormat format);

// Step 3, true stereo (REGPU_VR=stereo, gpu.cpp stereo::): the frame's main view drawn twice, once per eye.
// StereoFrame: this frame is to be drawn in stereo (a head-tracked frame, from SubmitFramePose; or REGPU_VR_STEREO_TEST
// without a headset); `clipPerMetre` = P00 of the frame's projection (clip x per metre of sideways offset at w 1),
// `halfIpd` = half the eye distance (metres).
bool StereoFrame(float& clipPerMetre, float& halfIpd);
// The headset's image for eye 0 (left) or 1 (right) at w x h (a storage image; its state tracked as ReGPU's own), or
// null without a session. (eye 2: the HUD panel's image, premultiplied alpha; StereoHudCaptured once it is written
// this frame.)
vk::Image* StereoTarget(int eye, uint32_t w, uint32_t h);
void StereoHudCaptured();
// Head-tracked mono frames (REGPU_VR=cockpit) take the HUD apart too (the same capture as stereo: renderer.cpp
// StereoHudDraw): true on a posed frame while a headset session runs (or REGPU_VR_STEREO_TEST).
bool MonoHudPanel();
// The frame whose copy is recorded next (CaptureFrame): its pose tag (regpu::vr::SubmitTaggedPose, low 16 bits; 0 =
// none) | kFrameStereo | kFrameHud. Set by the renderer right before the frame's back-buffer copy, on the thread that
// records it (the asynchronous frame end's: it can run a frame behind the draw thread).
constexpr uint32_t kFrameStereo = 1u << 16, kFrameHud = 1u << 17;
void SetCaptureTag(uint32_t frame);
// On the draw thread at the swap: whether this frame's eyes and HUD panel image were written (kFrameStereo, kFrameHud),
// cleared for the next frame. (They used to be read at the copy, which a late frame end let the next frame take first:
// session 39, the view went flat for a moment at hitches.)
uint32_t TakeFrameFlags();
// Whether `tag` is a pose handed in during the last 250 ms (the renderer takes only those from the constants: another
// value with the same bit pattern is not a tag).
bool TagKnown(uint32_t tag);
// Headset pacing (REGPU_VR_PACE, default on in cockpit/stereo; 0 off): the frame cap the headset wants (its display
// rate x REGPU_VR_PACE_MULT, default 1), or 0 (no session, pacing off).
uint32_t PaceCap();
}
