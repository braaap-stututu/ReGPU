#pragma once
#include <cstdint>
#include <vector>

#include "gpu/vk/vk_loader.h"

// Window + Vulkan swap chain (REGPU_HEADLESS=1 disables it).
void PresenterInit();
// Host::presentImage is set and REGPU_NO_HOST_PRESENT is not (decided in PresenterInit): the renderer
// hands its pictures to the host instead of PresenterRecordCopy / PresenterPresent.
bool PresenterHostPresents();
// The size the game renders at (settings.h: resolution and forced aspect ratio). The swap
// chain is the resolution; the image is centred in it and the swap chain stretches to the
// window.
void RenderSize(uint32_t& width, uint32_t& height);
// The primary display's refresh rate (the frame cap with vsync on).
uint32_t DisplayRefreshHz();
// Called from VdSwap: the guest finished a frame.
void PresenterSubmitFrame(uint64_t frame);
// Shows a CPU-side RGBA8 image (scaled 1:1 into the 1280x720 back buffer).
void PresenterSubmitImage(uint32_t width, uint32_t height, std::vector<uint8_t> rgba);
// The Vulkan backend (gpu/vk): an image of any size, returning the previous one's storage; its own window and frames.
std::vector<uint8_t> PresenterExchangeImage(uint32_t width, uint32_t height, std::vector<uint8_t> rgba);
void* PresenterClaimWindow();
void PresenterReleaseWindow();
void PresenterCountFrame();
// Keyboard input goes to the game only while the window has focus.
bool PresenterHasFocus();

namespace vk
{
struct Device;
struct Image;
}
// The Vulkan device (gpu/vk/device.h), created on first use, also when headless; the host's when Config::hostDevice and
// the rest are set.
vk::Device& RenderVkDevice();
// Records copying src (TRANSFER_SRC_OPTIMAL, RGBA8 at RenderSize()) into the next swap chain image (acquired here) and
// leaves that image ready to present; returns false when there is no window. Submit the buffer with PresenterSubmit
// (it waits for the acquire), then call PresenterPresent().
// overlay (TRANSFER_SRC_OPTIMAL RGBA8, overlayW x overlayH) is copied over the top-left corner when given.
// timer (same kind, timerW x timerH) goes in the very corner, the overlay below it.
// fps (same kind, fpsW x fpsH: the fps panel) goes at the top centre.
// The picture sits on the output-size canvas, which is scaled to the swap chain (the window's size).
// grade (FM3 port, session 33): instead of the copy, grade(cb, view, format, width, height, gradeCtx) draws the picture
// into a colour attachment (COLOR_ATTACHMENT_OPTIMAL; the display gamma pass) when src is the canvas's size; the
// back buffer itself when it is that size too.
using PresenterGrade = void (*)(VkCommandBuffer cb, VkImageView target, VkFormat format, uint32_t width, uint32_t height, void* ctx);
bool PresenterRecordCopy(VkCommandBuffer cb, const vk::Image* src, const vk::Image* overlay = nullptr, uint32_t overlayW = 0, uint32_t overlayH = 0,
    const vk::Image* timer = nullptr, uint32_t timerW = 0, uint32_t timerH = 0, const vk::Image* fps = nullptr, uint32_t fpsW = 0, uint32_t fpsH = 0,
    PresenterGrade grade = nullptr, void* gradeCtx = nullptr);
// Submits the (ended) command buffer that holds PresenterRecordCopy's copy: it waits for the swap chain image and
// signals the present. Any other buffer goes as a plain submission. Returns the device timeline value.
void PresenterSubmit(VkCommandBuffer cb);
uint64_t PresenterSubmitValue(VkCommandBuffer cb);
void PresenterPresent();
// The swap chain to a new output size (host GPU idle; renderer.cpp ApplyOutputSize).
void PresenterResize(uint32_t width, uint32_t height);
// Called by the renderer: the presenter stops drawing its own frames.
void PresenterUseRenderer();
