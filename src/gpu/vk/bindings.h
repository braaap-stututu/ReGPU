#pragma once
// What the translated shaders bind (docs/vulkan-conversion-plan.md, step 4), replacing the D3D12 root signature, the
// shader-visible descriptor heap and the sampler heap. DXC maps the HLSL registers (hlsl.cpp's preamble, after
// vk::AdaptHlsl) with -fvk-b-shift 0, -fvk-t-shift 100, -fvk-s-shift 200 for space 0, and register spaces to sets:
//
//   set 0, per draw stream (bound once per frame slot and sampler bank; dynamic offsets per draw):
//     binding 0    VSConstants (b0)          uniform buffer, dynamic: a range of the frame's upload ring
//     binding 1    PSConstants (b1)          uniform buffer, dynamic
//     binding 2    motion constants (b2)     uniform buffer, dynamic (motion-vector pipelines; unused by the others)
//     binding 100  gmem (t0)                 storage buffer: the frame's upload ring (vertex data copied this frame)
//     binding 101  gphys (t1)                storage buffer: guest physical memory (512 MB)
//     binding 200  gSamplers[8] (s0-s7)      samplers: linear wrap, linear clamp, point clamp, point wrap, and linear wrap
//                                            at 2x/4x/8x/16x anisotropy (renderer.cpp's Init and WriteSamplerBank)
//   set 1  gtex[65536]   (t0, space1)        sampled images, 2D: update-after-bind, partially bound
//   set 2  gcube[65536]  (t0, space2)        sampled images, cube
//   set 3  gvol[65536]   (t0, space3)        sampled images, 3D
//
// Sets 1-3 are written as textures come and go (the D3D12 renderer's BindlessIndexLocked) and bound once per command
// buffer: draws rebind set 0 only (Tankyx's FM2 notes: rebinding all sets per draw cost time on the render thread).
#include <cstdint>
#include <mutex>
#include <vector>

#include "gpu/vk/device.h"

namespace vk
{
constexpr uint32_t kBindlessCount = 1u << 16; // renderer.cpp's kBindless; vk::AdaptHlsl sizes the arrays to it
constexpr uint32_t kSamplerCount = 8;

enum class TextureKind : uint32_t
{
    Tex2D = 0,
    Cube = 1,
    Tex3D = 2,
};

struct DrawBindings
{
    VkDescriptorSetLayout set0 = VK_NULL_HANDLE;
    VkDescriptorSetLayout textures[3] = {};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;        // set 0s
    VkDescriptorPool texturePool = VK_NULL_HANDLE; // sets 1-3 (update-after-bind)
    VkDescriptorSet textureSets[3] = {};
    // The uniform ranges of the dynamic bindings, in bytes (sizeof VSConstants, PSConstants, the motion block).
    uint32_t constantBytes[3] = {};

    // Pending bindless writes (WriteTexture), applied by FlushTextureWrites before the next submission.
    std::mutex writeMutex;
    struct PendingWrite
    {
        TextureKind kind;
        uint32_t index;
        VkImageView view;
    };
    std::vector<PendingWrite> pending;
};

// Creates the layouts, the pipeline layout and the bindless sets. vsBytes / psBytes / motionBytes: the constant blocks'
// sizes. False (and a log line) when the device can't hold 65536 update-after-bind images per stage.
bool CreateDrawBindings(Device& d, DrawBindings& b, uint32_t vsBytes, uint32_t psBytes, uint32_t motionBytes);

// A set 0 over one frame slot's upload ring, guest physical memory and a bank of eight samplers. motionRing: b2 (the
// motion draws' last-frame vertex constants) over that buffer instead of `ring` (the previous frame slot's ring).
VkDescriptorSet CreateSet0(Device& d, DrawBindings& b, VkBuffer ring, VkDeviceSize ringBytes, VkBuffer guest, VkDeviceSize guestBytes,
    const VkSampler (&samplers)[kSamplerCount], VkBuffer motionRing = VK_NULL_HANDLE, VkDeviceSize motionRingBytes = 0);
// Frees a set 0 (its ring grew: a new one replaces it once the frames using it are done).
void FreeSet0(Device& d, DrawBindings& b, VkDescriptorSet set);

// Writes a bindless entry (VK_NULL_HANDLE: the null texture, reads return 0). Thread-safe; applied by
// FlushTextureWrites, which the renderer calls before each submission (as the D3D12 descriptor copies were visible at
// once). Update-after-bind lets entries change while the sets are bound in recorded buffers, as long as the entries
// those buffers use stay valid until they finish (the renderer frees an index only after its last frame completed).
void WriteTexture(DrawBindings& b, TextureKind kind, uint32_t index, VkImageView view);
void FlushTextureWrites(Device& d, DrawBindings& b);

// The eight samplers (s0 linear wrap, s1 linear clamp, s2 point clamp, s3 point wrap, s4-s7 linear wrap), with each one's
// anisotropy: 0 or 1 trilinear, 2-16 anisotropic (capped at the device's limit). renderer.cpp works the values out from
// REGPU_NO_ANISO, REGPU_ANISO_FROM_FETCH and regpu::SetAnisotropy, as its D3D12 Init and WriteSamplerBank did.
void CreateSamplers(Device& d, const int (&anisotropy)[kSamplerCount], VkSampler (&out)[kSamplerCount]);
}
