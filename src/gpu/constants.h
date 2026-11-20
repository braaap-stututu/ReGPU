#pragma once
// The constant blocks the translated shaders read, byte for byte (hlsl.cpp's cbuffer VSConstants and
// PSConstants, and renderer.cpp's VSConstants / PSConstants, which the Direct3D 12 path fills). The guest's
// constant banks are big-endian and the shaders index them as float4 arrays, so the layout is part of the
// shader contract. The Vulkan backend (gpu/vk) uploads exactly this.
#include <cstdint>

namespace gpu
{
// cbuffer VSConstants (register b0).
struct VSConstants
{
    float c[256][4];
    uint32_t b[8];        // 256 bool constants
    uint32_t l[32];       // loop constants: count | start << 8 | step << 16
    uint32_t vf[96][4];   // vertex fetch: x byte offset (gmem, or gphys if z), y endian, w size in dwords
    uint32_t info[4];     // x index list byte offset, y position mode | points, z/w target size
    float point[4];       // xy constant point diameter, z/w min/max
    float vscale[4], voffset[4]; // PA_CL_VPORT_*SCALE / *OFFSET
    uint32_t vte[4];      // PA_CL_VTE_CNTL
    uint32_t vfl[32][4];  // vertex fetch layout of the bound shader
    float vuser[4][4];    // the game's per-draw constants (Host::onDraw; hlsl.cpp vuser), zero by default
    float vdepth[4];      // x: the polygon offset's constant part (hlsl.cpp vdepth)
    float vwide[4];       // x: Hor+ factor for wide outputs
    float vjitter[4];     // motion-vector variants: this frame's jitter (NDC)
};

// cbuffer PSConstants (register b1).
struct PSConstants
{
    float c[256][4];
    uint32_t b[8];
    uint32_t l[32];        // loop constants
    uint32_t info[4];      // x alpha test function, y generated-parameter register, z flags, w colour exponent bias
    float info2[4];        // x alpha reference, y/z 1 / render scale, w colour ceiling
    uint32_t tinfo[32][4]; // x sampler, y/z size, w dimension (3 = cube)
    float tbias[32][4];    // x LOD bias, y (as uint) bindless descriptor index
};

static_assert(sizeof(VSConstants) == 256 * 16 + 8 * 4 + 32 * 4 + 96 * 16 + 16 + 16 + 2 * 16 + 16 + 32 * 16 + 64 + 3 * 16,
    "VSConstants must match cbuffer VSConstants in hlsl.cpp");
static_assert(sizeof(PSConstants) == 256 * 16 + 8 * 4 + 32 * 4 + 16 + 16 + 32 * 16 + 32 * 16,
    "PSConstants must match cbuffer PSConstants in hlsl.cpp");
}
