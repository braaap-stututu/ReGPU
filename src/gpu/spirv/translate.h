#pragma once
// EXPERIMENTAL: Xenos microcode straight to SPIR-V (no HLSL, no DXC), for the same interface the HLSL path gives a
// shader once vk::AdaptHlsl and DXC are through with it (gpu/vk/bindings.h: set 0 binding 0/1 the constant blocks;
// interpolators o[16] at locations 0-15, the point coordinate at 16; entry point "main"). It mirrors hlsl.cpp's
// Emitter for the subset it supports and returns false with a short reason for the rest, so a renderer can fall
// back to the HLSL path per shader. Not wired into the renderer.
//
// Supported: pixel shaders made of ALU instructions and texture fetches (1D/2D/3D/cube as TFetch, setTexLOD,
// getGradients, getWeights) (every vector op but CUBE, every scalar op hlsl.cpp knows),
// predicates, kills, float/bool/loop constants (relative addressing, packed constants), all control flow (exec,
// conditional exec, jumps, loops, calls), colour and depth exports and the HLSL epilogue (alpha test, alpha to mask,
// exponent bias, target clamps). Not yet: vertex shaders (the HLSL wrapper: index fetch, rectangles, points,
// viewport), vertex fetches, the REGPU_PLAIN_MUL_* / REGPU_HYPER_PLAIN_MUL / REGPU_HALF_REGS variants.
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/hlsl.h"
#include "gpu/ucode.h"

// The renderer's pixel shader variants, made from the microcode as the HLSL path makes them from its text:
// - noAlphaTest: no alpha test and no alpha to mask (NoKillVariant);
// - plainFetchMask: 2D fetches of these slots (pixel, computed LOD, no register LOD) sample with the constant's sampler and
//   nothing else (FetchVariant's TFetchPlain: the slots' constants are plain this draw);
// - specs: these slots' fetch-constant words tinfo.x and tinfo.w as literals (SpecVariant's TFetch2DK), so each fetch
//   compiles to what its constant asks for.
struct SpirvFetchSpec
{
    uint32_t slot, info, w;
};
struct SpirvOptions
{
    bool noAlphaTest = false;
    uint32_t plainFetchMask = 0;
    std::vector<SpirvFetchSpec> specs;
    // A 2D fetch's sub-image scale (tbias.z) behind a branch, as the HLSL has it: each fetch its own block, so the driver
    // doesn't issue a shader's samples together (faster where vector registers are plenty: the RX 7900 XTX's 4K race
    // +4%; on RDNA2 graphics their registers cost occupancy: the 9800X3D's ~4% slower). The renderer's choice per GPU.
    bool fetchBranches = false;
    // The motion-vector variant for temporal upscalers (renderer.cpp MotionVariant, the HLSL path's): the vertex shader
    // also runs the guest program with last frame's constants (vcPrev, set 0 binding 2) and outputs this and last
    // frame's clip positions (locations 17, 18; the drawn position jittered by vjitter, the VS block's member 13); the
    // pixel shader writes the motion (location 1, RG) and the reactive mask (location 2, R). Combines with the other
    // options, so a motion draw keeps its alpha-test-free and fetch variants.
    bool motion = false;
    // A vertex shader's draw words as literals (renderer.cpp's REGPU_VS_SPECIALIZE), so its wrapper compiles to the path
    // the draw takes: vte (PA_CL_VTE_CNTL with ReGPU's flags: the viewport transform), vteFlags (the index source: direct,
    // strip, 32-bit, auto, quads, endian), vinfoFlags (points), rects (false: no rectangle-list corners), and each vertex
    // fetch constant's endian swap and buffer (gmem / gphys). Addresses, offsets, sizes and scales stay uniforms.
    struct VertexSpec
    {
        uint32_t fetch, endian, phys;
    };
    bool vsSpec = false;
    uint32_t vte = 0, vteFlags = 0, vinfoFlags = 0;
    bool rects = true;
    std::vector<VertexSpec> vertexSpecs;
};

// hlslInfo: the HLSL path's result for the same shader (TranslateShader, after the renderer's PackConstants): its
// constant mask and packedConstants decide the constant numbering.
bool TranslateSpirv(const ucode::Shader& shader, const TranslatedShader& hlslInfo, std::vector<uint32_t>& words, std::string& why,
    const SpirvOptions& options = {});
