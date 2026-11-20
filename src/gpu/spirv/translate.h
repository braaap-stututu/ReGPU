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

// hlslInfo: the HLSL path's result for the same shader (TranslateShader, after the renderer's PackConstants): its
// constant mask and packedConstants decide the constant numbering.
bool TranslateSpirv(const ucode::Shader& shader, const TranslatedShader& hlslInfo, std::vector<uint32_t>& words, std::string& why);
