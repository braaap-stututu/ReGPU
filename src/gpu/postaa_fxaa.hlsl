// FXAA 3.11 (NVIDIA, Timothy Lottes; the copy in Intel's CMAA 2 sample, BSD-style licence in the header) as a compute
// pass over postaa.hlsl's compressed image (luma in alpha). Quality preset 39 (the highest; 12 = fastest).
// Fxaa3_11.h comes from the host's REGPU_AA_DIR/fxaa.
#define FXAA_PC 1
#define FXAA_HLSL_5 1
#define FXAA_GREEN_AS_LUMA 0
#define FXAA_QUALITY__PRESET 39
#include "Fxaa3_11.h"

cbuffer C : register(b0)
{
    uint4 size;
    float4 rt;  // 1/w, 1/h, w, h
    float4 p0;  // x subpixel amount, y edge threshold, z edge threshold minimum
    float4 p1;
};
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState linearClamp : register(s0);

[numthreads(8, 8, 1)] void FxaaMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    FxaaTex tex;
    tex.smpl = linearClamp;
    tex.tex = src;
    const float2 uv = (float2(id) + 0.5) * rt.xy;
    dst[id] = FxaaPixelShader(uv, 0, tex, tex, tex, tex, rt.xy, 0, 0, 0, p0.x, p0.y, p0.z, 0, 0, 0, 0);
}
