// SMAA 1x (Jimenez, Echevarria, Sousa, Gutierrez; MIT licence) as three compute passes over postaa.hlsl's compressed
// image: edges (colour), blending weights (area and search textures), neighbourhood blending. SMAA.hlsl comes from
// the host's REGPU_AA_DIR/smaa. Its HLSL 4 port uses Sample (implicit derivatives), which compute shaders can't, so the
// porting macros are ours: every fetch is SampleLevel 0. `discard` (edge pass, no edge) becomes a no-op: the pass then
// writes zero edges, which is what a cleared edge texture holds.
cbuffer C : register(b0)
{
    uint4 size;
    float4 rt;  // 1/w, 1/h, w, h
    float4 p0;
    float4 p1;
};
#define SMAA_RT_METRICS rt
#define SMAA_PRESET_ULTRA 1
#define SMAA_CUSTOM_SL 1
SamplerState LinearSampler : register(s0);
SamplerState PointSampler : register(s1);
#define SMAATexture2D(tex) Texture2D tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex.SampleLevel(LinearSampler, coord, 0)
#define SMAASampleLevelZeroPoint(tex, coord) tex.SampleLevel(PointSampler, coord, 0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) tex.SampleLevel(LinearSampler, coord, 0, offset)
#define SMAASample(tex, coord) tex.SampleLevel(LinearSampler, coord, 0)
#define SMAASamplePoint(tex, coord) tex.SampleLevel(PointSampler, coord, 0)
#define SMAASampleOffset(tex, coord, offset) tex.SampleLevel(LinearSampler, coord, 0, offset)
#define SMAAGather(tex, coord) tex.Gather(LinearSampler, coord, 0)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]
#define discard
#include "SMAA.hlsl"
#undef discard

Texture2D<float4> t0 : register(t0); // colour (edges, blending) / edges (weights)
Texture2D<float4> t1 : register(t1); // area texture (weights) / blending weights (blending)
Texture2D<float4> t2 : register(t2); // search texture (weights)
RWTexture2D<float4> u0 : register(u0);

[numthreads(8, 8, 1)] void SmaaEdgesMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const float2 uv = (float2(id) + 0.5) * rt.xy;
    float4 offset[3];
    SMAAEdgeDetectionVS(uv, offset);
    u0[id] = float4(SMAAColorEdgeDetectionPS(uv, offset, t0), 0, 0);
}

[numthreads(8, 8, 1)] void SmaaWeightsMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const float2 uv = (float2(id) + 0.5) * rt.xy;
    float2 pixcoord;
    float4 offset[3];
    SMAABlendingWeightCalculationVS(uv, pixcoord, offset);
    u0[id] = SMAABlendingWeightCalculationPS(uv, pixcoord, offset, t0, t1, t2, float4(0, 0, 0, 0));
}

[numthreads(8, 8, 1)] void SmaaBlendMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const float2 uv = (float2(id) + 0.5) * rt.xy;
    float4 offset;
    SMAANeighborhoodBlendingVS(uv, offset);
    u0[id] = SMAANeighborhoodBlendingPS(uv, offset, t0, t1);
}
