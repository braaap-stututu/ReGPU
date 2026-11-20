// Anti-aliasing passes of ReGPU's own (FM3 port, session 36): the tone-map wrap the third-party filters run inside
// (postaa_fxaa/smaa/cmaa2.hlsl), the temporal resolve (TAA, on the motion vectors and jitter the temporal upscalers
// use), a copy, and the supersampling downsample. Compiled by tools/embed_hlsl.py into postaa_cs.h.
//
// The scene image is linear HDR (16-bit float; the console's 7e3 range goes to ~32): FXAA, SMAA and CMAA 2 expect a
// display-range picture. PrepMain compresses it reversibly (Reinhard on the largest channel, then a 2.0 gamma) into
// `work` with its luma in alpha; FinishMain expands the filtered result and writes back only the pixels the filter
// changed, so the rest of the image stays bit-identical.

cbuffer C : register(b0)
{
    uint4 size;   // xy: image size; zw: SSAA output size
    float4 rt;    // 1/w, 1/h, w, h
    float4 p0;    // TAA: x blend weight of the new frame, y 1 = no history, z clip strength; SSAA: x factor
    float4 p1;
};
Texture2D<float4> t0 : register(t0);
Texture2D<float4> t1 : register(t1);
Texture2D<float2> t2 : register(t2);
Texture2D<float> t3 : register(t3);
RWTexture2D<float4> u0 : register(u0);
RWTexture2D<float4> u1 : register(u1);
SamplerState linearClamp : register(s0);
SamplerState pointClamp : register(s1);

float3 Compress(float3 c)
{
    c = max(c, 0.0);
    return sqrt(c / (1.0 + max(c.r, max(c.g, c.b))));
}
float3 Expand(float3 g)
{
    const float3 t = g * g;
    return t / max(1.0 - max(t.r, max(t.g, t.b)), 1.0 / 4096.0);
}
float Luma(float3 g) { return dot(g, float3(0.299, 0.587, 0.114)); }

// t0 scene -> u0 work (compressed, luma in alpha).
[numthreads(8, 8, 1)] void PrepMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const float3 g = Compress(t0.Load(int3(id, 0)).rgb);
    u0[id] = float4(g, Luma(g));
}

// t0 work before, t1 work after the filter -> u0 scene (only where the filter changed the pixel; alpha kept).
[numthreads(8, 8, 1)] void FinishMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const float3 before = t0.Load(int3(id, 0)).rgb, after = t1.Load(int3(id, 0)).rgb;
    if (all(abs(after - before) < 1.0 / 2048.0))
        return;
    const float4 old = u0[id];
    u0[id] = float4(Expand(after), old.a);
}

// t0 -> u0, same size.
[numthreads(8, 8, 1)] void CopyMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    u0[id] = t0.Load(int3(id, 0));
}

// ---- TAA ----------------------------------------------------------------------------------------------------------
// t0 scene (this frame, jittered), t1 history (last output, same size), t2 motion (UV offset to last frame, top left of
// an output-sized target), u0 new history. Variance clipping in YCoCg of the compressed colour, the motion of the
// longest vector in the 3x3 (edges of moving objects take their motion), a 5-tap Catmull-Rom history fetch.
float3 ToYCoCg(float3 c) { return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25))); }
float3 FromYCoCg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

float3 HistoryCatmullRom(float2 uv)
{
    const float2 pos = uv * rt.zw;
    const float2 c1 = floor(pos - 0.5) + 0.5;
    const float2 f = pos - c1;
    const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);
    const float2 w12 = w1 + w2;
    const float2 tc0 = (c1 - 1.0) * rt.xy, tc3 = (c1 + 2.0) * rt.xy, tc12 = (c1 + w2 / w12) * rt.xy;
    float3 r = t1.SampleLevel(linearClamp, float2(tc12.x, tc0.y), 0).rgb * (w12.x * w0.y);
    r += t1.SampleLevel(linearClamp, float2(tc0.x, tc12.y), 0).rgb * (w0.x * w12.y);
    r += t1.SampleLevel(linearClamp, float2(tc12.x, tc12.y), 0).rgb * (w12.x * w12.y);
    r += t1.SampleLevel(linearClamp, float2(tc3.x, tc12.y), 0).rgb * (w3.x * w12.y);
    r += t1.SampleLevel(linearClamp, float2(tc12.x, tc3.y), 0).rgb * (w12.x * w3.y);
    const float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / wsum, 0.0);
}

[numthreads(8, 8, 1)] void TaaMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.xy))
        return;
    const int2 p = int2(id);
    float3 m1 = 0, m2 = 0, centre = 0;
    float2 motion = 0;
    float best = -1.0;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
        {
            const int2 q = clamp(p + int2(x, y), int2(0, 0), int2(size.xy) - 1);
            const float3 c = ToYCoCg(Compress(t0.Load(int3(q, 0)).rgb));
            m1 += c;
            m2 += c * c;
            if (x == 0 && y == 0)
                centre = c;
            const float2 mv = t2.Load(int3(q, 0));
            const float len = dot(mv, mv);
            if (len > best)
            {
                best = len;
                motion = mv;
            }
        }
    const float3 mean = m1 / 9.0, sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0)) * p0.z;
    const float3 lo = mean - sigma, hi = mean + sigma;
    const float2 uv = (float2(id) + 0.5) * rt.xy;
    const float2 prevUv = uv + motion;
    float3 result = centre;
    if (p0.y < 0.5 && all(prevUv > 0.0) && all(prevUv < 1.0))
    {
        float3 hist = ToYCoCg(Compress(HistoryCatmullRom(prevUv)));
        // Clip toward the mean (not a clamp per channel: keeps the hue of the history).
        const float3 d = hist - mean;
        const float3 ext = max(hi - mean, 1e-4);
        const float3 unit = abs(d / ext);
        const float m = max(unit.x, max(unit.y, unit.z));
        if (m > 1.0)
            hist = mean + d / m;
        result = lerp(hist, centre, p0.x);
    }
    u0[id] = float4(Expand(FromYCoCg(result)), 1.0);
}

// ---- SSAA ---------------------------------------------------------------------------------------------------------
// t0 the supersampled frame (size.xy), u0 the shown frame (size.zw): each output pixel the area-weighted mean of the
// input texels its footprint covers (exact for any factor; at 2x a 2x2 box).
[numthreads(8, 8, 1)] void SsaaMain(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= size.zw))
        return;
    const float2 f = float2(size.xy) / float2(size.zw);
    const float2 a = float2(id) * f, b = a + f;
    const int2 i0 = int2(floor(a)), i1 = min(int2(ceil(b)), int2(size.xy));
    float4 sum = 0;
    float wsum = 0;
    for (int y = i0.y; y < i1.y; y++)
    {
        const float wy = min(b.y, y + 1.0) - max(a.y, float(y));
        for (int x = i0.x; x < i1.x; x++)
        {
            const float w = wy * (min(b.x, x + 1.0) - max(a.x, float(x)));
            sum += t0.Load(int3(x, y, 0)) * w;
            wsum += w;
        }
    }
    u0[id] = sum / max(wsum, 1e-6);
}
