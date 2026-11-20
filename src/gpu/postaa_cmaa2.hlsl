// CMAA 2 (Intel, Filip Strugar; Apache 2.0) over postaa.hlsl's compressed image: CMAA2.hlsl's four kernels, from the
// host's REGPU_AA_DIR/cmaa2. Input t0 = the compressed image, output u0 = a copy of it the last pass blends into (only
// pixels on edges are written). RGBA16F output (typed store, float), single-sample, luma computed in the shader,
// quality preset HIGH (2). Working buffers u1-u7 as Intel's DX12 sample sizes them (renderer.cpp, PostAA).
#define CMAA2_UAV_STORE_TYPED 1
#define CMAA2_UAV_STORE_TYPED_UNORM_FLOAT 0
#define CMAA2_UAV_STORE_CONVERT_TO_SRGB 0
#define CMAA2_SUPPORT_HDR_COLOR_RANGE 0
#define CMAA2_EDGE_DETECTION_LUMA_PATH 1
#define CMAA2_STATIC_QUALITY_PRESET 2
#define CMAA2_EXTRA_SHARPNESS 0
#include "CMAA2.hlsl"
