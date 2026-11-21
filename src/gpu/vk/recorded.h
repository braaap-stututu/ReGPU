#pragma once
// The Vulkan command recorder (docs/vulkan-conversion-plan.md, step 3): the D3D12 RecordedList's job on a
// VkCommandBuffer. The renderer's calls are written to a compact buffer on the draw thread and replayed into the
// real command buffer on a worker thread, so the command processor never waits on the driver. REGPU_NO_RECORD_THREAD=1
// calls the driver directly. It replaces gpu/recorded_list.h once renderer.cpp is converted.
//
// What the replay adds over a plain command buffer:
// - Render targets are recorded, not begun: SetTargets only notes them, and the first draw or clear begins
//   vkCmdBeginRendering with them (load and store, as D3D12's OMSetRenderTargets kept the contents). A barrier, copy,
//   dispatch, query copy, predication change, new targets or the end of the buffer ends it. The renderer never begins
//   or ends rendering itself.
// - Predication (SetPredication, D3D12's) becomes VK_EXT_conditional_rendering, always begun and ended outside a
//   render pass instance (so rendering never has to be split around it); the renderer falls back to CPU decisions
//   where the device lacks the extension (Device::conditionalRendering).
// - The command buffer's pool is used by the replay thread only; Real() (after a drain) and the hand-offs are the
//   only times another thread touches the buffer.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <algorithm>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "gpu/vk/vk_loader.h"

namespace vk
{
// REGPU_PERF: recorded calls by op, and barriers in them.
inline std::atomic<uint64_t> g_recordedOps[48];
inline std::atomic<uint64_t> g_recordedBarriers{ 0 };
inline const bool g_countRecorded = getenv("REGPU_PERF") != nullptr;

template <typename T>
struct NoInitAllocator : std::allocator<T>
{
    template <typename U>
    struct rebind
    {
        using other = NoInitAllocator<U>;
    };
    NoInitAllocator() = default;
    template <typename U>
    NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... A>
    void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
using RecordBytes = std::vector<uint8_t, NoInitAllocator<uint8_t>>;

// Render passes always load and store their targets (VK_ATTACHMENT_LOAD_OP_LOAD), so a barrier into an attachment
// layout must make both reads and writes of the attachment wait (synchronization validation: SYNC-HAZARD-READ-AFTER-
// WRITE on vkCmdBeginRendering otherwise). Use these as the barrier's dstAccessMask (and srcAccessMask out of it).
constexpr VkAccessFlags2 kColorTargetAccess = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
constexpr VkAccessFlags2 kDepthTargetAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
constexpr VkPipelineStageFlags2 kDepthTargetStages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

// The render targets of the next draws (D3D12's OMSetRenderTargets): views in their attachment layouts.
struct Targets
{
    static constexpr uint32_t kMax = 8;
    uint32_t colorCount = 0;
    VkImageView color[kMax] = {};
    VkImageView depth = VK_NULL_HANDLE; // a depth (and stencil) view, or none
    bool depthHasStencil = false;
    bool depthReadOnly = false;         // DEPTH_STENCIL_READ_ONLY_OPTIMAL instead of ATTACHMENT_OPTIMAL
    VkExtent2D extent{};                // the render area (the attachments' smallest size)
    uint32_t generalColor = 0;          // colour attachments in GENERAL (bit per index: Image::general), not ATTACHMENT_OPTIMAL

    bool operator==(const Targets& o) const
    {
        if (colorCount != o.colorCount || depth != o.depth || depthHasStencil != o.depthHasStencil || depthReadOnly != o.depthReadOnly ||
            generalColor != o.generalColor ||
            extent.width != o.extent.width || extent.height != o.extent.height)
            return false;
        for (uint32_t i = 0; i < colorCount; i++)
            if (color[i] != o.color[i])
                return false;
        return true;
    }
};

// A draw's depth, stencil, cull, depth-bias and (with Device::dynamicBlend) attachment 0 blend state, for pipelines made
// with PipelineDesc::dynamicDepthStencil / dynamicBlend (the renderer's draws: one pipeline per shader pair and target
// layout instead of one per state). Compared byte for byte: zero-initialise.
struct DrawState
{
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;
    uint32_t depthTest = 0, depthWrite = 0, stencilTest = 0, depthBias = 0;
    VkCompareOp depthCompare = VK_COMPARE_OP_ALWAYS;
    VkStencilOp frontFail = VK_STENCIL_OP_KEEP, frontPass = VK_STENCIL_OP_KEEP, frontDepthFail = VK_STENCIL_OP_KEEP;
    VkCompareOp frontCompare = VK_COMPARE_OP_ALWAYS;
    VkStencilOp backFail = VK_STENCIL_OP_KEEP, backPass = VK_STENCIL_OP_KEEP, backDepthFail = VK_STENCIL_OP_KEEP;
    VkCompareOp backCompare = VK_COMPARE_OP_ALWAYS;
    uint32_t stencilReadMask = 0xFF, stencilWriteMask = 0xFF;
    float biasConstant = 0, biasSlope = 0;
    uint32_t blend = 0; // 1: dynamic blend set (the fields below)
    uint32_t blendEnable = 0;
    VkColorBlendEquationEXT equation{};
    VkColorComponentFlags writeMask = 0xF;
};

// DrawState's fields as RecordedList::SetDrawState's mask bits.
enum DrawStateField : uint32_t
{
    kDsCull = 1, kDsFrontFace = 2, kDsDepthTest = 4, kDsDepthWrite = 8, kDsDepthCompare = 16, kDsStencilTest = 32, kDsFrontOps = 64,
    kDsBackOps = 128, kDsReadMask = 256, kDsWriteMask = 512, kDsBiasEnable = 1024, kDsBias = 2048, kDsBlend = 4096
};
struct DrawStateSet
{
    DrawState s;
    uint32_t mask;
};

class RecordedList
{
public:
    // direct: call the buffer right away (draw workers, which run in parallel themselves).
    void Init(VkCommandBuffer real, bool direct = false)
    {
        m_real = real;
        m_direct = direct || getenv("REGPU_NO_RECORD_THREAD") != nullptr;
        if (!m_direct)
            std::thread([this] { Worker(); }).detach();
    }

    // Records into another (begun) buffer from now on, after replaying everything so far.
    void Rebind(VkCommandBuffer real)
    {
        Drain();
        m_real = real;
    }

    // Hands the buffer over in order, without waiting: when the replay reaches this point it ends any rendering and
    // conditional rendering, fn gets the buffer so far (to end and submit), and the replay goes on into next (which fn
    // or the caller has begun; null: fn is just called, the replay stays in the same buffer).
    using HandOffFn = void (*)(VkCommandBuffer closed, void* ctx);
    void HandOff(VkCommandBuffer next, HandOffFn fn, void* ctx)
    {
        struct A { VkCommandBuffer next; HandOffFn fn; void* ctx; AcquireFn acquire; void* actx; } a{ next, fn, ctx, nullptr, nullptr };
        Put(kHandOff, &a, sizeof(a));
        Submit();
    }
    // The next buffer, allocated and begun by the replay (a command pool is used by one thread at a time: the one
    // replaying into its buffers allocates the next one, so the draw thread never touches a pool in use).
    using AcquireFn = VkCommandBuffer (*)(void* ctx);
    void HandOffNew(AcquireFn acquire, void* actx, HandOffFn fn, void* ctx)
    {
        struct A { VkCommandBuffer next; HandOffFn fn; void* ctx; AcquireFn acquire; void* actx; } a{ VK_NULL_HANDLE, fn, ctx, acquire, actx };
        Put(kHandOff, &a, sizeof(a));
        Submit();
    }
    // fn(buffer, ctx) at this point of the replay, outside any rendering; the replay goes on in the same buffer (unless
    // fn ended it: then Switch must follow before anything else is recorded). Also how one-off commands are recorded
    // (an upscaler's dispatch, a debug label).
    void Call(HandOffFn fn, void* ctx)
    {
        struct A { HandOffFn fn; void* ctx; } a{ fn, ctx };
        Put(kCall, &a, sizeof(a));
        Submit();
    }
    // Records into another (begun) buffer from here on, without waiting.
    void Switch(VkCommandBuffer next)
    {
        struct A { VkCommandBuffer next; AcquireFn acquire; void* actx; } a{ next, nullptr, nullptr };
        Put(kSwitch, &a, sizeof(a));
    }
    // Switch to a buffer the replay acquires (HandOffNew).
    void SwitchNew(AcquireFn acquire, void* actx)
    {
        struct A { VkCommandBuffer next; AcquireFn acquire; void* actx; } a{ VK_NULL_HANDLE, acquire, actx };
        Put(kSwitch, &a, sizeof(a));
    }

    // The real buffer, with everything recorded so far replayed into it and rendering ended (predication stays).
    VkCommandBuffer Real()
    {
        Put(kEndRendering, nullptr, 0);
        Drain();
        return m_real;
    }

    // synchronization2 barriers (layouts come from the renderer's per-image tracking).
    void Barrier(uint32_t imageCount, const VkImageMemoryBarrier2* images, uint32_t bufferCount = 0, const VkBufferMemoryBarrier2* buffers = nullptr,
        const VkMemoryBarrier2* global = nullptr)
    {
        if (g_countRecorded)
            g_recordedBarriers.fetch_add(imageCount + bufferCount + (global != nullptr), std::memory_order_relaxed);
        struct A { uint32_t images, buffers, global; } a{ imageCount, bufferCount, global != nullptr };
        const size_t ni = sizeof(*images) * imageCount, nb = sizeof(*buffers) * bufferCount, ng = global ? sizeof(*global) : 0;
        Put3(kBarrier, &a, sizeof(a), images, ni, buffers, nb, global, ng);
    }

    void SetTargets(const Targets& t) { Put(kTargets, &t, sizeof(t)); }

    // Clears a colour image's rectangles (rect count 0: the whole render area). The image must be bound as target
    // `index` of the current targets (D3D12's ClearRenderTargetView on a bound RTV), or `view` with `extent` when not:
    // the replay then clears it in a render pass of its own.
    // (general: the image is kept in GENERAL, Image::general.)
    void ClearColor(VkImageView view, VkExtent2D extent, const float c[4], uint32_t rectCount, const VkRect2D* rects, bool general = false)
    {
        struct A { VkImageView view; VkExtent2D extent; float c[4]; uint32_t n, general; } a{ view, extent, { c[0], c[1], c[2], c[3] }, rectCount,
            general };
        Put(kClearColor, &a, sizeof(a), rects, sizeof(*rects) * rectCount);
    }
    void ClearDepthStencil(VkImageView view, VkExtent2D extent, bool hasStencil, VkImageAspectFlags aspects, float depth, uint8_t stencil,
        uint32_t rectCount, const VkRect2D* rects)
    {
        struct A { VkImageView view; VkExtent2D extent; uint32_t hasStencil; VkImageAspectFlags aspects; float depth; uint32_t stencil, n; } a{
            view, extent, hasStencil, aspects, depth, stencil, rectCount
        };
        Put(kClearDepth, &a, sizeof(a), rects, sizeof(*rects) * rectCount);
    }

    void BindPipeline(VkPipelineBindPoint point, VkPipeline p) { struct A { VkPipelineBindPoint point; VkPipeline p; } a{ point, p }; Put(kPipeline, &a, sizeof(a)); }
    // A pipeline still being built: the replay calls resolve(ctx) when it reaches the bind (on the replay thread, so the
    // recording goes on meanwhile) and binds what it returns. VK_NULL_HANDLE (the build failed): the draws up to the
    // next bind are dropped.
    using ResolveFn = VkPipeline (*)(void* ctx);
    void BindPipelineDeferred(VkPipelineBindPoint point, ResolveFn resolve, void* ctx)
    {
        struct A { VkPipelineBindPoint point; ResolveFn fn; void* ctx; } a{ point, resolve, ctx };
        Put(kPipelineDeferred, &a, sizeof(a));
    }
    bool Direct() const { return m_direct; }
    void BindDescriptorSets(VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet* sets,
        uint32_t dynamicCount = 0, const uint32_t* dynamicOffsets = nullptr)
    {
        struct A { VkPipelineBindPoint point; VkPipelineLayout layout; uint32_t first, count, dynamicCount; } a{ point, layout, first, count, dynamicCount };
        Put3(kSets, &a, sizeof(a), sets, sizeof(*sets) * count, dynamicOffsets, sizeof(*dynamicOffsets) * dynamicCount, nullptr, 0);
    }
    // Root constants (SetGraphicsRoot32BitConstants / SetComputeRoot32BitConstants).
    void PushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t bytes, const void* data)
    {
        struct A { VkPipelineLayout layout; VkShaderStageFlags stages; uint32_t offset, bytes; } a{ layout, stages, offset, bytes };
        Put(kPush, &a, sizeof(a), data, bytes);
    }
    void SetViewports(uint32_t n, const VkViewport* v) { Put(kViewports, &n, sizeof(n), v, sizeof(*v) * n); }
    void SetScissors(uint32_t n, const VkRect2D* r) { Put(kScissors, &n, sizeof(n), r, sizeof(*r) * n); }
    void SetStencilReference(uint32_t ref) { Put(kStencilRef, &ref, sizeof(ref)); }
    void SetBlendConstants(const float f[4]) { Put(kBlendConstants, f, sizeof(float) * 4); }
    void SetPrimitiveTopology(VkPrimitiveTopology t) { Put(kTopology, &t, sizeof(t)); }
    // mask: the fields to set (DrawStateField bits; all after an invalidation), so a draw that changes one state doesn't
    // send the other eleven again.
    void SetDrawState(const DrawState& s, uint32_t mask = ~0u)
    {
        const DrawStateSet a{ s, mask };
        Put(kDrawState, &a, sizeof(a));
    }
    void BindIndexBuffer(VkBuffer b, VkDeviceSize offset, VkIndexType type)
    {
        struct A { VkBuffer b; VkDeviceSize offset; VkIndexType type; } a{ b, offset, type };
        Put(kIndexBuffer, &a, sizeof(a));
    }
    void Draw(uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance)
    {
        uint32_t a[4] = { vertices, instances, firstVertex, firstInstance };
        Put(kDraw, a, sizeof(a));
    }
    void DrawIndexed(uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
    {
        uint32_t a[5] = { indices, instances, firstIndex, uint32_t(vertexOffset), firstInstance };
        Put(kDrawIndexed, a, sizeof(a));
    }
    void Dispatch(uint32_t x, uint32_t y, uint32_t z) { uint32_t a[3] = { x, y, z }; Put(kDispatch, a, sizeof(a)); }

    void CopyBuffer(VkBuffer dst, VkDeviceSize dstOffset, VkBuffer src, VkDeviceSize srcOffset, VkDeviceSize bytes)
    {
        struct A { VkBuffer dst; VkDeviceSize dstOffset; VkBuffer src; VkDeviceSize srcOffset, bytes; } a{ dst, dstOffset, src, srcOffset, bytes };
        Put(kCopyBuffer, &a, sizeof(a));
    }
    void CopyBufferToImage(VkBuffer src, VkImage dst, VkImageLayout layout, uint32_t n, const VkBufferImageCopy* regions)
    {
        struct A { VkBuffer src; VkImage dst; VkImageLayout layout; uint32_t n; } a{ src, dst, layout, n };
        Put(kBufferToImage, &a, sizeof(a), regions, sizeof(*regions) * n);
    }
    void CopyImageToBuffer(VkImage src, VkImageLayout layout, VkBuffer dst, uint32_t n, const VkBufferImageCopy* regions)
    {
        struct A { VkImage src; VkImageLayout layout; VkBuffer dst; uint32_t n; } a{ src, layout, dst, n };
        Put(kImageToBuffer, &a, sizeof(a), regions, sizeof(*regions) * n);
    }
    void CopyImage(VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout, uint32_t n, const VkImageCopy* regions)
    {
        struct A { VkImage src; VkImageLayout srcLayout; VkImage dst; VkImageLayout dstLayout; uint32_t n; } a{ src, srcLayout, dst, dstLayout, n };
        Put(kCopyImage, &a, sizeof(a), regions, sizeof(*regions) * n);
    }
    // An MSAA image's samples averaged into a 1x one (D3D12's ResolveSubresourceRegion).
    void ResolveImage(VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout, const VkImageResolve& region)
    {
        struct A { VkImage src; VkImageLayout srcLayout; VkImage dst; VkImageLayout dstLayout; VkImageResolve r; } a{ src, srcLayout, dst, dstLayout, region };
        Put(kResolve, &a, sizeof(a));
    }
    void FillBuffer(VkBuffer dst, VkDeviceSize offset, VkDeviceSize bytes, uint32_t value)
    {
        struct A { VkBuffer dst; VkDeviceSize offset, bytes; uint32_t value; } a{ dst, offset, bytes, value };
        Put(kFill, &a, sizeof(a));
    }

    // Occlusion queries (precise: the sample count, as D3D12_QUERY_TYPE_OCCLUSION). Query pools are reset with
    // ResetQueries outside rendering before reuse.
    void ResetQueries(VkQueryPool pool, uint32_t first, uint32_t count)
    {
        struct A { VkQueryPool pool; uint32_t first, count; } a{ pool, first, count };
        Put(kResetQueries, &a, sizeof(a));
    }
    // inPass: around one draw (or a few) with the same targets: begun and ended inside the render pass instance, so the
    // query costs no render pass break. Otherwise begun and ended outside rendering (a query over many draws, target
    // changes and barriers, as D3D12's could be).
    void BeginQuery(VkQueryPool pool, uint32_t index, bool precise = true, bool inPass = false)
    {
        struct A { VkQueryPool pool; uint32_t index, precise, inPass; } a{ pool, index, precise, inPass };
        Put(kBeginQuery, &a, sizeof(a));
    }
    void EndQuery(VkQueryPool pool, uint32_t index, bool inPass = false)
    {
        struct A { VkQueryPool pool; uint32_t index, inPass; } a{ pool, index, inPass };
        Put(kEndQuery, &a, sizeof(a));
    }
    // D3D12's ResolveQueryData: 64-bit results, `stride` bytes apart (pipeline statistics: 8 per counter), waiting for
    // them on the GPU.
    void CopyQueryResults(VkQueryPool pool, uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride = 8)
    {
        struct A { VkQueryPool pool; uint32_t first, count; VkBuffer dst; VkDeviceSize offset, stride; } a{ pool, first, count, dst, offset, stride };
        Put(kCopyQueries, &a, sizeof(a));
    }
    void WriteTimestamp(VkQueryPool pool, uint32_t index) { struct A { VkQueryPool pool; uint32_t index; } a{ pool, index }; Put(kTimestamp, &a, sizeof(a)); }

    // D3D12's SetPredication: draws until the next call are skipped when the 32-bit value at buffer+offset is zero
    // (inverted: non-zero). buffer null ends predication.
    void SetPredication(VkBuffer buffer, VkDeviceSize offset, bool inverted)
    {
        struct A { VkBuffer b; VkDeviceSize offset; uint32_t inverted; } a{ buffer, offset, inverted };
        Put(kPredication, &a, sizeof(a));
    }

    // A debug label around the following commands (RenderDoc, RGP; vkCmdBeginDebugUtilsLabelEXT when available).
    void BeginLabel(const char* name) { Put(kBeginLabel, name, strlen(name) + 1); }
    void EndLabel() { Put(kEndLabel, nullptr, 0); }

    // Time spent waiting in Drain for the replay thread (summed; the long-frame log reads the difference per frame).
    double m_drainWaitMs = 0;

private:
    enum Op : uint32_t
    {
        kBarrier, kTargets, kClearColor, kClearDepth, kPipeline, kSets, kPush, kViewports, kScissors, kStencilRef, kBlendConstants, kTopology, kDrawState,
        kIndexBuffer, kDraw, kDrawIndexed, kDispatch, kCopyBuffer, kBufferToImage, kImageToBuffer, kCopyImage, kResolve, kFill, kResetQueries,
        kBeginQuery, kEndQuery, kCopyQueries, kTimestamp, kPredication, kBeginLabel, kEndLabel, kHandOff, kCall, kSwitch, kEndRendering,
        kPipelineDeferred
    };
    static constexpr size_t kChunk = 64 << 10;

    VkCommandBuffer m_real = VK_NULL_HANDLE;
    bool m_direct = false;
    bool m_dropDraws = false; // replay: a deferred pipeline failed, the draws until the next bind are dropped
    RecordBytes m_current;
    std::mutex m_mutex;
    std::condition_variable m_work, m_idle;
    std::deque<RecordBytes> m_queue, m_spare;
    bool m_busy = false;

    // Replay-thread state.
    Targets m_targets{};
    bool m_targetsSet = false;
    bool m_rendering = false;
    bool m_predicated = false;

    void Put(Op op, const void* a, size_t na, const void* b = nullptr, size_t nb = 0) { Put3(op, a, na, b, nb, nullptr, 0, nullptr, 0); }

    void Put3(Op op, const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc, const void* d, size_t nd)
    {
        if (g_countRecorded)
            g_recordedOps[op].fetch_add(1, std::memory_order_relaxed);
        const size_t at = m_current.size();
        const uint32_t header[2] = { op, uint32_t((na + nb + nc + nd + 7) & ~size_t(7)) }; // 8-byte aligned commands
        m_current.resize(at + sizeof(header) + header[1]);
        uint8_t* p = m_current.data() + at + sizeof(header);
        memcpy(p - sizeof(header), header, sizeof(header));
        if (na)
            memcpy(p, a, na);
        if (nb)
            memcpy(p + na, b, nb);
        if (nc)
            memcpy(p + na + nb, c, nc);
        if (nd)
            memcpy(p + na + nb + nc, d, nd);
        if (m_direct)
        {
            Replay(m_current.data(), m_current.size());
            m_current.clear();
        }
        else if (op == kTargets)
        {
            // Hands what precedes new targets to the replay thread at once (the D3D12 recorder did this for new views:
            // PGR4 on an FX-8350 52.8 -> 54.8 fps). REGPU_NO_VIEW_SUBMIT=1: at the chunk size only.
            static const bool noViewSubmit = getenv("REGPU_NO_VIEW_SUBMIT") != nullptr;
            if (noViewSubmit && m_current.size() < kChunk)
                return;
            Submit();
        }
        else if (m_current.size() >= kChunk)
            Submit();
    }

    void Submit()
    {
        if (m_current.empty() || m_direct)
            return;
        std::lock_guard lock(m_mutex);
        m_queue.push_back(std::move(m_current));
        if (!m_spare.empty())
        {
            m_current = std::move(m_spare.back());
            m_spare.pop_back();
        }
        m_current.clear();
        m_current.reserve(kChunk + 4096);
        m_work.notify_one();
    }

    void Drain()
    {
        if (m_direct)
            return;
        Submit();
        std::unique_lock lock(m_mutex);
        if (m_queue.empty() && !m_busy)
            return;
        const auto w0 = std::chrono::steady_clock::now();
        m_idle.wait(lock, [this] { return m_queue.empty() && !m_busy; });
        m_drainWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    }

    void Worker()
    {
#if defined(_WIN32)
        SetThreadDescription(GetCurrentThread(), L"command buffer replay");
        if (const char* p = getenv("REGPU_DRAW_THREAD_PRIORITY"); p && *p) // (gpu.cpp WorkerLoop)
            SetThreadPriority(GetCurrentThread(), atoi(p));
#endif
        for (;;)
        {
            RecordBytes chunk;
            {
                std::unique_lock lock(m_mutex);
                m_work.wait(lock, [this] { return !m_queue.empty(); });
                chunk = std::move(m_queue.front());
                m_queue.pop_front();
                m_busy = true;
            }
            Replay(chunk.data(), chunk.size());
            {
                std::lock_guard lock(m_mutex);
                chunk.clear();
                m_spare.push_back(std::move(chunk));
                m_busy = false;
                if (m_queue.empty())
                    m_idle.notify_all();
            }
        }
    }

    template <typename T>
    static const T& At(const uint8_t* p) { return *reinterpret_cast<const T*>(p); }

    void BeginRendering(VkCommandBuffer cb, const Targets& t)
    {
        VkRenderingAttachmentInfo color[Targets::kMax];
        for (uint32_t i = 0; i < t.colorCount; i++)
        {
            color[i] = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            color[i].imageView = t.color[i];
            color[i].imageLayout = (t.generalColor >> i) & 1 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingAttachmentInfo depth{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depth.imageView = t.depth;
        depth.imageLayout = t.depthReadOnly ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = t.depthReadOnly ? VK_ATTACHMENT_STORE_OP_NONE : VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea = { { 0, 0 }, t.extent };
        ri.layerCount = 1;
        ri.colorAttachmentCount = t.colorCount;
        ri.pColorAttachments = color;
        ri.pDepthAttachment = t.depth ? &depth : nullptr;
        ri.pStencilAttachment = t.depth && t.depthHasStencil ? &depth : nullptr;
        vkCmdBeginRendering(cb, &ri);
    }

    void EnsureRendering(VkCommandBuffer cb)
    {
        if (!m_rendering && m_targetsSet)
        {
            BeginRendering(cb, m_targets);
            m_rendering = true;
        }
    }

    void EndRendering(VkCommandBuffer cb)
    {
        if (m_rendering)
        {
            vkCmdEndRendering(cb);
            m_rendering = false;
        }
    }

    // Outside rendering and predication: what a hand-off or a Call needs.
    void EndAll(VkCommandBuffer cb)
    {
        EndRendering(cb);
        if (m_predicated)
        {
            vkCmdEndConditionalRenderingEXT(cb);
            m_predicated = false;
        }
    }

    // Clears rectangles of one attachment: inside the current rendering when it is one of the targets, else in a
    // render pass of its own.
    void ClearAttachment(VkCommandBuffer cb, VkImageView view, VkExtent2D extent, const VkClearAttachment& clear, uint32_t n, const VkRect2D* rects,
        bool depth, bool hasStencil, bool general = false)
    {
        int bound = -1;
        if (m_targetsSet)
        {
            if (depth && m_targets.depth == view)
                bound = 0;
            for (uint32_t i = 0; !depth && i < m_targets.colorCount; i++)
                if (m_targets.color[i] == view)
                    bound = int(i);
        }
        VkClearAttachment c = clear;
        VkRect2D whole{ { 0, 0 }, bound >= 0 ? m_targets.extent : extent };
        if (bound >= 0)
        {
            EnsureRendering(cb);
            if (!depth)
                c.colorAttachment = uint32_t(bound);
        }
        else
        {
            EndRendering(cb);
            Targets own{};
            if (depth)
            {
                own.depth = view;
                own.depthHasStencil = hasStencil;
            }
            else
            {
                own.colorCount = 1;
                own.color[0] = view;
                own.generalColor = general ? 1u : 0u;
                c.colorAttachment = 0;
            }
            own.extent = extent;
            BeginRendering(cb, own);
        }
        VkClearRect cr[16];
        uint32_t done = 0;
        do
        {
            const uint32_t batch = n ? std::min<uint32_t>(n - done, 16) : 1;
            for (uint32_t i = 0; i < batch; i++)
                cr[i] = { n ? rects[done + i] : whole, 0, 1 };
            vkCmdClearAttachments(cb, 1, &c, batch, cr);
            done += batch;
        } while (done < n);
        if (bound < 0)
            vkCmdEndRendering(cb);
    }

    void Replay(const uint8_t* p, size_t size)
    {
        VkCommandBuffer cb = m_real;
        for (const uint8_t* end = p + size; p < end;)
        {
            uint32_t header[2];
            memcpy(header, p, sizeof(header));
            const uint8_t* a = p + sizeof(header);
            p = a + header[1];
            switch (header[0])
            {
            case kBarrier:
            {
                struct A { uint32_t images, buffers, global; };
                const A& c = At<A>(a);
                EndRendering(cb);
                // Barriers recorded back to back (nothing recorded between them) go out as one vkCmdPipelineBarrier2: on
                // AMD each call waits and flushes on its own (RGP, FM4's grid: 93 of 319 barriers came right after
                // another). Not merged when the next one names an image already in the batch (a second transition of
                // the same image must follow the first). REGPU_BARRIER_MERGE=1 (start-time, off by default): FM4's 1440p 4x
                // grid on an RX 7900 XTX ran the same with it (3.52-3.62 ms against 3.52-3.54, alternated boots), and
                // synchronization validation reported nothing.
                static const bool noMerge = !(getenv("REGPU_BARRIER_MERGE") && strcmp(getenv("REGPU_BARRIER_MERGE"), "0") != 0);
                if (!noMerge && p < end)
                {
                    uint32_t next[2];
                    memcpy(next, p, sizeof(next));
                    if (next[0] == kBarrier)
                    {
                        thread_local std::vector<VkImageMemoryBarrier2> images;
                        thread_local std::vector<VkBufferMemoryBarrier2> buffers;
                        thread_local std::vector<VkMemoryBarrier2> globals;
                        images.clear();
                        buffers.clear();
                        globals.clear();
                        auto take = [&](const uint8_t* q) {
                            const A& x = At<A>(q);
                            const auto* im = reinterpret_cast<const VkImageMemoryBarrier2*>(q + sizeof(A));
                            const auto* bu = reinterpret_cast<const VkBufferMemoryBarrier2*>(q + sizeof(A) + sizeof(VkImageMemoryBarrier2) * x.images);
                            const auto* gl = reinterpret_cast<const VkMemoryBarrier2*>(
                                q + sizeof(A) + sizeof(VkImageMemoryBarrier2) * x.images + sizeof(VkBufferMemoryBarrier2) * x.buffers);
                            images.insert(images.end(), im, im + x.images);
                            buffers.insert(buffers.end(), bu, bu + x.buffers);
                            globals.insert(globals.end(), gl, gl + x.global);
                        };
                        take(a);
                        while (p < end)
                        {
                            memcpy(next, p, sizeof(next));
                            if (next[0] != kBarrier)
                                break;
                            const uint8_t* q = p + sizeof(next);
                            const A& x = At<A>(q);
                            const auto* im = reinterpret_cast<const VkImageMemoryBarrier2*>(q + sizeof(A));
                            bool repeat = false;
                            for (uint32_t i = 0; i < x.images && !repeat; i++)
                                for (const VkImageMemoryBarrier2& have : images)
                                    if (have.image == im[i].image)
                                    {
                                        repeat = true;
                                        break;
                                    }
                            if (repeat || x.buffers || x.global || !buffers.empty() || !globals.empty())
                                break; // (buffer and global barriers: kept apart, as recorded)
                            take(q);
                            p = q + next[1];
                        }
                        VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                        di.imageMemoryBarrierCount = uint32_t(images.size());
                        di.pImageMemoryBarriers = images.data();
                        di.bufferMemoryBarrierCount = uint32_t(buffers.size());
                        di.pBufferMemoryBarriers = buffers.data();
                        di.memoryBarrierCount = uint32_t(globals.size());
                        di.pMemoryBarriers = globals.data();
                        vkCmdPipelineBarrier2(cb, &di);
                        break;
                    }
                }
                VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                di.imageMemoryBarrierCount = c.images;
                di.pImageMemoryBarriers = reinterpret_cast<const VkImageMemoryBarrier2*>(a + sizeof(A));
                di.bufferMemoryBarrierCount = c.buffers;
                di.pBufferMemoryBarriers = reinterpret_cast<const VkBufferMemoryBarrier2*>(a + sizeof(A) + sizeof(VkImageMemoryBarrier2) * c.images);
                di.memoryBarrierCount = c.global;
                di.pMemoryBarriers = reinterpret_cast<const VkMemoryBarrier2*>(
                    a + sizeof(A) + sizeof(VkImageMemoryBarrier2) * c.images + sizeof(VkBufferMemoryBarrier2) * c.buffers);
                vkCmdPipelineBarrier2(cb, &di);
                break;
            }
            case kTargets:
            {
                const Targets& t = At<Targets>(a);
                // The same targets again keep the current rendering going.
                if (m_targetsSet && t == m_targets)
                    break;
                EndRendering(cb);
                m_targets = t;
                m_targetsSet = t.colorCount || t.depth;
                break;
            }
            case kClearColor:
            {
                struct A { VkImageView view; VkExtent2D extent; float c[4]; uint32_t n, general; };
                const A& c = At<A>(a);
                VkClearAttachment clear{ VK_IMAGE_ASPECT_COLOR_BIT, 0, {} };
                memcpy(clear.clearValue.color.float32, c.c, sizeof(c.c));
                ClearAttachment(cb, c.view, c.extent, clear, c.n, reinterpret_cast<const VkRect2D*>(a + sizeof(A)), false, false, c.general != 0);
                break;
            }
            case kClearDepth:
            {
                struct A { VkImageView view; VkExtent2D extent; uint32_t hasStencil; VkImageAspectFlags aspects; float depth; uint32_t stencil, n; };
                const A& c = At<A>(a);
                VkClearAttachment clear{ c.aspects, 0, {} };
                clear.clearValue.depthStencil = { c.depth, c.stencil };
                ClearAttachment(cb, c.view, c.extent, clear, c.n, reinterpret_cast<const VkRect2D*>(a + sizeof(A)), true, c.hasStencil != 0);
                break;
            }
            case kPipeline:
            {
                struct A { VkPipelineBindPoint point; VkPipeline p; };
                vkCmdBindPipeline(cb, At<A>(a).point, At<A>(a).p);
                if (At<A>(a).point == VK_PIPELINE_BIND_POINT_GRAPHICS)
                    m_dropDraws = false;
                break;
            }
            case kPipelineDeferred:
            {
                struct A { VkPipelineBindPoint point; ResolveFn fn; void* ctx; };
                const A& d = At<A>(a);
                const VkPipeline p = d.fn(d.ctx);
                if (d.point == VK_PIPELINE_BIND_POINT_GRAPHICS)
                    m_dropDraws = !p;
                if (p)
                    vkCmdBindPipeline(cb, d.point, p);
                break;
            }
            case kSets:
            {
                struct A { VkPipelineBindPoint point; VkPipelineLayout layout; uint32_t first, count, dynamicCount; };
                const A& c = At<A>(a);
                const auto* sets = reinterpret_cast<const VkDescriptorSet*>(a + sizeof(A));
                vkCmdBindDescriptorSets(cb, c.point, c.layout, c.first, c.count, sets, c.dynamicCount,
                    reinterpret_cast<const uint32_t*>(a + sizeof(A) + sizeof(VkDescriptorSet) * c.count));
                break;
            }
            case kPush:
            {
                struct A { VkPipelineLayout layout; VkShaderStageFlags stages; uint32_t offset, bytes; };
                const A& c = At<A>(a);
                vkCmdPushConstants(cb, c.layout, c.stages, c.offset, c.bytes, a + sizeof(A));
                break;
            }
            case kViewports: vkCmdSetViewport(cb, 0, At<uint32_t>(a), reinterpret_cast<const VkViewport*>(a + sizeof(uint32_t))); break;
            case kScissors: vkCmdSetScissor(cb, 0, At<uint32_t>(a), reinterpret_cast<const VkRect2D*>(a + sizeof(uint32_t))); break;
            case kStencilRef: vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, At<uint32_t>(a)); break;
            case kBlendConstants: vkCmdSetBlendConstants(cb, reinterpret_cast<const float*>(a)); break;
            case kTopology: vkCmdSetPrimitiveTopology(cb, At<VkPrimitiveTopology>(a)); break;
            case kDrawState:
            {
                const DrawStateSet& set = At<DrawStateSet>(a);
                const DrawState& s = set.s;
                const uint32_t m = set.mask;
                if (m & kDsCull)
                    vkCmdSetCullMode(cb, s.cull);
                if (m & kDsFrontFace)
                    vkCmdSetFrontFace(cb, s.frontFace);
                if (m & kDsDepthTest)
                    vkCmdSetDepthTestEnable(cb, s.depthTest);
                if (m & kDsDepthWrite)
                    vkCmdSetDepthWriteEnable(cb, s.depthWrite);
                if (m & kDsDepthCompare)
                    vkCmdSetDepthCompareOp(cb, s.depthCompare);
                if (m & kDsStencilTest)
                    vkCmdSetStencilTestEnable(cb, s.stencilTest);
                if (m & kDsFrontOps)
                    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, s.frontFail, s.frontPass, s.frontDepthFail, s.frontCompare);
                if (m & kDsBackOps)
                    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, s.backFail, s.backPass, s.backDepthFail, s.backCompare);
                if (m & kDsReadMask)
                    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, s.stencilReadMask);
                if (m & kDsWriteMask)
                    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, s.stencilWriteMask);
                if (m & kDsBiasEnable)
                    vkCmdSetDepthBiasEnable(cb, s.depthBias);
                if (m & kDsBias)
                    vkCmdSetDepthBias(cb, s.biasConstant, 0.0f, s.biasSlope);
                if (s.blend && (m & kDsBlend))
                {
                    vkCmdSetColorBlendEnableEXT(cb, 0, 1, &s.blendEnable);
                    vkCmdSetColorBlendEquationEXT(cb, 0, 1, &s.equation);
                    vkCmdSetColorWriteMaskEXT(cb, 0, 1, &s.writeMask);
                }
                break;
            }
            case kIndexBuffer:
            {
                struct A { VkBuffer b; VkDeviceSize offset; VkIndexType type; };
                vkCmdBindIndexBuffer(cb, At<A>(a).b, At<A>(a).offset, At<A>(a).type);
                break;
            }
            case kDraw:
            {
                if (m_dropDraws)
                    break;
                const uint32_t* d = reinterpret_cast<const uint32_t*>(a);
                EnsureRendering(cb);
                vkCmdDraw(cb, d[0], d[1], d[2], d[3]);
                break;
            }
            case kDrawIndexed:
            {
                if (m_dropDraws)
                    break;
                const uint32_t* d = reinterpret_cast<const uint32_t*>(a);
                EnsureRendering(cb);
                vkCmdDrawIndexed(cb, d[0], d[1], d[2], int32_t(d[3]), d[4]);
                break;
            }
            case kDispatch:
            {
                const uint32_t* d = reinterpret_cast<const uint32_t*>(a);
                EndRendering(cb);
                vkCmdDispatch(cb, d[0], d[1], d[2]);
                break;
            }
            case kCopyBuffer:
            {
                struct A { VkBuffer dst; VkDeviceSize dstOffset; VkBuffer src; VkDeviceSize srcOffset, bytes; };
                const A& c = At<A>(a);
                EndRendering(cb);
                const VkBufferCopy region{ c.srcOffset, c.dstOffset, c.bytes };
                vkCmdCopyBuffer(cb, c.src, c.dst, 1, &region);
                break;
            }
            case kBufferToImage:
            {
                struct A { VkBuffer src; VkImage dst; VkImageLayout layout; uint32_t n; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdCopyBufferToImage(cb, c.src, c.dst, c.layout, c.n, reinterpret_cast<const VkBufferImageCopy*>(a + sizeof(A)));
                break;
            }
            case kImageToBuffer:
            {
                struct A { VkImage src; VkImageLayout layout; VkBuffer dst; uint32_t n; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdCopyImageToBuffer(cb, c.src, c.layout, c.dst, c.n, reinterpret_cast<const VkBufferImageCopy*>(a + sizeof(A)));
                break;
            }
            case kCopyImage:
            {
                struct A { VkImage src; VkImageLayout srcLayout; VkImage dst; VkImageLayout dstLayout; uint32_t n; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdCopyImage(cb, c.src, c.srcLayout, c.dst, c.dstLayout, c.n, reinterpret_cast<const VkImageCopy*>(a + sizeof(A)));
                break;
            }
            case kResolve:
            {
                struct A { VkImage src; VkImageLayout srcLayout; VkImage dst; VkImageLayout dstLayout; VkImageResolve r; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdResolveImage(cb, c.src, c.srcLayout, c.dst, c.dstLayout, 1, &c.r);
                break;
            }
            case kFill:
            {
                struct A { VkBuffer dst; VkDeviceSize offset, bytes; uint32_t value; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdFillBuffer(cb, c.dst, c.offset, c.bytes, c.value);
                break;
            }
            case kResetQueries:
            {
                struct A { VkQueryPool pool; uint32_t first, count; };
                EndRendering(cb);
                vkCmdResetQueryPool(cb, At<A>(a).pool, At<A>(a).first, At<A>(a).count);
                break;
            }
            case kBeginQuery:
            {
                struct A { VkQueryPool pool; uint32_t index, precise, inPass; };
                // Outside rendering, so a query may span target changes and barriers as D3D12's did (a query begun
                // inside a render pass instance must end inside it); inPass (around a draw): inside, no break.
                if (At<A>(a).inPass)
                    EnsureRendering(cb);
                else
                    EndRendering(cb);
                vkCmdBeginQuery(cb, At<A>(a).pool, At<A>(a).index, At<A>(a).precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
                break;
            }
            case kEndQuery:
            {
                struct A { VkQueryPool pool; uint32_t index, inPass; };
                if (!At<A>(a).inPass)
                    EndRendering(cb);
                vkCmdEndQuery(cb, At<A>(a).pool, At<A>(a).index);
                break;
            }
            case kCopyQueries:
            {
                struct A { VkQueryPool pool; uint32_t first, count; VkBuffer dst; VkDeviceSize offset, stride; };
                const A& c = At<A>(a);
                EndRendering(cb);
                vkCmdCopyQueryPoolResults(cb, c.pool, c.first, c.count, c.dst, c.offset, c.stride, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                break;
            }
            case kTimestamp:
            {
                struct A { VkQueryPool pool; uint32_t index; };
                vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, At<A>(a).pool, At<A>(a).index);
                break;
            }
            case kPredication:
            {
                struct A { VkBuffer b; VkDeviceSize offset; uint32_t inverted; };
                const A& c = At<A>(a);
                EndRendering(cb);
                if (m_predicated)
                {
                    vkCmdEndConditionalRenderingEXT(cb);
                    m_predicated = false;
                }
                if (c.b)
                {
                    VkConditionalRenderingBeginInfoEXT ci{ VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT };
                    ci.buffer = c.b;
                    ci.offset = c.offset;
                    ci.flags = c.inverted ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0;
                    vkCmdBeginConditionalRenderingEXT(cb, &ci);
                    m_predicated = true;
                }
                break;
            }
            case kBeginLabel:
                if (vkCmdBeginDebugUtilsLabelEXT)
                {
                    VkDebugUtilsLabelEXT label{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
                    label.pLabelName = reinterpret_cast<const char*>(a);
                    vkCmdBeginDebugUtilsLabelEXT(cb, &label);
                }
                break;
            case kEndLabel:
                if (vkCmdEndDebugUtilsLabelEXT)
                    vkCmdEndDebugUtilsLabelEXT(cb);
                break;
            case kCall:
            {
                struct A { HandOffFn fn; void* ctx; };
                EndAll(cb);
                At<A>(a).fn(cb, At<A>(a).ctx);
                break;
            }
            case kSwitch:
            {
                struct A { VkCommandBuffer next; AcquireFn acquire; void* actx; };
                const A& c = At<A>(a);
                EndAll(cb);
                m_targetsSet = false;
                cb = m_real = c.acquire ? c.acquire(c.actx) : c.next;
                break;
            }
            case kHandOff:
            {
                struct A { VkCommandBuffer next; HandOffFn fn; void* ctx; AcquireFn acquire; void* actx; };
                const A& c = At<A>(a);
                EndAll(cb);
                const bool fresh = c.next || c.acquire;
                if (fresh)
                    m_targetsSet = false; // a new buffer starts with no rendering state
                c.fn(cb, c.ctx);
                if (fresh)
                    cb = m_real = c.acquire ? c.acquire(c.actx) : c.next;
                break;
            }
            case kEndRendering: EndRendering(cb); break;
            }
        }
    }
};
}
