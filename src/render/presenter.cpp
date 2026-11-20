// Native output: a window with a Vulkan swap chain. The guest's VdSwap hands a frame over; until the renderer draws
// (PresenterUseRenderer), a render thread presents the presenter's own frames (a clear whose shade pulses with the
// frame count, or a CPU image: PresenterSubmitImage). Then the renderer copies each finished frame into the swap
// chain's image (PresenterRecordCopy), submits it with the acquire's wait (PresenterSubmit) and presents
// (PresenterPresent).
#include "render/xr.h"
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include <windows.h>

#include "live_switch.h"
#include "log.h"
#include "precise_sleep.h"
#include "render/presenter.h"
#include "gpu/vk/device.h"
#include "gpu/vk/resources.h"
#include "regpu/regpu.h"
#include "settings.h"


namespace
{
constexpr UINT kBuffers = 3;  // the presenter's own frames in flight
constexpr UINT kAcquire = 8;  // acquire semaphores, used in turn
constexpr UINT kWidth = 1280, kHeight = 720;

struct Presenter
{
    HWND window = nullptr;
    HWND surface = nullptr; // the child window holding the swap chain (not with Config::hostWindow)
    std::atomic<bool> fullscreen{ false };

    // The swap chain.
    vk::Device* d = nullptr;
    VkSurfaceKHR vkSurface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    bool vsync = true;           // the mode it was made with (FIFO; else immediate where offered)
    std::atomic<bool> recreate{ false };
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> rendered; // per image: the frame's commands are done (the present waits on it)
    VkSemaphore acquired[kAcquire] = {};
    uint64_t acquiredUse[kAcquire] = {}; // the device timeline value of the submission that waited on each
    uint32_t acquireNext = 0;
    // The image acquired for the frame being made (PresenterRecordCopy) until it is presented; its acquire semaphore
    // until a submission waits on it (PresenterSubmit).
    uint32_t imageIndex = UINT32_MAX;
    int pendingAcquire = -1;
    bool submitted = false;
    std::recursive_mutex swapMutex; // (the presenter's thread and the renderer's)
    vk::Image gradeTarget;          // a graded picture of another size than the back buffer, before it is scaled

    // The presenter's own frames.
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cbs[kBuffers] = {};
    uint64_t cbDone[kBuffers] = {};
    vk::Buffer upload[kBuffers];
    vk::Image cpuImage; // the submitted CPU image on the GPU (RGBA8)
    UINT ownFrame = 0;

    std::mutex mutex;
    std::condition_variable cv;
    uint64_t pendingFrame = 0;
    std::vector<uint8_t> image; // latest submitted RGBA8, imageWidth x imageHeight
    uint32_t imageWidth = 0, imageHeight = 0;
    // The parts-bin Vulkan backend presents into the window itself (PresenterClaimWindow): the presenter then draws nothing.
    std::atomic<bool> claimed{ false };
    uint64_t presentedFrame = 0;
    std::atomic<bool> ready{ false };
};

Presenter g_p;
std::atomic<bool> g_rendererActive{ false };
bool g_hostPresents = false; // Host::presentImage in use (PresenterInit, before the renderer's thread starts)
}

// The size the game renders at (settings: resolution and forced aspect ratio).
void RenderSize(uint32_t& width, uint32_t& height)
{
    GetSettings().RenderSize(width, height);
}

namespace
{

void LayoutSurface();

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (msg == WM_SIZE && hwnd == g_p.window)
        LayoutSurface();
    if (msg == WM_CLOSE)
    {
        Log("presenter: window closed, exiting");
        std::_Exit(0);
    }
    if (msg == WM_CHAR)
    {
        if (regpu::GetHost().onChar)
            regpu::GetHost().onChar(wchar_t(wp));
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// A windowed game's client area at exactly `width` x `height` pixels (the output resolution), as a
// game's windowed mode does: not shrunk to fit the screen (the shrink gave 2369x1333 for 2560x1440
// on a 1440-row display, so the picture was never the size chosen). When it is bigger than the work
// area the window is placed at its top left, title bar on screen. Pixels are physical: the process is
// made per-monitor DPI aware before ReGPU opens a window (a host's window has its own awareness).
// Maximised or borderless (fullscreen) windows are left alone. Asynchronous: safe off the window's thread.
RECT MonitorRow();
bool SpanMonitors();
void SizeWindow(HWND hwnd, uint32_t width, uint32_t height)
{
    if (!hwnd || !width || !height || IsZoomed(hwnd))
        return;
    const DWORD style = DWORD(GetWindowLongW(hwnd, GWL_STYLE)), exStyle = DWORD(GetWindowLongW(hwnd, GWL_EXSTYLE));
    if (!(style & WS_CAPTION))
        return;
    RECT r{ 0, 0, LONG(width), LONG(height) };
    AdjustWindowRectExForDpi(&r, style, FALSE, exStyle, GetDpiForWindow(hwnd));
    const int w = r.right - r.left, h = r.bottom - r.top;
    MONITORINFO mi{ sizeof(mi) };
    GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
    if (SpanMonitors())
        mi.rcWork = MonitorRow(); // (triple screen: the whole row of monitors, not the nearest one)
    RECT now{};
    GetWindowRect(hwnd, &now);
    // Kept on the work area as far as it fits; its top left (the title bar) always is.
    const int x = std::max<int>(mi.rcWork.left, std::min<int>(now.left, mi.rcWork.right - w));
    const int y = std::max<int>(mi.rcWork.top, std::min<int>(now.top, mi.rcWork.bottom - h));
    const bool own = GetWindowThreadProcessId(hwnd, nullptr) == GetCurrentThreadId();
    SetWindowPos(hwnd, nullptr, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE | (own ? 0u : UINT(SWP_ASYNCWINDOWPOS)));
    Log("presenter: window client %ux%u", width, height);
}

// The swap chain's child window in its parent: fullscreen, the largest rectangle of the output's aspect
// ratio, centred (the parent's black background shows around it, and other windows cannot cover the
// bars); windowed, the whole client area (the picture stretches with the window).
void LayoutSurface()
{
    if (!g_p.surface)
        return;
    RECT c{};
    GetClientRect(g_p.window, &c);
    LONG cw = c.right, ch = c.bottom, w = cw, h = ch;
    const Settings& st = GetSettings();
    if (g_p.fullscreen && st.width && st.height)
    {
        if (int64_t(cw) * st.height > int64_t(ch) * st.width)
            w = LONG(int64_t(ch) * st.width / st.height);
        else
            h = LONG(int64_t(cw) * st.height / st.width);
    }
    const bool own = GetWindowThreadProcessId(g_p.surface, nullptr) == GetCurrentThreadId();
    SetWindowPos(g_p.surface, nullptr, (cw - w) / 2, (ch - h) / 2, w, h, SWP_NOZORDER | SWP_NOACTIVATE | (own ? 0u : UINT(SWP_ASYNCWINDOWPOS)));
}

// Fullscreen over several monitors (REGPU_TRIPLE, or REGPU_FULLSCREEN_SPAN=1; FM3 port session 39): the row of monitors
// the primary one is in (those beside it that overlap its rows, side by side), as one rectangle. Separate monitors, not
// an Eyefinity/Surround desktop, show the triple-wide picture this way. One monitor: just it.
RECT MonitorRow()
{
    struct Ctx { RECT primary{}; std::vector<RECT> all; } ctx;
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR m, HDC, LPRECT, LPARAM p) -> BOOL {
        auto& c = *reinterpret_cast<Ctx*>(p);
        MONITORINFO mi{ sizeof(mi) };
        if (GetMonitorInfoW(m, &mi))
        {
            c.all.push_back(mi.rcMonitor);
            if (mi.dwFlags & MONITORINFOF_PRIMARY)
                c.primary = mi.rcMonitor;
        }
        return TRUE;
    }, LPARAM(&ctx));
    RECT row = ctx.primary;
    // Grow left and right with monitors that touch the row's edge and share most of its height.
    for (bool grew = true; grew;)
    {
        grew = false;
        for (const RECT& r : ctx.all)
        {
            const LONG overlap = std::min(r.bottom, row.bottom) - std::max(r.top, row.top);
            if (overlap < (row.bottom - row.top) / 2)
                continue;
            if (std::abs(r.right - row.left) <= 8 && r.left < row.left) { row.left = r.left; grew = true; }
            if (std::abs(r.left - row.right) <= 8 && r.right > row.right) { row.right = r.right; grew = true; }
        }
    }
    return row;
}

bool SpanMonitors()
{
    static const bool on = [] {
        const char* t = getenv("REGPU_TRIPLE");
        const char* s = getenv("REGPU_FULLSCREEN_SPAN");
        return (t && *t && strcmp(t, "0") != 0) || (s && *s && strcmp(s, "0") != 0);
    }();
    return on;
}

// The swap chain made (again) at the surface's size: after a resize, a vsync change, or an out-of-date present. The
// queue is idle first (the old images may still be in a present).
bool CreateSwapchain()
{
    vk::Device& d = *g_p.d;
    VkSurfaceCapabilitiesKHR caps{};
    vk::Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d.physical, g_p.vkSurface, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX)
        extent = { std::clamp(GetSettings().width ? GetSettings().width : kWidth, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp(GetSettings().height ? GetSettings().height : kHeight, caps.minImageExtent.height, caps.maxImageExtent.height) };
    if (!extent.width || !extent.height)
        return false; // (minimised)
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(d.physical, g_p.vkSurface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(d.physical, g_p.vkSurface, &count, formats.data());
    VkSurfaceFormatKHR format = formats.empty() ? VkSurfaceFormatKHR{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR } : formats[0];
    for (VkFormat want : { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM })
    {
        auto it = std::find_if(formats.begin(), formats.end(),
            [&](const VkSurfaceFormatKHR& f) { return f.format == want && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; });
        if (it != formats.end())
        {
            format = *it;
            break;
        }
    }
    vkGetPhysicalDeviceSurfacePresentModesKHR(d.physical, g_p.vkSurface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(d.physical, g_p.vkSurface, &count, modes.data());
    const bool vsync = GetSettings().vsync;
    auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
    const VkPresentModeKHR mode = vsync ? VK_PRESENT_MODE_FIFO_KHR
        : has(VK_PRESENT_MODE_IMMEDIATE_KHR) ? VK_PRESENT_MODE_IMMEDIATE_KHR
        : has(VK_PRESENT_MODE_MAILBOX_KHR)   ? VK_PRESENT_MODE_MAILBOX_KHR
                                             : VK_PRESENT_MODE_FIFO_KHR;
    VkSwapchainCreateInfoKHR ci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    ci.surface = g_p.vkSurface;
    ci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount)
        ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = format.format;
    ci.imageColorSpace = format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = g_p.swapchain;
    {
        std::lock_guard lock(d.queueLock);
        vkQueueWaitIdle(d.queue);
    }
    VkSwapchainKHR made = VK_NULL_HANDLE;
    const VkResult r = vkCreateSwapchainKHR(d.device, &ci, nullptr, &made);
    for (VkImageView v : g_p.views)
        vkDestroyImageView(d.device, v, nullptr);
    for (VkSemaphore sem : g_p.rendered)
        vkDestroySemaphore(d.device, sem, nullptr);
    g_p.views.clear();
    g_p.rendered.clear();
    g_p.images.clear();
    if (g_p.swapchain)
        vkDestroySwapchainKHR(d.device, g_p.swapchain, nullptr);
    g_p.swapchain = VK_NULL_HANDLE;
    g_p.imageIndex = UINT32_MAX;
    g_p.submitted = false;
    if (r != VK_SUCCESS)
    {
        Log("presenter: swap chain %ux%u not made: %s", extent.width, extent.height, vk::ResultName(r));
        return false;
    }
    g_p.swapchain = made;
    g_p.format = format.format;
    g_p.extent = extent;
    g_p.vsync = vsync;
    vkGetSwapchainImagesKHR(d.device, made, &count, nullptr);
    g_p.images.resize(count);
    vkGetSwapchainImagesKHR(d.device, made, &count, g_p.images.data());
    for (VkImage image : g_p.images)
    {
        VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format.format;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkImageView view = VK_NULL_HANDLE;
        vk::Check(vkCreateImageView(d.device, &vi, nullptr, &view), "vkCreateImageView(swap chain)");
        g_p.views.push_back(view);
        VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkSemaphore sem = VK_NULL_HANDLE;
        vk::Check(vkCreateSemaphore(d.device, &si, nullptr, &sem), "vkCreateSemaphore(rendered)");
        g_p.rendered.push_back(sem);
    }
    Log("presenter: swap chain %ux%u, %u images, format %d, %s", extent.width, extent.height, count, int(format.format),
        mode == VK_PRESENT_MODE_FIFO_KHR ? "vsync" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate (tearing)" : "mailbox");
    return true;
}

// The next image of the swap chain for the frame being made (the one already acquired, if not yet presented).
bool Acquire()
{
    vk::Device& d = *g_p.d;
    if (g_p.imageIndex != UINT32_MAX)
        return true;
    if (!g_p.swapchain || g_p.recreate || g_p.vsync != GetSettings().vsync)
    {
        g_p.recreate = false;
        if (!CreateSwapchain())
            return false;
    }
    for (int attempt = 0; attempt < 2; attempt++)
    {
        const uint32_t slot = g_p.acquireNext;
        if (g_p.acquiredUse[slot])
            vk::WaitFor(d, g_p.acquiredUse[slot]); // (its last wait has run: it can be signalled again)
        uint32_t index = 0;
        const VkResult r = vkAcquireNextImageKHR(d.device, g_p.swapchain, UINT64_MAX, g_p.acquired[slot], VK_NULL_HANDLE, &index);
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR)
        {
            if (r == VK_SUBOPTIMAL_KHR)
                g_p.recreate = true; // (after this frame)
            g_p.imageIndex = index;
            g_p.pendingAcquire = int(slot);
            g_p.acquireNext = (slot + 1) % kAcquire;
            return true;
        }
        if (r != VK_ERROR_OUT_OF_DATE_KHR || !CreateSwapchain())
        {
            Log("presenter: vkAcquireNextImageKHR: %s", vk::ResultName(r));
            return false;
        }
    }
    return false;
}

void ImageBarrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 srcStages, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStages, VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStages;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStages;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

// A blit of src's rectangle (sx0, sy0)-(sx1, sy1) to the back buffer's (dx0, dy0)-(dx1, dy1); both in their transfer
// layouts. Linear filtering when the size changes.
void Blit(VkCommandBuffer cb, VkImage src, VkImage dst, int32_t sx0, int32_t sy0, int32_t sx1, int32_t sy1, int32_t dx0, int32_t dy0, int32_t dx1,
    int32_t dy1)
{
    if (sx1 <= sx0 || sy1 <= sy0 || dx1 <= dx0 || dy1 <= dy0)
        return;
    VkImageBlit b{};
    b.srcSubresource = b.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    b.srcOffsets[0] = { sx0, sy0, 0 };
    b.srcOffsets[1] = { sx1, sy1, 1 };
    b.dstOffsets[0] = { dx0, dy0, 0 };
    b.dstOffsets[1] = { dx1, dy1, 1 };
    const bool scaled = sx1 - sx0 != dx1 - dx0 || sy1 - sy0 != dy1 - dy0;
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, scaled ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// Presents the acquired image (its frame submitted: PresenterSubmit).
void PresentAcquired()
{
    std::lock_guard swap(g_p.swapMutex);
    if (g_p.imageIndex == UINT32_MAX || !g_p.submitted)
        return;
    vk::Device& d = *g_p.d;
    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &g_p.rendered[g_p.imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &g_p.swapchain;
    pi.pImageIndices = &g_p.imageIndex;
    VkResult r;
    {
        std::lock_guard lock(d.queueLock);
        r = vkQueuePresentKHR(d.queue, &pi);
    }
    g_p.imageIndex = UINT32_MAX;
    g_p.submitted = false;
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        g_p.recreate = true;
    else if (r != VK_SUCCESS)
        Log("presenter: vkQueuePresentKHR: %s", vk::ResultName(r));
}

void CreateWindowAndDevice()
{
    // Physical pixels for the window ReGPU opens (an unaware window at 125% scaling would be stretched).
    if (!GetSettings().hostWindow)
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"regpu";
    wc.hbrBackground = HBRUSH(GetStockObject(BLACK_BRUSH));
    wc.hIcon = HICON(LoadImageW(wc.hInstance, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR));
    RegisterClassW(&wc);
    // The swap chain is the chosen resolution (the game's image is centred in it with black
    // bars when an aspect ratio is forced). Windowed: the client area is exactly that size
    // (SizeWindow). Fullscreen: a borderless window over the primary display. Either way the
    // swap chain stretches to the window.
    const Settings& settings = GetSettings();
    const uint32_t outW = settings.width, outH = settings.height;
    if (settings.hostWindow)
    {
        // The host's window (Config::hostWindow): the swap chain goes straight onto it, and it
        // stretches with it. The host owns its messages (close, resize, fullscreen).
        g_p.window = HWND(settings.hostWindow);
        Log("presenter: presenting into the host's window %p", settings.hostWindow);
    }
    else if (settings.fullscreen)
    {
        // Borderless window over the whole display; the picture is its child window (LayoutSurface). Triple screen:
        // over the row of monitors (MonitorRow).
        RECT area{ 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
        if (SpanMonitors())
        {
            area = MonitorRow();
            Log("presenter: fullscreen over the monitors at %ld,%ld, %ldx%ld", area.left, area.top, area.right - area.left, area.bottom - area.top);
        }
        const int sw = area.right - area.left, sh = area.bottom - area.top;
        g_p.fullscreen = true;
        g_p.window = CreateWindowExW(0, L"regpu", GetSettings().windowTitle.c_str(), WS_POPUP | WS_VISIBLE, area.left, area.top, sw, sh, nullptr, nullptr,
            wc.hInstance, nullptr);
    }
    else
    {
        g_p.window = CreateWindowW(L"regpu", GetSettings().windowTitle.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
            CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, wc.hInstance, nullptr);
        // (Sized from the work area's corner; triple screen: from the monitor row's left edge, so a three-screen-wide window
        // starts on the left monitor.)
        const LONG left = SpanMonitors() ? MonitorRow().left : 0;
        SetWindowPos(g_p.window, nullptr, left, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
        SizeWindow(g_p.window, outW, outH);
        ShowWindow(g_p.window, SW_SHOW);
    }
    // The swap chain goes on the window itself, except fullscreen at an aspect ratio other than the display's:
    // there the picture is a child window between black bars (LayoutSurface). A child covering the whole window
    // showed black in fullscreen while focused on an AMD card (FM2 report #59). Switching live (SetFullscreen)
    // into fullscreen at another aspect ratio stretches the picture until the next start.
    if (!settings.hostWindow && settings.fullscreen)
    {
        RECT client{};
        GetClientRect(g_p.window, &client);
        if (int64_t(client.right) * outH != int64_t(client.bottom) * outW)
        {
            g_p.surface = CreateWindowExW(0, L"regpu", nullptr, WS_CHILD | WS_VISIBLE, 0, 0, 1, 1, g_p.window, nullptr, wc.hInstance, nullptr);
            LayoutSurface();
        }
    }
    Log("presenter: picture in %s", g_p.surface ? "a child window (black bars)" : "the game window");

    // The device (ReGPU's own, or the host's) and the swap chain on the window.
    vk::Device& d = RenderVkDevice();
    g_p.d = &d;
    if (!d.swapchain)
        Fatal("presenter: the Vulkan device has no swap chain support (VK_KHR_swapchain)");
    VkWin32SurfaceCreateInfoKHR si{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    si.hinstance = GetModuleHandleW(nullptr);
    si.hwnd = g_p.surface ? g_p.surface : g_p.window;
    vk::Check(vkCreateWin32SurfaceKHR(d.instance, &si, nullptr, &g_p.vkSurface), "vkCreateWin32SurfaceKHR");
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(d.physical, d.queueFamily, g_p.vkSurface, &supported);
    if (!supported)
        Fatal("presenter: the graphics queue can't present to this window");
    for (VkSemaphore& sem : g_p.acquired)
    {
        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        vk::Check(vkCreateSemaphore(d.device, &sci, nullptr, &sem), "vkCreateSemaphore(acquired)");
    }
    {
        std::lock_guard swap(g_p.swapMutex);
        CreateSwapchain();
    }
    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = d.queueFamily;
    vk::Check(vkCreateCommandPool(d.device, &pci, nullptr, &g_p.pool), "vkCreateCommandPool(presenter)");
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_p.pool;
    ai.commandBufferCount = kBuffers;
    vk::Check(vkAllocateCommandBuffers(d.device, &ai, g_p.cbs), "vkAllocateCommandBuffers(presenter)");
    regpu::xr::Start(d); // (REGPU_VR: the headset thread, on this device and queue)
}

// The presenter's own frame (before the renderer draws): the submitted CPU image centred, or a pulsing clear.
void DrawFrame(uint64_t frame)
{
    vk::Device& d = *g_p.d;
    std::lock_guard swap(g_p.swapMutex);
    const UINT i = g_p.ownFrame++ % kBuffers;
    if (g_p.cbDone[i])
        vk::WaitFor(d, g_p.cbDone[i]);
    VkCommandBuffer cb = g_p.cbs[i];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    // The submitted image (any size: the parts-bin backend sends the output size) goes to the GPU through this frame's
    // upload buffer.
    bool copied = false;
    uint32_t imageW = 0, imageH = 0;
    {
        std::lock_guard lock(g_p.mutex);
        imageW = g_p.imageWidth;
        imageH = g_p.imageHeight;
        const uint64_t bytes = uint64_t(imageW) * imageH * 4;
        if (bytes && g_p.image.size() == bytes)
        {
            if (g_p.upload[i].size < bytes)
            {
                vk::Destroy(d, g_p.upload[i]); // (its last use was waited for above)
                g_p.upload[i] = vk::CreateBuffer(d, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::Memory::Upload, "presenter upload");
            }
            if (g_p.upload[i].mapped)
            {
                memcpy(g_p.upload[i].mapped, g_p.image.data(), bytes);
                copied = true;
            }
        }
    }
    if (!Acquire())
    {
        vkEndCommandBuffer(cb);
        return;
    }
    const VkImage back = g_p.images[g_p.imageIndex];
    ImageBarrier(cb, back, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
        VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const float pulse = 0.15f + 0.1f * float((frame % 120) < 60 ? frame % 60 : 60 - frame % 60) / 60.0f;
    VkClearColorValue clear{};
    if (!copied)
    {
        clear.float32[0] = 0.05f, clear.float32[1] = pulse, clear.float32[2] = 0.12f;
    }
    clear.float32[3] = 1.0f;
    const VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cb, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    if (copied)
    {
        if (!g_p.cpuImage || g_p.cpuImage.desc.width != imageW || g_p.cpuImage.desc.height != imageH)
        {
            vk::WaitFor(d, d.submitted.load()); // (rare: the image changed size)
            vk::Destroy(d, g_p.cpuImage);
            vk::ImageDesc desc;
            desc.width = imageW;
            desc.height = imageH;
            desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            desc.name = "presenter image";
            g_p.cpuImage = vk::CreateImage(d, desc);
        }
        ImageBarrier(cb, g_p.cpuImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy c{};
        c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        c.imageExtent = { imageW, imageH, 1 };
        vkCmdCopyBufferToImage(cb, g_p.upload[i].buffer, g_p.cpuImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ImageBarrier(cb, g_p.cpuImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        // Centred and cropped to the back buffer.
        const uint32_t w = std::min(imageW, g_p.extent.width), h = std::min(imageH, g_p.extent.height);
        const int32_t sx = int32_t(imageW - w) / 2, sy = int32_t(imageH - h) / 2, dx = int32_t(g_p.extent.width - w) / 2,
                      dy = int32_t(g_p.extent.height - h) / 2;
        Blit(cb, g_p.cpuImage.image, back, sx, sy, sx + int32_t(w), sy + int32_t(h), dx, dy, dx + int32_t(w), dy + int32_t(h));
    }
    ImageBarrier(cb, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    vkEndCommandBuffer(cb);
    g_p.cbDone[i] = PresenterSubmitValue(cb);
    PresentAcquired();
}

void RenderThread()
{
    SetThreadDescription(GetCurrentThread(), L"presenter");
    CreateWindowAndDevice();
    g_p.ready = true;
    Log("presenter: window and Vulkan swap chain ready");
    for (;;)
    {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        uint64_t frame;
        {
            std::unique_lock lock(g_p.mutex);
            g_p.cv.wait_for(lock, std::chrono::milliseconds(16), [] { return g_p.pendingFrame != g_p.presentedFrame; });
            frame = g_p.pendingFrame;
            if (frame == g_p.presentedFrame || g_rendererActive)
                continue;
            g_p.presentedFrame = frame;
            if (g_p.claimed)
                continue;
        }
        DrawFrame(frame);
    }
}
}

void PresenterInit()
{
    // A host's device (Config::hostInstance ... hostQueue) is taken now, with or without a host presenter and headless
    // too. Without one, ReGPU's own device is made on first use.
    if (GetSettings().hostDevice || GetSettings().hostQueue)
        RenderVkDevice();
    if (getenv("REGPU_HEADLESS"))
        return;
    // Host::presentImage: the host presents, so no thread, window or swap chain here. Its window
    // (Config::hostWindow) still answers PresenterHasFocus and SetWindowSize. Decided once, here:
    // REGPU_NO_HOST_PRESENT is read at Start only.
    if (regpu::GetHost().presentImage)
    {
        static LiveSwitch off("REGPU_NO_HOST_PRESENT");
        if (!off)
        {
            // The host's present is ordered against ReGPU's writes by going on the queue ReGPU renders on, so that
            // queue (and device) must be the host's.
            if (!GetSettings().hostDevice || !GetSettings().hostQueue)
                Fatal("presenter: Host::presentImage needs the host's Vulkan device (Config::hostInstance, hostPhysicalDevice, hostDevice, hostQueue)");
            g_hostPresents = true;
            g_p.window = HWND(GetSettings().hostWindow);
            Log("presenter: the host presents (no swap chain)");
            return;
        }
    }
    std::thread(RenderThread).detach();
}

bool PresenterHostPresents()
{
    return g_hostPresents;
}

void PresenterSubmitFrame(uint64_t frame)
{
    {
        std::lock_guard lock(g_p.mutex);
        g_p.pendingFrame = frame;
    }
    g_p.cv.notify_one();
}

bool PresenterHasFocus()
{
    return !g_p.window || GetForegroundWindow() == g_p.window;
}

// The swap chain to a new output size (the renderer's ApplyOutputSize, the host GPU idle): made again at the next
// frame (it follows the window's size; the picture is scaled to it).
void PresenterResize(uint32_t width, uint32_t height)
{
    if (!g_p.swapchain)
        return;
    g_p.recreate = true;
    Log("presenter: output %ux%u", width, height);
    LayoutSurface(); // (fullscreen: the bars follow the new aspect ratio)
}

void regpu::SetFullscreen(bool on)
{
    // The host presents: ReGPU has no swap chain or window of its own to change. The early return below
    // already makes the call do nothing then (the window is null or the host's); this one is here for its
    // log line, so a host can see why nothing happened. Config::fullscreen is left as it is.
    if (g_hostPresents)
    {
        Log("presenter: SetFullscreen(%s) ignored: the host presents", on ? "true" : "false");
        return;
    }
    if (!g_p.window || GetSettings().hostWindow || g_p.fullscreen == on)
        return;
    g_p.fullscreen = on;
    GetSettings().fullscreen = on;
    if (on)
    {
        MONITORINFO mi{ sizeof(mi) };
        GetMonitorInfoW(MonitorFromWindow(g_p.window, MONITOR_DEFAULTTONEAREST), &mi);
        if (IsZoomed(g_p.window))
            ShowWindowAsync(g_p.window, SW_RESTORE);
        SetWindowLongW(g_p.window, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(g_p.window, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
            mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED | SWP_ASYNCWINDOWPOS);
    }
    else
    {
        SetWindowLongW(g_p.window, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
        SetWindowPos(g_p.window, nullptr, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_ASYNCWINDOWPOS);
        SizeWindow(g_p.window, GetSettings().width, GetSettings().height);
    }
    Log("presenter: %s (live)", on ? "fullscreen" : "windowed");
}

void regpu::SetWindowSize(uint32_t width, uint32_t height)
{
    SizeWindow(g_p.window, width, height);
}

void PresenterSubmitImage(uint32_t width, uint32_t height, std::vector<uint8_t> rgba)
{
    PresenterExchangeImage(width, height, std::move(rgba));
}

// The Vulkan backend's frames, read back to the CPU (any size: centred in the back buffer). Returns the image shown
// before, whose storage the caller reuses for its next frame (from the FM2 port).
std::vector<uint8_t> PresenterExchangeImage(uint32_t width, uint32_t height, std::vector<uint8_t> rgba)
{
    if (!width || !height || rgba.size() != size_t(width) * height * 4)
        return rgba;
    {
        std::lock_guard lock(g_p.mutex);
        g_p.image.swap(rgba);
        g_p.imageWidth = width;
        g_p.imageHeight = height;
        g_p.pendingFrame++;
    }
    g_p.cv.notify_one();
    return rgba;
}

// The window for a Vulkan swap chain, or null when it can't have one: headless, or the D3D12 swap chain already
// holds it (DXGI's flip model owns the window; the Vulkan backend then hands its frames over, PresenterExchangeImage).
void* PresenterClaimWindow()
{
    if (getenv("REGPU_HEADLESS"))
        return nullptr;
    for (int i = 0; i < 2000 && !g_p.ready; i++) // the window is made on the presenter's thread
        Sleep(5);
    if (!g_p.ready || g_p.swapchain)
        return nullptr;
    g_p.claimed = true;
    return g_p.surface ? g_p.surface : g_p.window;
}

void PresenterReleaseWindow()
{
    g_p.claimed = false;
}

// A frame shown by the Vulkan swap chain (counted with the presenter's own; the window title's frame rate).
std::atomic<uint32_t> g_vkPresents{ 0 };
void PresenterCountFrame()
{
    g_vkPresents++;
}

vk::Device& RenderVkDevice()
{
    static vk::Device device;
    static std::once_flag once;
    std::call_once(once, [] {
        const Settings& st = GetSettings();
        if (st.hostInstance || st.hostPhysicalDevice || st.hostDevice || st.hostQueue)
        {
            if (!st.hostInstance || !st.hostPhysicalDevice || !st.hostDevice || !st.hostQueue)
                Fatal("presenter: Config::hostInstance, hostPhysicalDevice, hostDevice and hostQueue are set together or not at all");
            vk::HostDevice host;
            host.instance = VkInstance(st.hostInstance);
            host.physical = VkPhysicalDevice(st.hostPhysicalDevice);
            host.device = VkDevice(st.hostDevice);
            host.queue = VkQueue(st.hostQueue);
            host.queueFamily = st.hostQueueFamily;
            host.extensions = st.hostDeviceExtensions;
            vk::CreateDevice(device, regpu::GetConfig().userDir, &host);
        }
        else
            vk::CreateDevice(device, regpu::GetConfig().userDir);
    });
    return device;
}

void PresenterUseRenderer()
{
    g_rendererActive = true;
}

uint64_t PresenterSubmitValue(VkCommandBuffer cb)
{
    std::lock_guard swap(g_p.swapMutex);
    vk::Device& d = *g_p.d;
    if (g_p.pendingAcquire < 0)
        return vk::Submit(d, &cb, 1);
    // The copy waits for the image to be free (the acquire) and signals the present's semaphore.
    const vk::SemaphoreUse wait{ g_p.acquired[g_p.pendingAcquire], 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
    const vk::SemaphoreUse signal{ g_p.rendered[g_p.imageIndex], 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
    const uint64_t value = vk::Submit(d, &cb, 1, &wait, 1, &signal, 1);
    g_p.acquiredUse[g_p.pendingAcquire] = value;
    g_p.pendingAcquire = -1;
    g_p.submitted = true;
    return value;
}

void PresenterSubmit(VkCommandBuffer cb)
{
    PresenterSubmitValue(cb);
}

bool PresenterRecordCopy(VkCommandBuffer cb, const vk::Image* src, const vk::Image* overlay, uint32_t overlayW, uint32_t overlayH, const vk::Image* timer,
    uint32_t timerW, uint32_t timerH, const vk::Image* fps, uint32_t fpsW, uint32_t fpsH, PresenterGrade grade, void* gradeCtx)
{
    if (!g_p.ready || !src)
        return false;
    std::lock_guard swap(g_p.swapMutex);
    if (!Acquire())
        return false;
    vk::Device& d = *g_p.d;
    const VkImage back = g_p.images[g_p.imageIndex];
    const uint32_t bw = g_p.extent.width, bh = g_p.extent.height;
    // The picture on the output-size canvas (the image is the render size; with a forced aspect ratio it is smaller and
    // goes in the middle, on black); the canvas is stretched to the back buffer (the window's size), as D3D12's swap
    // chain was stretched to its window.
    const uint32_t cw = GetSettings().width ? GetSettings().width : bw, ch = GetSettings().height ? GetSettings().height : bh;
    const uint32_t sw = src->desc.width, sh = src->desc.height;
    const uint32_t shownW = std::min(sw, cw), shownH = std::min(sh, ch);
    const uint32_t offsetX = (cw - shownW) / 2, offsetY = (ch - shownH) / 2;
    auto X = [&](uint32_t x) { return int32_t(uint64_t(x) * bw / cw); };
    auto Y = [&](uint32_t y) { return int32_t(uint64_t(y) * bh / ch); };
    ImageBarrier(cb, back, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
        VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    if (offsetX || offsetY)
    {
        VkClearColorValue black{};
        black.float32[3] = 1.0f;
        const VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    const bool canvasSize = !offsetX && !offsetY && sw == cw && sh == ch;
    if (grade && canvasSize && bw == cw && bh == ch)
    {
        // The gamma pass writes the back buffer itself: one full-size write and read less than grading into an image
        // and copying that (session 33: ~4% at 4K 4x).
        ImageBarrier(cb, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        grade(cb, g_p.views[g_p.imageIndex], g_p.format, bw, bh, gradeCtx);
        ImageBarrier(cb, back, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    else if (grade && canvasSize)
    {
        // Graded at the picture's size, then scaled to the back buffer.
        if (!g_p.gradeTarget || g_p.gradeTarget.desc.width != sw || g_p.gradeTarget.desc.height != sh)
        {
            {
                std::lock_guard lock(d.queueLock);
                vkQueueWaitIdle(d.queue); // (rare: the size changed; the old one may be in flight)
            }
            vk::Destroy(d, g_p.gradeTarget);
            vk::ImageDesc desc;
            desc.width = sw;
            desc.height = sh;
            desc.format = VK_FORMAT_R8G8B8A8_UNORM;
            desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            desc.name = "graded picture";
            g_p.gradeTarget = vk::CreateImage(d, desc);
        }
        ImageBarrier(cb, g_p.gradeTarget.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        grade(cb, g_p.gradeTarget.target, VK_FORMAT_R8G8B8A8_UNORM, sw, sh, gradeCtx);
        ImageBarrier(cb, g_p.gradeTarget.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
        Blit(cb, g_p.gradeTarget.image, back, 0, 0, int32_t(sw), int32_t(sh), 0, 0, int32_t(bw), int32_t(bh));
    }
    else
        Blit(cb, src->image, back, 0, 0, int32_t(shownW), int32_t(shownH), X(offsetX), Y(offsetY), X(offsetX + shownW), Y(offsetY + shownH));
    if (overlay)
    {
        const uint32_t x = offsetX + 24, y = offsetY + 24 + (timer ? timerH : 0);
        Blit(cb, overlay->image, back, 0, 0, int32_t(overlayW), int32_t(overlayH), X(x), Y(y), X(x + overlayW), Y(y + overlayH));
    }
    if (timer && timerW <= cw && timerH <= ch)
        Blit(cb, timer->image, back, 0, 0, int32_t(timerW), int32_t(timerH), X(offsetX), Y(offsetY), X(offsetX + timerW), Y(offsetY + timerH));
    if (fps && fpsW <= shownW && fpsH + 12 <= ch)
    {
        // Top centre of the picture.
        const uint32_t x = offsetX + (shownW - fpsW) / 2, y = offsetY + 12;
        Blit(cb, fps->image, back, 0, 0, int32_t(fpsW), int32_t(fpsH), X(x), Y(y), X(x + fpsW), Y(y + fpsH));
    }
    regpu::xr::CaptureFrame(cb, back, bw, bh, g_p.format); // (REGPU_VR: the finished frame for the headset's virtual screen)
    ImageBarrier(cb, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    return true;
}

uint32_t RegpuFrameCap();
bool RegpuPerfLiteOn();

// The window, the picture's own window and the swap chain, in physical pixels: logged at the first present and whenever
// they change (session 39: a triple-screen picture shown at a third of its size on the centre screen).
void LogWindowSizes()
{
    static RECT lastWin{}, lastSurf{};
    static UINT lastW = 0, lastH = 0;
    if (!g_p.window || !g_p.swapchain)
        return;
    RECT win{}, surf{};
    GetWindowRect(g_p.window, &win);
    GetWindowRect(g_p.surface ? g_p.surface : g_p.window, &surf);
    const UINT w = g_p.extent.width, h = g_p.extent.height;
    if (memcmp(&win, &lastWin, sizeof(win)) == 0 && memcmp(&surf, &lastSurf, sizeof(surf)) == 0 && w == lastW && h == lastH)
        return;
    lastWin = win, lastSurf = surf, lastW = w, lastH = h;
    Log("presenter: window %ld,%ld %ldx%ld, picture %ld,%ld %ldx%ld, swap chain %ux%u, DPI %u", win.left, win.top, win.right - win.left, win.bottom - win.top,
        surf.left, surf.top, surf.right - surf.left, surf.bottom - surf.top, w, h, GetDpiForWindow(g_p.window));
}

void PresenterPresent()
{
    using Clock = std::chrono::steady_clock;
    {
        static Clock::time_point lastCheck{};
        if (Clock::now() - lastCheck > std::chrono::seconds(5))
        {
            lastCheck = Clock::now();
            LogWindowSizes();
        }
    }
    // REGPU_PRESENT_PACE=1 (live, session 33 of the FM3 port): presents held to an even spacing at the frame cap. The cap
    // gates the title's VdSwap, at the start of the pipeline; the frame then reaches here after the command processor,
    // the draw thread and the replay thread, whose time varies with the frame's load: the presents came unevenly (the
    // 1% slowest present gaps ~15% over the cap's at 1440p even in light seconds). Here a frame waits for its slot
    // (high-resolution timer, then a short spin); one that comes late starts the spacing again from now.
    static LiveSwitch pace("REGPU_PRESENT_PACE");
    static Clock::time_point slot{};
    if (pace && g_p.ready && !GetSettings().vsync)
    {
        const auto interval = std::chrono::microseconds(1000000 / std::max(RegpuFrameCap(), 30u));
        const auto now = Clock::now();
        if (slot > now && slot - now <= interval)
            PreciseSleepUntil(slot, 300);
        slot = std::max(slot + interval, Clock::now() - interval / 4); // (a little catch-up; no burst after a late one)
    }
    // Present gaps for the light performance log: their 99th percentile once a second (what reaches the screen; the
    // perf lite line's frame times are the draw thread's).
    if (RegpuPerfLiteOn())
    {
        static std::vector<float> gaps;
        static Clock::time_point last{}, windowStart = Clock::now();
        const auto now = Clock::now();
        if (last != Clock::time_point{})
            gaps.push_back(std::chrono::duration<float, std::milli>(now - last).count());
        last = now;
        if (now - windowStart >= std::chrono::seconds(1) && gaps.size() > 10)
        {
            std::vector<float> sorted = gaps;
            const size_t k = std::min(sorted.size() - 1, size_t(double(sorted.size()) * 0.99));
            std::nth_element(sorted.begin(), sorted.begin() + ptrdiff_t(k), sorted.end());
            const float p99 = sorted[k], worst = *std::max_element(gaps.begin(), gaps.end());
            double sum = 0;
            for (float g : gaps)
                sum += g;
            Log("perf lite: presents %zu, gap avg %.3f ms, p99 %.3f ms, max %.2f ms%s", gaps.size(), sum / double(gaps.size()), p99, worst,
                pace ? " (paced)" : "");
            gaps.clear();
            windowStart = now;
        }
    }
    // Vsync (FIFO) or not (immediate: tearing allowed) is the swap chain's present mode: Acquire makes it again when
    // the setting changes.
    if (g_p.ready)
        PresentAcquired();
}

uint32_t DisplayRefreshHz()
{
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) && mode.dmDisplayFrequency > 1)
        return mode.dmDisplayFrequency;
    return 60;
}
