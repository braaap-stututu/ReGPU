// Benchmark recording and its report (regpu.h BenchmarkRecord / BenchmarkWriteReport), and small public helpers
// (CompileIdle, WritePng). From the FM4 port's ReGPU; the frame times are the presenter's present gaps
// (BenchmarkFrame, called at every present).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <windows.h>

#include "gpu/renderer.h"
#include "log.h"
#include "png.h"
#include "regpu/regpu.h"
#include "settings.h"

const char* PresenterGpuName();     // render/presenter.cpp ("" before the device exists)
uint64_t PresenterGpuMemoryMB();    // ... its device-local memory

namespace
{
std::atomic<bool> g_recording{ false };
std::mutex g_mutex;
std::vector<float> g_frames; // ms
}

// The presenter, at every present: the gap since the last one.
void BenchmarkFrame(double ms)
{
    if (!g_recording.load(std::memory_order_relaxed))
        return;
    std::lock_guard lock(g_mutex);
    g_frames.push_back(float(ms));
}

void regpu::BenchmarkRecord(bool on)
{
    g_recording.store(on, std::memory_order_relaxed);
}

bool regpu::CompileIdle()
{
    return RendererCompileIdle();
}

bool regpu::WritePng(const std::filesystem::path& path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgba)
{
    return ::WritePng(path, width, height, rgba);
}

void regpu::BenchmarkWriteReport(const char* path)
{
    std::vector<float> frames;
    {
        std::lock_guard lock(g_mutex);
        frames = g_frames;
    }
    std::string cpu = "?";
    {
        char name[256];
        DWORD size = sizeof(name);
        if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString", RRF_RT_REG_SZ, nullptr,
                name, &size) == ERROR_SUCCESS)
            cpu = name;
        while (!cpu.empty() && cpu.back() == ' ')
            cpu.pop_back();
    }
    uint32_t cores = 0;
    {
        DWORD bytes = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
        std::vector<uint8_t> buffer(bytes);
        if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &bytes))
            for (DWORD at = 0; at < bytes; at += reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at)->Size)
                cores++;
    }
    MEMORYSTATUSEX memory{ sizeof(memory) };
    GlobalMemoryStatusEx(&memory);
    std::string gpu = PresenterGpuName();
    if (gpu.empty())
        gpu = "?";
    else
        gpu += " (" + std::to_string(PresenterGpuMemoryMB()) + " MB)";
    const Settings& st = GetSettings();
    static const char* upscalers[] = { "off", "FSR 1", "FSR 3", "DLSS", "FSR 4", "XeSS" };
    static const char* qualities[] = { "native (AA)", "quality", "balanced", "performance", "ultra performance" };
    std::string upscaler = upscalers[std::min<uint32_t>(uint32_t(st.upscaler), 5)];
    if (st.upscaler != Settings::Upscaler::Off)
        upscaler += std::string(", ") + qualities[std::min<uint32_t>(st.quality, 4)];

    FILE* out = fopen(path, "w");
    if (!out)
        return;
    fprintf(out, "ReGPU benchmark\n\n");
    fprintf(out, "CPU:        %s (%u cores, %u threads)\n", cpu.c_str(), cores, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    fprintf(out, "GPU:        %s (Vulkan)\n", gpu.c_str());
    fprintf(out, "RAM:        %.0f GB\n", memory.ullTotalPhys / 1073741824.0);
    fprintf(out, "Resolution: %ux%u%s, MSAA %ux\n", st.width, st.height, st.fullscreen ? " fullscreen" : " windowed", st.msaa);
    fprintf(out, "Upscaler:   %s\n\n", upscaler.c_str());
    if (frames.size() < 100)
    {
        fprintf(out, "Not enough frames recorded (%zu): the run ended too early. Start the game once first\n"
                     "(the first start compiles shaders), then run the benchmark again.\n", frames.size());
        fclose(out);
        return;
    }
    double total = 0;
    for (float f : frames)
        total += f;
    std::vector<float> sorted = frames;
    std::sort(sorted.begin(), sorted.end());
    // Lows: the average frame rate over the slowest 1% / 0.1% of frames.
    auto low = [&](double fraction) {
        const size_t n = std::max<size_t>(1, size_t(sorted.size() * fraction));
        double sum = 0;
        for (size_t i = sorted.size() - n; i < sorted.size(); i++)
            sum += sorted[i];
        return 1000.0 * double(n) / sum;
    };
    fprintf(out, "Average:    %.1f fps (%zu frames in %.1f s)\n", 1000.0 * double(frames.size()) / total, frames.size(), total / 1000.0);
    fprintf(out, "1%% low:     %.1f fps\n", low(0.01));
    fprintf(out, "0.1%% low:   %.1f fps\n", low(0.001));
    fprintf(out, "Slowest:    %.1f ms\n", double(sorted.back()));
    // Pacing: how much consecutive frames differ (0 for perfectly even frames).
    double jitter = 0;
    for (size_t i = 1; i < frames.size(); i++)
        jitter += std::fabs(double(frames[i]) - double(frames[i - 1]));
    fprintf(out, "Pacing:     %.2f ms average change between frames, median frame %.2f ms\n", jitter / double(frames.size() - 1),
        double(sorted[sorted.size() / 2]));
    // Timeline: average and slowest frame per 5 s (does the first stretch run slower?).
    fprintf(out, "Timeline:  ");
    double windowMs = 0, windowMax = 0;
    size_t windowFrames = 0;
    for (float f : frames)
    {
        windowMs += f;
        windowMax = std::max(windowMax, double(f));
        windowFrames++;
        if (windowMs >= 5000.0)
        {
            fprintf(out, " %.0f fps (max %.1f ms) |", 1000.0 * double(windowFrames) / windowMs, windowMax);
            windowMs = windowMax = 0;
            windowFrames = 0;
        }
    }
    fprintf(out, "\n");
    fclose(out);
    Log("benchmark: report written to %s", path);
}
