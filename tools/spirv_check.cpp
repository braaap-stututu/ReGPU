// regpu_spirv_check [--no-dxc] [--keep <dir>] <shader.bin>...: the experimental direct Xenos-to-SPIR-V translator
// (src/gpu/spirv/translate.h) over captured microcode (vs_*.bin / ps_*.bin, as xenos_dis reads them). For each
// shader: translate directly; validate the module with spirv-val (REGPU_SPIRV_VAL, else the Vulkan SDK's); and, for the
// shaders it translated, time the HLSL path (TranslateShader + AdaptHlsl + DXC, no cache) for comparison. Prints
// coverage, fallbacks grouped by reason, validation failures and average times.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "gpu/hlsl.h"
#include "gpu/spirv/translate.h"
#include "gpu/ucode.h"
#include "gpu/vk/vk_shader.h"

namespace
{
std::string Validator()
{
    if (const char* v = getenv("REGPU_SPIRV_VAL"))
        return v;
    std::error_code ec;
    if (const char* sdk = getenv("VULKAN_SDK"); sdk && std::filesystem::exists(std::filesystem::path(sdk) / "Bin" / "spirv-val.exe", ec))
        return (std::filesystem::path(sdk) / "Bin" / "spirv-val.exe").string();
    return "C:\\VulkanSDK\\1.4.363.1\\Bin\\spirv-val.exe";
}

double Ms(std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
}

int main(int argc, char** argv)
{
    bool dxc = true;
    SpirvOptions options; // --motion: the motion-vector variants (SpirvOptions::motion); --no-kill: alpha-test-free
    std::filesystem::path keep;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--no-dxc"))
            dxc = false;
        else if (!strcmp(argv[i], "--motion"))
            options.motion = true;
        else if (!strcmp(argv[i], "--no-kill"))
            options.noAlphaTest = true;
        else if (!strcmp(argv[i], "--keep") && i + 1 < argc)
            keep = argv[++i];
        else
            files.push_back(argv[i]);
    }
    const std::string val = Validator();
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "regpu_spirv_check";
    std::filesystem::create_directories(tmp);
    if (!keep.empty())
        std::filesystem::create_directories(keep);

    int translated = 0, fallback = 0, invalid = 0, hlslFailed = 0;
    std::map<std::string, int> reasons;
    double directMs = 0, hlslMs = 0;
    int hlslCount = 0;
    for (const std::string& file : files)
    {
        std::ifstream f(file, std::ios::binary);
        std::vector<char> bytes{ std::istreambuf_iterator<char>(f), {} };
        const std::string name = std::filesystem::path(file).filename().string();
        const bool pixel = name.find("ps_") != std::string::npos;
        const auto shader = ucode::Parse(reinterpret_cast<const uint32_t*>(bytes.data()), uint32_t(bytes.size() / 4), pixel);
        const TranslatedShader info = TranslateShader(shader);
        if (!info.ok)
        {
            hlslFailed++;
            printf("HLSL FAIL %s: %s\n", name.c_str(), info.error.c_str());
        }
        std::vector<uint32_t> words;
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = TranslateSpirv(shader, info, words, why, options);
        const auto t1 = std::chrono::steady_clock::now();
        if (!ok)
        {
            fallback++;
            reasons[why]++;
            continue;
        }
        translated++;
        directMs += Ms(t1 - t0);
        const std::filesystem::path spv = (keep.empty() ? tmp : keep) / (std::filesystem::path(name).stem().string() + ".spv");
        {
            std::ofstream o(spv, std::ios::binary);
            o.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        }
        const std::string cmd = "\"\"" + val + "\" --target-env vulkan1.3 \"" + spv.string() + "\" 2>&1\"";
        std::string output;
        if (FILE* pipe = _popen(cmd.c_str(), "r"))
        {
            char buf[512];
            while (fgets(buf, sizeof(buf), pipe))
                output += buf;
            if (_pclose(pipe) != 0)
            {
                invalid++;
                printf("INVALID %s:\n%s\n", name.c_str(), output.c_str());
            }
        }
        if (dxc && info.ok)
        {
            const auto h0 = std::chrono::steady_clock::now();
            const TranslatedShader again = TranslateShader(shader);
            std::vector<uint32_t> spirv;
            std::string error;
            const bool compiled = vk::CompileHlsl(vk::AdaptHlsl(again.hlsl), pixel, 0, {}, spirv, &error);
            const auto h1 = std::chrono::steady_clock::now();
            if (compiled)
            {
                hlslMs += Ms(h1 - h0);
                hlslCount++;
            }
            else
                printf("DXC FAIL %s: %s\n", name.c_str(), error.substr(0, 300).c_str());
        }
    }
    const int total = translated + fallback;
    printf("\n%d shaders: %d translated directly (%.1f%%), %d fall back to HLSL; %d spirv-val failures; %d HLSL translation failures\n", total,
        translated, total ? 100.0 * translated / total : 0.0, fallback, invalid, hlslFailed);
    for (const auto& [reason, n] : reasons)
        printf("  fallback %4d  %s\n", n, reason.c_str());
    if (translated)
        printf("direct: %.3f ms average over %d shaders\n", directMs / translated, translated);
    if (hlslCount)
        printf("HLSL + DXC (same shaders, no cache): %.1f ms average over %d\n", hlslMs / hlslCount, hlslCount);
    return invalid ? 1 : 0;
}
