#include "gpu/vk/vk_cache_hash.h"
#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_shader.h"


#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#if defined(_WIN32)
#include <windows.h>
#include <unknwn.h>
#include <dxcapi.h>
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace vk
{
namespace
{
std::string TempPath(const char* ext)
{
    char dir[512] = {};
#if defined(_WIN32)
    const char* tmp = getenv("TEMP");
    snprintf(dir, sizeof dir, "%s\\regpuvkXXXXXX", tmp ? tmp : ".");
#else
    snprintf(dir, sizeof dir, "/tmp/regpuvkXXXXXX");
    if (int fd = mkstemp(dir); fd >= 0)
        close(fd);
#endif
    return std::string(dir) + ext;
}

bool ReadAll(const std::string& path, std::vector<uint32_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<char> bytes{ std::istreambuf_iterator<char>(f), {} };
    if (bytes.empty() || (bytes.size() & 3))
        return false;
    out.resize(bytes.size() / 4);
    memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

#if defined(_WIN32)
std::wstring Wide(const char* s)
{
    std::wstring w;
    for (; *s; s++)
        w += wchar_t(*s);
    return w;
}

bool CompileDxcInProcess(const std::string& hlsl, const char* profile, const char* entry, bool vertexDefine, bool disableOptimizations,
    std::vector<uint32_t>& spirv, std::string* error, bool& unavailable)
{
    using CreateInstance = HRESULT(WINAPI*)(REFCLSID, REFIID, LPVOID*);
    static const auto create = [] {
        HMODULE library = GetModuleHandleW(L"dxcompiler.dll");
        if (!library)
            library = LoadLibraryW(L"dxcompiler.dll");
        return library ? reinterpret_cast<CreateInstance>(GetProcAddress(library, "DxcCreateInstance")) : nullptr;
    }();
    if (!create || getenv("REGPU_VK_DXC_PROCESS"))
    {
        unavailable = true;
        return false;
    }
    IDxcCompiler3* compiler = nullptr;
    if (FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))))
    {
        unavailable = true;
        return false;
    }
    const std::wstring wideProfile = Wide(profile), wideEntry = Wide(entry);
    std::vector<LPCWSTR> arguments = {
        L"-HV", L"2018", L"-T", wideProfile.c_str(), L"-E", wideEntry.c_str(), L"-spirv", L"-fvk-use-dx-layout",
        L"-fspv-target-env=vulkan1.2",
        L"-fvk-b-shift", L"0", L"0", L"-fvk-t-shift", L"100", L"0", L"-fvk-s-shift", L"200", L"0", L"-fvk-u-shift", L"300", L"0",
    };
    if (vertexDefine)
        arguments.insert(arguments.end(), { L"-D", L"REGPU_VERTEX_STAGE" });
    if (disableOptimizations || getenv("REGPU_VK_SHADER_OD"))
        arguments.push_back(L"-Od");
    DxcBuffer source{ hlsl.data(), hlsl.size(), CP_UTF8 };
    IDxcResult* result = nullptr;
    const HRESULT compiled = compiler->Compile(&source, arguments.data(), UINT32(arguments.size()), nullptr, IID_PPV_ARGS(&result));
    HRESULT status = E_FAIL;
    if (SUCCEEDED(compiled) && result)
        result->GetStatus(&status);
    IDxcBlobEncoding* errors = nullptr;
    if (result && FAILED(status) && SUCCEEDED(result->GetErrorBuffer(&errors)) && errors && errors->GetBufferSize() && error)
        error->assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
    IDxcBlob* object = nullptr;
    if (SUCCEEDED(status) && SUCCEEDED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) && object)
    {
        const auto* bytes = static_cast<const uint8_t*>(object->GetBufferPointer());
        const size_t size = object->GetBufferSize();
        if (!(size & 3))
        {
            spirv.resize(size / 4);
            memcpy(spirv.data(), bytes, size);
        }
    }
    if (object)
        object->Release();
    if (errors)
        errors->Release();
    if (result)
        result->Release();
    compiler->Release();
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        fprintf(stderr, "vk: shader compiler dxcompiler.dll (in process)\n");
    }
    return !spirv.empty();
}
#endif

// Written under a temporary name and renamed: the warm-up threads (vk_shaders.cpp) and the renderer may reach the same
// shader at once, and a reader must never see half a file.
void WriteCached(const std::filesystem::path& cacheDir, uint64_t key, uint64_t contentHash, const std::vector<uint32_t>& spirv)
{
    std::error_code ec;
    std::filesystem::create_directories(cacheDir / "spv", ec);
    const std::filesystem::path final = cacheDir / "spv" / (std::to_string(key) + "_" + std::to_string(contentHash) + ".spv");
    static std::atomic<uint32_t> serial{ 0 };
    const std::filesystem::path temporary = cacheDir / "spv" /
        (std::to_string(key) + "_" + std::to_string(contentHash) + "." + std::to_string(serial++) + ".tmp");
    {
        std::ofstream out(temporary, std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size() * 4));
        if (!out)
        {
            out.close();
            std::filesystem::remove(temporary, ec);
            return;
        }
    }
    std::filesystem::rename(temporary, final, ec);
    if (ec)
        std::filesystem::remove(temporary, ec);
}
}

bool SpirvCached(const std::string& hlsl, uint64_t key, const std::filesystem::path& cacheDir)
{
    if (cacheDir.empty())
        return false;
    std::error_code ec;
    return std::filesystem::exists(cacheDir / "spv" / (std::to_string(key) + "_" + std::to_string(HlslContentHash(hlsl)) + ".spv"), ec);
}

namespace
{
bool Compile(const std::string& hlsl, const char* profile, const char* entry, bool vertexDefine, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error, bool disableOptimizations);
}

bool CompileHlsl(const std::string& hlsl, bool pixel, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error, bool disableOptimizations)
{
    return Compile(hlsl, pixel ? "ps_6_0" : "vs_6_0", "main", !pixel, key, cacheDir, spirv, error, disableOptimizations);
}

bool CompileHlslStage(const std::string& hlsl, const char* profile, const char* entry, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error)
{
    const uint64_t key = HashSource(std::string(profile) + ":" + entry);
    return Compile(hlsl, profile, entry, false, key, cacheDir, spirv, error, false);
}

namespace
{
bool Compile(const std::string& hlsl, const char* profile, const char* entry, bool vertexDefine, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error, bool disableOptimizations)
{
    // The key is the GUEST shader hash, which does not change when the translator does, so the cache file name also
    // carries a content hash of the generated HLSL: an edit to the translator must not be served stale SPIR-V.
    const uint64_t contentHash = HlslContentHash(hlsl);

    if (!cacheDir.empty())
    {
        const auto cached = cacheDir / "spv" / (std::to_string(key) + "_" + std::to_string(contentHash) + ".spv");
        if (ReadAll(cached.string(), spirv))
            return true;
    }

#if defined(_WIN32)
    bool unavailable = false;
    if (CompileDxcInProcess(hlsl, profile, entry, vertexDefine, disableOptimizations, spirv, error, unavailable))
    {
        if (!cacheDir.empty())
            WriteCached(cacheDir, key, contentHash, spirv);
        return true;
    }
    if (!unavailable)
        return false;
#endif

    const std::string hlslPath = TempPath(".hlsl");
    const std::string spvPath = TempPath(".spv");
    {
        std::ofstream f(hlslPath, std::ios::binary);
        if (!f)
            return false;
        f.write(hlsl.data(), std::streamsize(hlsl.size()));
    }
    const char* dxc = getenv("REGPU_DXC");
    // REGPU_DXC is often set to a Unix path (the build passes /home/.../dxc). The Windows-side shell cannot execute
    // that, the compile fails and nothing draws, so on Windows a Unix value is ignored with a warning and the bundled
    // dxc or the one on PATH is used. The native Linux test binaries execute a Unix path directly.
#if defined(_WIN32)
    if (dxc && dxc[0] == '/')
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            fprintf(stderr, "vk: REGPU_DXC=%s is a Unix path and cannot be executed from here; using bundled dxc or PATH\n", dxc);
        }
        dxc = nullptr;
    }
#endif
    std::string dxcCommand = dxc ? dxc : "dxc";
#if defined(_WIN32)
    if (!dxc)
    {
        char executable[MAX_PATH];
        const DWORD length = GetModuleFileNameA(nullptr, executable, MAX_PATH);
        if (length && length < MAX_PATH)
        {
            const auto bundled = std::filesystem::path(std::string(executable, length)).parent_path() / "dxc.exe";
            if (std::filesystem::exists(bundled))
                dxcCommand = bundled.string();
        }
    }
#endif
    if (dxcCommand.find_first_of(" \t") != std::string::npos)
        dxcCommand = "\"" + dxcCommand + "\"";
    static bool loggedCompiler = false;
    if (!loggedCompiler)
    {
        loggedCompiler = true;
        fprintf(stderr, "vk: shader compiler %s\n", dxcCommand.c_str());
    }
    // Sized for the worst case (the flags plus two TEMP paths), and truncation is detected rather than silently
    // handing the shell a command cut mid-flag.
    char command[4096];
    // REGPU_VK_SHADER_OD turns the optimiser off. A translated pixel shader is a state machine
    // in a [loop] with a guard, and dxc's optimiser is the last thing between it and the same
    // result the Direct3D 12 compiler gives; this is how to tell a miscompile from a translation.
    const char* optimise = getenv("REGPU_VK_SHADER_OD") || disableOptimizations ? " -Od" : "";
    const int written = snprintf(command, sizeof command,
        "%s -HV 2018 -T %s -E %s -spirv -fvk-use-dx-layout -fspv-target-env=vulkan1.2%s%s"
        " -fvk-b-shift 0 0 -fvk-t-shift 100 0 -fvk-s-shift 200 0 -fvk-u-shift 300 0"
        " -Fo \"%s\" \"%s\" 2>&1",
        dxcCommand.c_str(), profile, entry, vertexDefine ? " -D REGPU_VERTEX_STAGE" : "", optimise, spvPath.c_str(),
        hlslPath.c_str());
    if (written < 0 || size_t(written) >= sizeof command)
    {
        if (error)
            *error = "shader compile command truncated";
        return false;
    }
    bool ok = false;
    if (FILE* pipe = popen(command, "r"))
    {
        char buffer[512];
        std::string log;
        while (fgets(buffer, sizeof buffer, pipe))
            log += buffer;
        ok = pclose(pipe) == 0;
        if (!ok && error)
            *error = log;
    }
    if (ok)
        ok = ReadAll(spvPath, spirv);
    std::error_code ignored;
    std::filesystem::remove(spvPath, ignored);
    std::filesystem::remove(hlslPath, ignored);

    if (ok && !cacheDir.empty())
    {
        std::filesystem::create_directories(cacheDir / "spv");
        std::ofstream out(cacheDir / "spv" / (std::to_string(key) + "_" + std::to_string(contentHash) + ".spv"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size() * 4));
    }
    return ok;
}
}
}
