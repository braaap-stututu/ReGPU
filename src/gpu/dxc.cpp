// REGPU_DXC=1 (FM3 port, session 34; read once, at the first compile): the translated shaders compiled by DXC
// (dxcompiler.dll + dxil.dll beside the game executable, Microsoft's redistributables from the Windows SDK's
// Redist\D3D folder) to DXIL shader model 6.0 instead of by FXC to DXBC 5.1. Offline (RGA, RDNA3) DXC's code had
// 7-13% fewer ALU instructions than FXC's unoptimised build, which ReGPU compiles at run time. Shaders and pipelines
// cache apart (renderer.cpp: .dxil files, pipelines-dxil*.bin). HLSL 2018 rules (-HV 2018): DXC's default 2021
// rejects the translator's vector ?: and ||. Without the DLLs: FXC, with a log line.
#include <windows.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "log.h"

using Microsoft::WRL::ComPtr;

namespace
{
DxcCreateInstanceProc CreateInstance()
{
    static const DxcCreateInstanceProc create = [] () -> DxcCreateInstanceProc {
        wchar_t path[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
        HMODULE m = nullptr;
        if (n && n < MAX_PATH)
        {
            std::wstring dir(path, n);
            dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
            // dxil.dll first, by full path: dxcompiler signs its output with it (unsigned DXIL is rejected).
            LoadLibraryExW((dir + L"dxil.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            m = LoadLibraryExW((dir + L"dxcompiler.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
        if (!m)
        {
            Log("dxc: dxcompiler.dll not found beside the executable: shaders compile with FXC");
            return nullptr;
        }
        auto f = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(m, "DxcCreateInstance"));
        Log("dxc: %s", f ? "translated shaders compile with DXC (shader model 6.0)" : "DxcCreateInstance missing: shaders compile with FXC");
        return f;
    }();
    return create;
}
} // namespace

// REGPU_DXC: every translated shader or none: D3D12 refuses a pipeline mixing a DXBC vertex shader with a DXIL
// pixel shader (tried: every such pipeline failed).
int DxcStages() // bit 0 vertex, bit 1 pixel
{
    static const int stages = [] {
        const char* v = getenv("REGPU_DXC");
        return v && *v && strcmp(v, "0") != 0 && CreateInstance() ? 3 : 0;
    }();
    return stages;
}

bool DxcEnabled()
{
    return DxcStages() != 0;
}

bool DxcFor(bool pixel)
{
    return (DxcStages() >> (pixel ? 1 : 0)) & 1;
}

// The cache tag for REGPU_DXC_ARGS: "" without extra arguments, else "-<hash>", so builds with other arguments
// (and their pipelines) cache apart.
std::string DxcCacheTag()
{
    static const std::string tag = [] {
        const char* e = getenv("REGPU_DXC_ARGS");
        const std::string stage;
        if (!e || !*e)
            return stage;
        uint32_t h = 2166136261u;
        for (const char* c = e; *c; c++)
            h = (h ^ uint8_t(*c)) * 16777619u;
        char b[16];
        snprintf(b, sizeof(b), "-%08x", h);
        return stage + b;
    }();
    return tag;
}

// One compiler per thread (DXC's compiler objects are not shared between threads). The DXIL object is copied into a
// D3D blob, so callers keep using ID3DBlob as with FXC.
HRESULT DxcCompileShader(const std::string& hlsl, bool pixel, ID3DBlob** code, std::string* errors)
{
    *code = nullptr;
    const DxcCreateInstanceProc create = CreateInstance();
    if (!create)
        return E_NOTIMPL;
    thread_local ComPtr<IDxcCompiler3> compiler;
    if (!compiler && FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))))
        return E_FAIL;
    // REGPU_DXC_ARGS: extra arguments, space-separated (e.g. "-O0", "-Od"; default DXC's -O3).
    static const std::vector<std::wstring> extra = [] {
        std::vector<std::wstring> v;
        if (const char* e = getenv("REGPU_DXC_ARGS"))
        {
            std::string s = e;
            for (size_t at = 0; at < s.size();)
            {
                const size_t end = s.find(' ', at);
                const std::string a = s.substr(at, end == std::string::npos ? std::string::npos : end - at);
                if (!a.empty())
                    v.emplace_back(a.begin(), a.end());
                at = end == std::string::npos ? s.size() : end + 1;
            }
        }
        return v;
    }();
    std::vector<LPCWSTR> args = { L"-E", L"main", L"-T", pixel ? L"ps_6_0" : L"vs_6_0", L"-HV", L"2018" };
    if (!pixel)
        args.insert(args.end(), { L"-D", L"REGPU_VERTEX_STAGE=1" });
    for (const std::wstring& a : extra)
        args.push_back(a.c_str());
    const DxcBuffer source{ hlsl.data(), hlsl.size(), DXC_CP_UTF8 };
    ComPtr<IDxcResult> result;
    if (FAILED(compiler->Compile(&source, args.data(), UINT32(args.size()), nullptr, IID_PPV_ARGS(&result))))
        return E_FAIL;
    HRESULT status = E_FAIL;
    result->GetStatus(&status);
    if (errors)
    {
        ComPtr<IDxcBlobUtf8> text;
        if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&text), nullptr)) && text && text->GetStringLength())
            errors->assign(text->GetStringPointer(), text->GetStringLength());
    }
    if (FAILED(status))
        return status;
    ComPtr<IDxcBlob> object;
    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) || !object || !object->GetBufferSize())
        return E_FAIL;
    if (FAILED(D3DCreateBlob(object->GetBufferSize(), code)))
        return E_OUTOFMEMORY;
    memcpy((*code)->GetBufferPointer(), object->GetBufferPointer(), object->GetBufferSize());
    return S_OK;
}
