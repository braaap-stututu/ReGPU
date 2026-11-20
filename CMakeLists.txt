cmake_minimum_required(VERSION 3.25)
project(ReGPU CXX)

# ReGPU: the Xbox 360 GPU for XenonRecomp ports (include/regpu/regpu.h). A static library the
# game links; the game's own CMake sets the paths below (or add_subdirectory with them set).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "ReGPU needs clang-cl, like XenonRecomp's output")
endif()

# xenonrecomp: ReGPU provides the Vd* exports (a XenonRecomp port). rexglue: the host's kernel
# provides them and calls regpu::driver (include/regpu/driver.h); no game headers needed.
set(REGPU_GUEST_ABI "xenonrecomp" CACHE STRING "Guest ABI: xenonrecomp or rexglue")
set_property(CACHE REGPU_GUEST_ABI PROPERTY STRINGS xenonrecomp rexglue)
set(REGPU_XENONRECOMP "" CACHE PATH "XenonRecomp checkout (for XenonUtils/ppc_context.h)")
set(REGPU_PPC_DIR "" CACHE PATH "The game's XenonRecomp output folder (its ppc_config.h)")
set(REGPU_REXGLUE_DIR "" CACHE PATH "ReXGlue SDK source or install (its include/rex/ppc/context.h), with REGPU_GUEST_ABI=rexglue")
set(REGPU_FFX "" CACHE PATH "AMD FidelityFX SDK's Kits/FidelityFX folder (FSR1 shader headers; FSR3 runtime with REGPU_FSR3)")
option(REGPU_FSR3 "AMD FSR 3 upscaling (needs the FidelityFX SDK's signed DLLs)" OFF)
option(REGPU_DLSS "NVIDIA DLSS (needs the DLSS SDK)" OFF)
set(REGPU_DLSS_DIR "" CACHE PATH "NVIDIA DLSS SDK folder (with REGPU_DLSS)")
set(REGPU_D3DCOMPILER "C:/Program Files (x86)/Windows Kits/10/Redist/D3D/x64/d3dcompiler_47.dll" CACHE FILEPATH
    "Redistributable d3dcompiler_47.dll, to ship beside the game executable (src/gpu/d3dcompiler.cpp)")

if(REGPU_GUEST_ABI STREQUAL "xenonrecomp")
    if(NOT EXISTS "${REGPU_XENONRECOMP}/XenonUtils/ppc_context.h")
        message(FATAL_ERROR "Set REGPU_XENONRECOMP to a XenonRecomp checkout")
    endif()
    if(NOT EXISTS "${REGPU_PPC_DIR}/ppc_config.h")
        message(FATAL_ERROR "Set REGPU_PPC_DIR to the game's XenonRecomp output (with ppc_config.h)")
    endif()
    set(REGPU_GUEST_INCLUDES "${REGPU_PPC_DIR}" "${REGPU_XENONRECOMP}/XenonUtils" "${REGPU_XENONRECOMP}/thirdparty/simde")
    set(REGPU_GUEST_DEFINES REGPU_GUEST_EXPORTS=1 REGPU_REXGLUE=0)
elseif(REGPU_GUEST_ABI STREQUAL "rexglue")
    if(NOT EXISTS "${REGPU_REXGLUE_DIR}/include/rex/ppc/context.h")
        message(FATAL_ERROR "Set REGPU_REXGLUE_DIR to a ReXGlue SDK (with include/rex/ppc/context.h)")
    endif()
    # ReXGlue's headers are C++23 and use SIMDe (its thirdparty/simde submodule).
    set(CMAKE_CXX_STANDARD 23)
    set(REGPU_GUEST_INCLUDES "${REGPU_REXGLUE_DIR}/include" "${REGPU_REXGLUE_DIR}/thirdparty/simde")
    set(REGPU_GUEST_DEFINES REGPU_GUEST_EXPORTS=0 REGPU_REXGLUE=1)
else()
    message(FATAL_ERROR "REGPU_GUEST_ABI must be xenonrecomp or rexglue")
endif()
set(REGPU_FSR1_CS "" CACHE FILEPATH "A pre-built fsr1_cs.h (tools/embed_hlsl.py's output): no fxc, py or FidelityFX headers at build time")
if(NOT REGPU_FSR1_CS AND NOT EXISTS "${REGPU_FFX}/upscalers/fsr3/include/gpu/fsr1/ffx_fsr1.h")
    message(FATAL_ERROR "Set REGPU_FFX to the FidelityFX SDK's Kits/FidelityFX folder (FSR1 shaders)")
endif()

file(GLOB_RECURSE REGPU_SOURCES CONFIGURE_DEPENDS src/*.cpp)
# Vulkan (branch `vulkan`: becoming ReGPU's one render backend, docs/vulkan-conversion-plan.md). Always built. Vulkan
# itself is loaded at run time through volk (thirdparty/volk), so only headers are needed: the vendored
# thirdparty/Vulkan-Headers unless REGPU_VULKAN_HEADERS names others. Memory: VMA (thirdparty/vma). The shaders go to
# SPIR-V through dxcompiler.dll (DirectXShaderCompiler, with dxil.dll), which the port ships beside its executable.
set(REGPU_VULKAN ON)
set(REGPU_VULKAN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Vulkan-Headers/include" CACHE PATH "Vulkan headers (the folder containing vulkan/vulkan.h)")
if(NOT REGPU_FSR3)
    list(REMOVE_ITEM REGPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/fsr3.cpp")
endif()
if(NOT REGPU_DLSS)
    list(REMOVE_ITEM REGPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/dlss.cpp")
endif()
# Intel XeSS (src/gpu/xess.cpp, FM3 port session 43): the SDK's headers (libxess.dll is loaded at run time, beside the exe).
set(REGPU_XESS_DIR "" CACHE PATH "Intel XeSS SDK folder (with inc/xess/xess_vk.h); empty: built without")
if(NOT (REGPU_XESS_DIR AND EXISTS "${REGPU_XESS_DIR}/inc/xess/xess_vk.h"))
    list(REMOVE_ITEM REGPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/xess.cpp")
endif()

# The FSR1, depth copy, gamma and enhance compute shaders, compiled into a header at build time (SPIR-V, by DXC: the Vulkan
# SDK's, REGPU_DXC_EXE, or dxc on PATH).
file(GLOB REGPU_SDK_DXC "$ENV{VULKAN_SDK}/Bin/dxc.exe" "C:/VulkanSDK/*/Bin/dxc.exe" "$ENV{VULKAN_SDK}/bin/dxc")
set(REGPU_SDK_DXC_FIRST "")
if(REGPU_SDK_DXC)
    list(SORT REGPU_SDK_DXC ORDER DESCENDING)
    list(GET REGPU_SDK_DXC 0 REGPU_SDK_DXC_FIRST)
endif()
if(REGPU_SDK_DXC_FIRST)
    set(REGPU_DXC_EXE "${REGPU_SDK_DXC_FIRST}" CACHE FILEPATH "DXC with SPIR-V output (for ReGPU's embedded compute shaders)")
else()
    find_program(REGPU_DXC_EXE NAMES dxc DOC "DXC with SPIR-V output (for ReGPU's embedded compute shaders)")
endif()
set(REGPU_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${REGPU_GENERATED}")
if(REGPU_FSR1_CS)
    add_custom_command(OUTPUT "${REGPU_GENERATED}/fsr1_cs.h"
        COMMAND ${CMAKE_COMMAND} -E copy "${REGPU_FSR1_CS}" "${REGPU_GENERATED}/fsr1_cs.h"
        DEPENDS "${REGPU_FSR1_CS}")
else()
add_custom_command(OUTPUT "${REGPU_GENERATED}/fsr1_cs.h"
    COMMAND py -3 -I "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py" --dxc "${REGPU_DXC_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/fsr1.hlsl" "${REGPU_GENERATED}/fsr1_cs.h"
            EasuMain,RcasMain,DepthCopyMain,GammaMain,EnhanceMain,ViewMain "${REGPU_FFX}/api/internal/gpu" "${REGPU_FFX}/upscalers/fsr3/include/gpu/fsr1"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/fsr1.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py")
endif()
list(APPEND REGPU_SOURCES "${REGPU_GENERATED}/fsr1_cs.h")

# Anti-aliasing filters (FM3 port, session 36): ReGPU's own passes (postaa.hlsl: tone-map wrap, TAA, SSAA) always; FXAA
# 3.11, SMAA and CMAA 2 from their original sources in REGPU_AA_DIR (fxaa/Fxaa3_11.h, smaa/SMAA.hlsl + AreaTex.h +
# SearchTex.h, cmaa2/CMAA2.hlsl), when the port provides it (REGPU_POSTAA_FILTERS=1).
set(REGPU_AA_DIR "" CACHE PATH "Folder with fxaa/, smaa/ and cmaa2/ (the third-party anti-aliasing shaders)")
add_custom_command(OUTPUT "${REGPU_GENERATED}/postaa_cs.h"
    COMMAND py -3 -I "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py" --dxc "${REGPU_DXC_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/postaa.hlsl" "${REGPU_GENERATED}/postaa_cs.h"
            PrepMain,FinishMain,CopyMain,TaaMain,SsaaMain
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/postaa.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py")
list(APPEND REGPU_SOURCES "${REGPU_GENERATED}/postaa_cs.h")
set(REGPU_POSTAA_FILTERS 0)
if(REGPU_AA_DIR AND EXISTS "${REGPU_AA_DIR}/fxaa/Fxaa3_11.h" AND EXISTS "${REGPU_AA_DIR}/smaa/SMAA.hlsl" AND EXISTS "${REGPU_AA_DIR}/cmaa2/CMAA2.hlsl")
    set(REGPU_POSTAA_FILTERS 1)
    foreach(aa fxaa smaa cmaa2)
        if(aa STREQUAL "fxaa")
            set(entries FxaaMain)
        elseif(aa STREQUAL "smaa")
            set(entries SmaaEdgesMain,SmaaWeightsMain,SmaaBlendMain)
        else()
            set(entries EdgesColor2x2CS,ProcessCandidatesCS,DeferredColorApply2x2CS,ComputeDispatchArgsCS)
        endif()
        file(GLOB aa_deps "${REGPU_AA_DIR}/${aa}/*")
        add_custom_command(OUTPUT "${REGPU_GENERATED}/postaa_${aa}_cs.h"
            COMMAND py -3 -I "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py" --dxc "${REGPU_DXC_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/postaa_${aa}.hlsl"
                    "${REGPU_GENERATED}/postaa_${aa}_cs.h" ${entries} "${REGPU_AA_DIR}/${aa}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/postaa_${aa}.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_hlsl.py" ${aa_deps})
        list(APPEND REGPU_SOURCES "${REGPU_GENERATED}/postaa_${aa}_cs.h")
    endforeach()
else()
    message(STATUS "ReGPU: REGPU_AA_DIR not set: FXAA, SMAA and CMAA 2 left out (TAA and SSAA stay)")
endif()

add_library(regpu STATIC ${REGPU_SOURCES})
# The experimental direct microcode-to-SPIR-V translator (src/gpu/spirv) builds on the vendored SPIR-V headers.
target_include_directories(regpu PRIVATE thirdparty/spirv-headers/include)
if(REGPU_VULKAN)
    target_include_directories(regpu PRIVATE "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu PUBLIC REGPU_VULKAN=1)
    target_compile_definitions(regpu PRIVATE VK_NO_PROTOTYPES $<$<BOOL:${WIN32}>:VK_USE_PLATFORM_WIN32_KHR>)
endif()
target_include_directories(regpu PUBLIC include ${REGPU_GUEST_INCLUDES})
target_compile_definitions(regpu PUBLIC ${REGPU_GUEST_DEFINES})
target_include_directories(regpu PRIVATE src "${REGPU_GENERATED}")
# VR through OpenXR (src/render/xr.cpp, REGPU_VR=screen): the OpenXR SDK's include/openxr headers. The loader DLL is
# loaded at run time (openxr_loader.dll beside the exe), so nothing links against it.
if(REGPU_XESS_DIR AND EXISTS "${REGPU_XESS_DIR}/inc/xess/xess_vk.h")
    target_compile_definitions(regpu PRIVATE REGPU_XESS=1)
    target_include_directories(regpu PRIVATE "${REGPU_XESS_DIR}/inc")
endif()
# Hyper's mesh decimation and vertex-cache order (src/gpu/renderer.cpp hyperdec, FM3 port session 43): meshoptimizer
# (MIT, github.com/zeux/meshoptimizer) compiled in when its folder is given; empty: built without (the cuts log so).
set(REGPU_MESHOPT_DIR "" CACHE PATH "meshoptimizer source folder (with src/meshoptimizer.h)")
if(REGPU_MESHOPT_DIR AND EXISTS "${REGPU_MESHOPT_DIR}/src/meshoptimizer.h")
    file(GLOB REGPU_MESHOPT_SOURCES "${REGPU_MESHOPT_DIR}/src/*.cpp")
    target_sources(regpu PRIVATE ${REGPU_MESHOPT_SOURCES})
    target_compile_definitions(regpu PRIVATE REGPU_MESHOPT=1)
    target_include_directories(regpu PRIVATE "${REGPU_MESHOPT_DIR}/src")
endif()
set(REGPU_OPENXR_DIR "" CACHE PATH "OpenXR SDK folder with include/openxr/openxr.h (VR; empty: built without)")
if(REGPU_OPENXR_DIR AND EXISTS "${REGPU_OPENXR_DIR}/include/openxr/openxr.h")
    target_compile_definitions(regpu PRIVATE REGPU_OPENXR=1)
    target_include_directories(regpu PRIVATE "${REGPU_OPENXR_DIR}/include")
endif()
target_compile_definitions(regpu PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
target_compile_definitions(regpu PRIVATE REGPU_POSTAA_FILTERS=${REGPU_POSTAA_FILTERS})
if(REGPU_POSTAA_FILTERS)
    target_include_directories(regpu PRIVATE "${REGPU_AA_DIR}/smaa") # AreaTex.h, SearchTex.h
endif()
target_compile_options(regpu PRIVATE -msse4.1 -fno-strict-aliasing -gline-tables-only)
target_link_libraries(regpu PUBLIC ws2_32 winmm synchronization onecore)
if(REGPU_FSR3)
    target_compile_definitions(regpu PRIVATE REGPU_FSR3=1)
    target_include_directories(regpu PRIVATE "${REGPU_FFX}/api/include" "${REGPU_FFX}/upscalers/include")
    target_link_libraries(regpu PUBLIC "${REGPU_FFX}/signedbin/amd_fidelityfx_loader_vk.lib")
endif()
if(REGPU_DLSS)
    target_compile_definitions(regpu PRIVATE REGPU_DLSS=1)
    target_include_directories(regpu PRIVATE "${REGPU_DLSS_DIR}/include")
    target_link_libraries(regpu PUBLIC "${REGPU_DLSS_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib")
endif()

# Build check only: ReGPU linked with nothing else (no unresolved symbols from a game).
add_executable(regpu_linkcheck EXCLUDE_FROM_ALL tests/linkcheck.cpp)
target_link_libraries(regpu_linkcheck PRIVATE regpu)

add_executable(regpu_texture_test EXCLUDE_FROM_ALL tests/texture.cpp)
target_include_directories(regpu_texture_test PRIVATE src)
target_link_libraries(regpu_texture_test PRIVATE regpu)

# Xenos shader disassembler / translator check (xenos_dis --hlsl <ucode files>).
add_executable(regpu_dis EXCLUDE_FROM_ALL tools/xenos_dis.cpp)
target_include_directories(regpu_dis PRIVATE src)
target_link_libraries(regpu_dis PRIVATE regpu)
# The experimental direct SPIR-V translator against the corpus: coverage, spirv-val, timing against HLSL + DXC.
add_executable(regpu_spirv_check EXCLUDE_FROM_ALL tools/spirv_check.cpp)
target_include_directories(regpu_spirv_check PRIVATE src)
target_link_libraries(regpu_spirv_check PRIVATE regpu)

# The Vulkan backend's smoke test: device and shader compile (tests/vkprobe.cpp). Copy dxcompiler.dll beside it.
if(REGPU_VULKAN)
    add_executable(regpu_vkprobe EXCLUDE_FROM_ALL tests/vkprobe.cpp)
    target_link_libraries(regpu_vkprobe PRIVATE regpu)
    target_include_directories(regpu_vkprobe PRIVATE src "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu_vkprobe PRIVATE VK_NO_PROTOTYPES)
    # vk::RecordedList on the GPU (tests/vkrecord.cpp): lazy render passes, clears, copies, queries, hand-off.
    add_executable(regpu_vkrecord EXCLUDE_FROM_ALL tests/vkrecord.cpp)
    target_link_libraries(regpu_vkrecord PRIVATE regpu)
    target_include_directories(regpu_vkrecord PRIVATE src "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu_vkrecord PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
    # gpu/vk/resources.h on the GPU (tests/vkresources.cpp): formats, transitions, round trips, the graveyard.
    add_executable(regpu_vkresources EXCLUDE_FROM_ALL tests/vkresources.cpp)
    target_link_libraries(regpu_vkresources PRIVATE regpu)
    target_include_directories(regpu_vkresources PRIVATE src "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu_vkresources PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
    # A draw through the converted pieces (tests/vkdraw.cpp): bindings, SPIR-V shaders, pipeline, recorder.
    add_executable(regpu_vkdraw EXCLUDE_FROM_ALL tests/vkdraw.cpp)
    target_link_libraries(regpu_vkdraw PRIVATE regpu)
    target_include_directories(regpu_vkdraw PRIVATE src "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu_vkdraw PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
    # gpu/vk/passes.h on the GPU (tests/vkpasses.cpp): a compute and a graphics pass from HLSL, layouts from SPIR-V.
    add_executable(regpu_vkpasses EXCLUDE_FROM_ALL tests/vkpasses.cpp)
    target_link_libraries(regpu_vkpasses PRIVATE regpu)
    target_include_directories(regpu_vkpasses PRIVATE src "${REGPU_VULKAN_HEADERS}" thirdparty/volk thirdparty/vma)
    target_compile_definitions(regpu_vkpasses PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
