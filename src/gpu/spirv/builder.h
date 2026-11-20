#pragma once
// A small SPIR-V module builder for the direct Xenos-to-SPIR-V translator (translate.h): ids, types and constants
// (deduplicated), decorations, global variables, one function body at a time with its blocks, structured
// selections, and GLSL.std.450 instructions. No validation of its own: spirv-val (regpu_spirv_check) checks the output.
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "spirv/unified1/GLSL.std.450.h"
#include "spirv/unified1/spirv.hpp"

namespace spirv
{
class Builder
{
public:
    Builder();

    uint32_t Id() { return next_++; }

    // Module header parts.
    void Capability(spv::Capability c);
    void Extension(const char* name);
    void EntryPoint(spv::ExecutionModel model, uint32_t function, const char* name, const std::vector<uint32_t>& interface);
    void ExecutionMode(uint32_t function, spv::ExecutionMode mode, std::initializer_list<uint32_t> args = {});
    void Name(uint32_t id, const char* name);
    void Decorate(uint32_t id, spv::Decoration d, std::initializer_list<uint32_t> args = {});
    void MemberDecorate(uint32_t type, uint32_t member, spv::Decoration d, std::initializer_list<uint32_t> args = {});

    // Types (deduplicated; arrays and structs too: callers decorate a struct right after creating it).
    uint32_t TypeVoid();
    uint32_t TypeBool();
    uint32_t TypeFloat();
    uint32_t TypeUint();
    uint32_t TypeInt();
    uint32_t TypeVector(uint32_t component, uint32_t count);
    uint32_t TypeArray(uint32_t element, uint32_t length); // length: a literal (made a uint constant)
    uint32_t TypeStruct(const std::vector<uint32_t>& members); // never deduplicated (decorations differ)
    uint32_t TypePointer(spv::StorageClass sc, uint32_t type);
    uint32_t TypeFunction(uint32_t ret, const std::vector<uint32_t>& params = {});
    // Any other type, deduplicated by its operands (OpTypeImage, OpTypeSampler, OpTypeSampledImage, OpTypeRuntimeArray).
    uint32_t Type(spv::Op op, const std::vector<uint32_t>& operands);

    // Constants (deduplicated).
    uint32_t ConstFloat(float v);
    uint32_t ConstUint(uint32_t v);
    uint32_t ConstInt(int32_t v);
    uint32_t ConstBool(bool v);
    uint32_t ConstComposite(uint32_t type, const std::vector<uint32_t>& parts);
    uint32_t ConstVec4(float x, float y, float z, float w);

    // Global variable (Input, Output, Uniform...).
    uint32_t GlobalVariable(uint32_t pointerType, spv::StorageClass sc);

    // Functions: one at a time. Begin opens the entry block; Variable adds a Function variable to it.
    void BeginFunction(uint32_t function, uint32_t ret, uint32_t fnType);
    // With parameters (their types; their ids are returned), before the entry block.
    std::vector<uint32_t> BeginFunction(uint32_t function, uint32_t ret, uint32_t fnType, const std::vector<uint32_t>& paramTypes);
    void ReturnValue(uint32_t value);
    uint32_t Variable(uint32_t pointerType);
    void EndFunction();

    // Instructions into the current block. Op returns the result id (resultType 0: no result type or id).
    uint32_t Op(spv::Op op, uint32_t resultType, std::initializer_list<uint32_t> args);
    uint32_t OpV(spv::Op op, uint32_t resultType, const std::vector<uint32_t>& args);
    void OpNoResult(spv::Op op, std::initializer_list<uint32_t> args);
    uint32_t Ext(uint32_t resultType, GLSLstd450 inst, std::initializer_list<uint32_t> args);

    // Blocks.
    void Label(uint32_t id);
    void Branch(uint32_t target);
    bool Terminated() const { return terminated_; }
    void Kill(); // OpKill (fragment)
    void BranchConditional(uint32_t cond, uint32_t t, uint32_t f);
    // if (cond) then() [else otherwise()]: a structured selection; the bodies may nest more.
    void If(uint32_t cond, const std::function<void()>& then, const std::function<void()>& otherwise = {},
        spv::SelectionControlMask control = spv::SelectionControlMaskNone);

    std::vector<uint32_t> Finish();

private:
    uint32_t next_ = 1;
    uint32_t glsl_ = 0;
    std::vector<uint32_t> caps_, exts_, entry_, modes_, debug_, annotations_, globals_, functions_;
    std::vector<uint32_t> fnVars_, fnBody_;
    bool terminated_ = true;
    std::map<std::vector<uint32_t>, uint32_t> dedup_;

    static void Emit(std::vector<uint32_t>& s, spv::Op op, const std::vector<uint32_t>& operands);
    static void String(std::vector<uint32_t>& s, const char* str);
    uint32_t Unique(spv::Op op, const std::vector<uint32_t>& operandsAfterResult, bool hasType);
};
}
