#include "gpu/spirv/builder.h"

#include <cstring>

namespace spirv
{
Builder::Builder()
{
    glsl_ = Id();
}

void Builder::Emit(std::vector<uint32_t>& s, spv::Op op, const std::vector<uint32_t>& operands)
{
    s.push_back((uint32_t(operands.size() + 1) << 16) | uint32_t(op));
    s.insert(s.end(), operands.begin(), operands.end());
}

void Builder::String(std::vector<uint32_t>& s, const char* str)
{
    const size_t n = strlen(str);
    for (size_t i = 0; i <= n; i += 4)
    {
        uint32_t w = 0;
        for (size_t k = 0; k < 4 && i + k < n; k++)
            w |= uint32_t(uint8_t(str[i + k])) << (8 * k);
        s.push_back(w);
    }
}

void Builder::Capability(spv::Capability c)
{
    Emit(caps_, spv::OpCapability, { uint32_t(c) });
}

void Builder::EntryPoint(spv::ExecutionModel model, uint32_t function, const char* name, const std::vector<uint32_t>& interface)
{
    std::vector<uint32_t> ops{ uint32_t(model), function };
    String(ops, name);
    ops.insert(ops.end(), interface.begin(), interface.end());
    Emit(entry_, spv::OpEntryPoint, ops);
}

void Builder::ExecutionMode(uint32_t function, spv::ExecutionMode mode, std::initializer_list<uint32_t> args)
{
    std::vector<uint32_t> ops{ function, uint32_t(mode) };
    ops.insert(ops.end(), args);
    Emit(modes_, spv::OpExecutionMode, ops);
}

void Builder::Name(uint32_t id, const char* name)
{
    std::vector<uint32_t> ops{ id };
    String(ops, name);
    Emit(debug_, spv::OpName, ops);
}

void Builder::Decorate(uint32_t id, spv::Decoration d, std::initializer_list<uint32_t> args)
{
    std::vector<uint32_t> ops{ id, uint32_t(d) };
    ops.insert(ops.end(), args);
    Emit(annotations_, spv::OpDecorate, ops);
}

void Builder::MemberDecorate(uint32_t type, uint32_t member, spv::Decoration d, std::initializer_list<uint32_t> args)
{
    std::vector<uint32_t> ops{ type, member, uint32_t(d) };
    ops.insert(ops.end(), args);
    Emit(annotations_, spv::OpMemberDecorate, ops);
}

uint32_t Builder::Unique(spv::Op op, const std::vector<uint32_t>& rest, bool hasType)
{
    // Key: opcode, then the operands (type first when there is one) without the result id.
    std::vector<uint32_t> key{ uint32_t(op) };
    key.insert(key.end(), rest.begin(), rest.end());
    if (auto it = dedup_.find(key); it != dedup_.end())
        return it->second;
    const uint32_t id = Id();
    std::vector<uint32_t> ops;
    if (hasType)
    {
        ops.push_back(rest[0]);
        ops.push_back(id);
        ops.insert(ops.end(), rest.begin() + 1, rest.end());
    }
    else
    {
        ops.push_back(id);
        ops.insert(ops.end(), rest.begin(), rest.end());
    }
    Emit(globals_, op, ops);
    dedup_[key] = id;
    return id;
}

uint32_t Builder::Type(spv::Op op, const std::vector<uint32_t>& operands) { return Unique(op, operands, false); }

void Builder::Extension(const char* name)
{
    std::vector<uint32_t> ops;
    String(ops, name);
    Emit(exts_, spv::OpExtension, ops);
}

uint32_t Builder::TypeVoid() { return Unique(spv::OpTypeVoid, {}, false); }
uint32_t Builder::TypeBool() { return Unique(spv::OpTypeBool, {}, false); }
uint32_t Builder::TypeFloat() { return Unique(spv::OpTypeFloat, { 32 }, false); }
uint32_t Builder::TypeUint() { return Unique(spv::OpTypeInt, { 32, 0 }, false); }
uint32_t Builder::TypeInt() { return Unique(spv::OpTypeInt, { 32, 1 }, false); }
uint32_t Builder::TypeVector(uint32_t component, uint32_t count) { return Unique(spv::OpTypeVector, { component, count }, false); }
uint32_t Builder::TypeArray(uint32_t element, uint32_t length) { return Unique(spv::OpTypeArray, { element, ConstUint(length) }, false); }
uint32_t Builder::TypePointer(spv::StorageClass sc, uint32_t type) { return Unique(spv::OpTypePointer, { uint32_t(sc), type }, false); }

uint32_t Builder::TypeStruct(const std::vector<uint32_t>& members)
{
    const uint32_t id = Id();
    std::vector<uint32_t> ops{ id };
    ops.insert(ops.end(), members.begin(), members.end());
    Emit(globals_, spv::OpTypeStruct, ops);
    return id;
}

uint32_t Builder::TypeFunction(uint32_t ret, const std::vector<uint32_t>& params)
{
    std::vector<uint32_t> rest{ ret };
    rest.insert(rest.end(), params.begin(), params.end());
    return Unique(spv::OpTypeFunction, rest, false);
}

uint32_t Builder::ConstFloat(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);
    return Unique(spv::OpConstant, { TypeFloat(), bits }, true);
}
uint32_t Builder::ConstUint(uint32_t v) { return Unique(spv::OpConstant, { TypeUint(), v }, true); }
uint32_t Builder::ConstInt(int32_t v) { return Unique(spv::OpConstant, { TypeInt(), uint32_t(v) }, true); }
uint32_t Builder::ConstBool(bool v) { return Unique(v ? spv::OpConstantTrue : spv::OpConstantFalse, { TypeBool() }, true); }

uint32_t Builder::ConstComposite(uint32_t type, const std::vector<uint32_t>& parts)
{
    std::vector<uint32_t> rest{ type };
    rest.insert(rest.end(), parts.begin(), parts.end());
    return Unique(spv::OpConstantComposite, rest, true);
}

uint32_t Builder::ConstVec4(float x, float y, float z, float w)
{
    return ConstComposite(TypeVector(TypeFloat(), 4), { ConstFloat(x), ConstFloat(y), ConstFloat(z), ConstFloat(w) });
}

uint32_t Builder::GlobalVariable(uint32_t pointerType, spv::StorageClass sc)
{
    const uint32_t id = Id();
    Emit(globals_, spv::OpVariable, { pointerType, id, uint32_t(sc) });
    return id;
}

void Builder::BeginFunction(uint32_t function, uint32_t ret, uint32_t fnType)
{
    Emit(functions_, spv::OpFunction, { ret, function, uint32_t(spv::FunctionControlMaskNone), fnType });
    Emit(functions_, spv::OpLabel, { Id() });
    fnVars_.clear();
    fnBody_.clear();
    terminated_ = false;
}

std::vector<uint32_t> Builder::BeginFunction(uint32_t function, uint32_t ret, uint32_t fnType, const std::vector<uint32_t>& paramTypes)
{
    Emit(functions_, spv::OpFunction, { ret, function, uint32_t(spv::FunctionControlMaskNone), fnType });
    std::vector<uint32_t> ids;
    for (uint32_t t : paramTypes)
    {
        ids.push_back(Id());
        Emit(functions_, spv::OpFunctionParameter, { t, ids.back() });
    }
    Emit(functions_, spv::OpLabel, { Id() });
    fnVars_.clear();
    fnBody_.clear();
    terminated_ = false;
    return ids;
}

void Builder::ReturnValue(uint32_t value)
{
    if (terminated_)
        Label(Id());
    Emit(fnBody_, spv::OpReturnValue, { value });
    terminated_ = true;
}

uint32_t Builder::Variable(uint32_t pointerType)
{
    const uint32_t id = Id();
    Emit(fnVars_, spv::OpVariable, { pointerType, id, uint32_t(spv::StorageClassFunction) });
    return id;
}

void Builder::EndFunction()
{
    if (!terminated_)
        Emit(fnBody_, spv::OpReturn, {});
    terminated_ = true;
    functions_.insert(functions_.end(), fnVars_.begin(), fnVars_.end());
    functions_.insert(functions_.end(), fnBody_.begin(), fnBody_.end());
    Emit(functions_, spv::OpFunctionEnd, {});
}

uint32_t Builder::OpV(spv::Op op, uint32_t resultType, const std::vector<uint32_t>& args)
{
    if (terminated_)
        Label(Id()); // unreachable code after a kill: its own (unreachable) block
    const uint32_t id = Id();
    std::vector<uint32_t> ops{ resultType, id };
    ops.insert(ops.end(), args.begin(), args.end());
    Emit(fnBody_, op, ops);
    return id;
}

uint32_t Builder::Op(spv::Op op, uint32_t resultType, std::initializer_list<uint32_t> args)
{
    return OpV(op, resultType, std::vector<uint32_t>(args));
}

void Builder::OpNoResult(spv::Op op, std::initializer_list<uint32_t> args)
{
    if (terminated_)
        Label(Id());
    Emit(fnBody_, op, std::vector<uint32_t>(args));
}

uint32_t Builder::Ext(uint32_t resultType, GLSLstd450 inst, std::initializer_list<uint32_t> args)
{
    std::vector<uint32_t> ops{ glsl_, uint32_t(inst) };
    ops.insert(ops.end(), args);
    return OpV(spv::OpExtInst, resultType, ops);
}

void Builder::Label(uint32_t id)
{
    if (!terminated_)
        Emit(fnBody_, spv::OpBranch, { id });
    Emit(fnBody_, spv::OpLabel, { id });
    terminated_ = false;
}

void Builder::Branch(uint32_t target)
{
    if (!terminated_)
        Emit(fnBody_, spv::OpBranch, { target });
    terminated_ = true;
}

void Builder::BranchConditional(uint32_t cond, uint32_t t, uint32_t f)
{
    if (terminated_)
        Label(Id());
    Emit(fnBody_, spv::OpBranchConditional, { cond, t, f });
    terminated_ = true;
}

void Builder::Kill()
{
    if (terminated_)
        Label(Id());
    Emit(fnBody_, spv::OpKill, {});
    terminated_ = true;
}

void Builder::If(uint32_t cond, const std::function<void()>& then, const std::function<void()>& otherwise, spv::SelectionControlMask control)
{
    if (terminated_)
        Label(Id());
    const uint32_t thenLabel = Id(), merge = Id(), elseLabel = otherwise ? Id() : merge;
    Emit(fnBody_, spv::OpSelectionMerge, { merge, uint32_t(control) });
    Emit(fnBody_, spv::OpBranchConditional, { cond, thenLabel, elseLabel });
    terminated_ = true;
    Label(thenLabel);
    then();
    Branch(merge);
    if (otherwise)
    {
        Label(elseLabel);
        otherwise();
        Branch(merge);
    }
    Label(merge);
}

std::vector<uint32_t> Builder::Finish()
{
    std::vector<uint32_t> w{ spv::MagicNumber, 0x00010300, 0, 0, 0 };
    w.insert(w.end(), caps_.begin(), caps_.end());
    w.insert(w.end(), exts_.begin(), exts_.end());
    {
        std::vector<uint32_t> ops{ glsl_ };
        String(ops, "GLSL.std.450");
        Emit(w, spv::OpExtInstImport, ops);
    }
    Emit(w, spv::OpMemoryModel, { uint32_t(spv::AddressingModelLogical), uint32_t(spv::MemoryModelGLSL450) });
    w.insert(w.end(), entry_.begin(), entry_.end());
    w.insert(w.end(), modes_.begin(), modes_.end());
    w.insert(w.end(), debug_.begin(), debug_.end());
    w.insert(w.end(), annotations_.begin(), annotations_.end());
    w.insert(w.end(), globals_.begin(), globals_.end());
    w.insert(w.end(), functions_.begin(), functions_.end());
    w[3] = next_;
    return w;
}
}
