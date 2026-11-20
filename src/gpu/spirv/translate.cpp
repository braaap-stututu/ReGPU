#include "gpu/spirv/translate.h"

#include <algorithm>
#include <cfloat>

#include "gpu/spirv/builder.h"

namespace
{
using namespace ucode;
using spirv::Builder;

struct Translator
{
    const Shader& shader;
    const TranslatedShader& info;
    Builder b;
    std::string why;

    // Types.
    uint32_t tVoid = 0, tF = 0, tU = 0, tI = 0, tB = 0, tV4 = 0, tV2 = 0, tUV4 = 0, tB4 = 0;
    uint32_t pFnV4 = 0, pFnF = 0, pFnU = 0, pFnI = 0, pFnB = 0;
    uint32_t pUniV4 = 0, pUniUV4 = 0, pUniU = 0, pInV4 = 0;
    // Globals.
    uint32_t cb = 0, inO = 0, inPcoord = 0, inPos = 0, outColor = 0, outDepth = 0;
    // Function variables.
    uint32_t r = 0, ps = 0, p = 0, a0 = 0, aL = 0, oC[4] = {}, oD = 0;
    uint32_t cfpc = 0, loopCount = 0, loopSaved = 0, loopDepth = 0, callRet = 0, callDepth = 0, guard = 0;
    uint32_t regCount = 1;
    bool exportsDepth = false;
    uint8_t rank[256] = {};

    Translator(const Shader& s, const TranslatedShader& i) : shader(s), info(i) {}

    // Small helpers.
    uint32_t F(float v) { return b.ConstFloat(v); }
    uint32_t U(uint32_t v) { return b.ConstUint(v); }
    uint32_t I(int32_t v) { return b.ConstInt(v); }
    uint32_t Load(uint32_t type, uint32_t ptr) { return b.Op(spv::OpLoad, type, { ptr }); }
    void Store(uint32_t ptr, uint32_t value) { b.OpNoResult(spv::OpStore, { ptr, value }); }
    uint32_t Comp(uint32_t v, uint32_t c, uint32_t type = 0) { return b.Op(spv::OpCompositeExtract, type ? type : tF, { v, c }); }
    uint32_t Splat(uint32_t s) { return b.Op(spv::OpCompositeConstruct, tV4, { s, s, s, s }); }
    uint32_t Vec4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return b.Op(spv::OpCompositeConstruct, tV4, { x, y, z, w }); }
    uint32_t Zero4() { return b.ConstVec4(0, 0, 0, 0); }
    uint32_t One4() { return b.ConstVec4(1, 1, 1, 1); }
    uint32_t Shuffle(uint32_t a, uint32_t c, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return b.Op(spv::OpVectorShuffle, tV4, { a, c, x, y, z, w });
    }
    uint32_t Bin(spv::Op op, uint32_t type, uint32_t x, uint32_t y) { return b.Op(op, type, { x, y }); }
    uint32_t Not(uint32_t x) { return b.Op(spv::OpLogicalNot, tB, { x }); }
    uint32_t And(uint32_t x, uint32_t y) { return Bin(spv::OpLogicalAnd, tB, x, y); }
    uint32_t Or(uint32_t x, uint32_t y) { return Bin(spv::OpLogicalOr, tB, x, y); }
    uint32_t SelF(uint32_t c, uint32_t x, uint32_t y) { return b.Op(spv::OpSelect, tF, { c, x, y }); }
    uint32_t BoolToF(uint32_t c) { return SelF(c, F(1), F(0)); }
    uint32_t Saturate(uint32_t x, bool vec)
    {
        return vec ? b.Ext(tV4, GLSLstd450FClamp, { x, Zero4(), One4() }) : b.Ext(tF, GLSLstd450FClamp, { x, F(0), F(1) });
    }
    uint32_t RegPtr(uint32_t n) { return b.Op(spv::OpAccessChain, pFnV4, { r, I(int32_t(n)) }); }

    // Constant block members (PSConstants: pc[256] 0, pb[2] 1, pl[8] 2, pinfo 3, pinfo2 4).
    uint32_t UniformU(uint32_t member, uint32_t index, uint32_t comp)
    {
        uint32_t ptr = member >= 3 ? b.Op(spv::OpAccessChain, pUniU, { cb, I(int32_t(member)), I(int32_t(comp)) })
                                   : b.Op(spv::OpAccessChain, pUniU, { cb, I(int32_t(member)), I(int32_t(index)), I(int32_t(comp)) });
        return Load(tU, ptr);
    }
    uint32_t Pinfo(uint32_t comp) { return UniformU(3, 0, comp); }
    uint32_t Pinfo2(uint32_t comp) { return Load(tF, b.Op(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniform, tF), { cb, I(4), I(int32_t(comp)) })); }
    uint32_t BoolConstant(uint32_t index)
    {
        const uint32_t w = UniformU(1, index >> 7, (index >> 5) & 3);
        const uint32_t bit = Bin(spv::OpBitwiseAnd, tU, Bin(spv::OpShiftRightLogical, tU, w, U(index & 31)), U(1));
        return Bin(spv::OpINotEqual, tB, bit, U(0));
    }
    uint32_t PinfoBit(uint32_t mask) { return Bin(spv::OpINotEqual, tB, Bin(spv::OpBitwiseAnd, tU, Pinfo(2), U(mask)), U(0)); }

    uint32_t Constant(uint32_t index, uint32_t relative)
    {
        uint32_t idx;
        if (relative)
        {
            const uint32_t reg = Load(tI, relative == 1 ? a0 : aL);
            idx = Bin(spv::OpBitwiseAnd, tI, Bin(spv::OpIAdd, tI, I(int32_t(index & 255)), reg), I(255));
        }
        else
            idx = I(int32_t(info.packedConstants ? rank[index & 255] : (index & 255)));
        return Load(tV4, b.Op(spv::OpAccessChain, pUniV4, { cb, I(0), idx }));
    }

    uint32_t Src(const Source& s)
    {
        uint32_t v = s.isConstant ? Constant(s.reg, s.relative) : Load(tV4, RegPtr(s.reg));
        if (s.abs)
            v = b.Ext(tV4, GLSLstd450FAbs, { v });
        if (!(s.swizzle[0] == 0 && s.swizzle[1] == 1 && s.swizzle[2] == 2 && s.swizzle[3] == 3))
            v = Shuffle(v, v, s.swizzle[0], s.swizzle[1], s.swizzle[2], s.swizzle[3]);
        if (s.negate)
            v = b.Op(spv::OpFNegate, tV4, { v });
        return v;
    }

    // Xenos multiply: 0 x anything = 0.
    uint32_t Mul9(uint32_t x, uint32_t y, bool vec)
    {
        const uint32_t t = vec ? tV4 : tF, bt = vec ? tB4 : tB, z = vec ? Zero4() : F(0);
        const uint32_t c = Bin(spv::OpLogicalOr, bt, Bin(spv::OpFOrdEqual, bt, x, z), Bin(spv::OpFOrdEqual, bt, y, z));
        return b.Op(spv::OpSelect, t, { c, z, Bin(spv::OpFMul, t, x, y) });
    }
    uint32_t Dot9(uint32_t x, uint32_t y)
    {
        const uint32_t m = Mul9(x, y, true);
        uint32_t s = Bin(spv::OpFAdd, tF, Comp(m, 0), Comp(m, 1));
        s = Bin(spv::OpFAdd, tF, s, Comp(m, 2));
        return Bin(spv::OpFAdd, tF, s, Comp(m, 3));
    }
    uint32_t Cmp4(spv::Op op, uint32_t x, uint32_t y) { return b.Op(spv::OpSelect, tV4, { Bin(op, tB4, x, y), One4(), Zero4() }); }

    // Writes value's components in mask to the vec4 behind ptr.
    void WriteMasked(uint32_t ptr, uint32_t value, uint32_t mask)
    {
        mask &= 15;
        if (!mask)
            return;
        if (mask == 15)
        {
            Store(ptr, value);
            return;
        }
        const uint32_t old = Load(tV4, ptr);
        auto sel = [&](uint32_t c) { return ((mask >> c) & 1) ? 4 + c : c; };
        Store(ptr, Shuffle(old, value, sel(0), sel(1), sel(2), sel(3)));
    }

    void KillIf(uint32_t cond)
    {
        // (pinfo.z bit 2: kills skipped, renderer.cpp: the vertex shader wrote no interpolators.)
        b.If(And(cond, Not(PinfoBit(4))), [&] { b.Kill(); });
    }

    bool CheckAlu(const Alu& a)
    {
        if (a.vectorOp == 18)
            return why = "vector op CUBE", false;
        if (a.vectorOp > 29)
            return why = "vector op " + std::to_string(a.vectorOp), false;
        if (a.scalarOp == 41 || a.scalarOp > 50)
            return why = "scalar op " + std::to_string(a.scalarOp), false;
        return true;
    }

    uint32_t VectorOp(const Alu& a, uint32_t A, uint32_t B, uint32_t C)
    {
        switch (a.vectorOp)
        {
        case 0: return Bin(spv::OpFAdd, tV4, A, B);
        case 1: return Mul9(A, B, true);
        case 2: case 29: return b.Ext(tV4, GLSLstd450FMax, { A, B });
        case 3: return b.Ext(tV4, GLSLstd450FMin, { A, B });
        case 4: return Cmp4(spv::OpFOrdEqual, A, B);
        case 5: return Cmp4(spv::OpFOrdGreaterThan, A, B);
        case 6: return Cmp4(spv::OpFOrdGreaterThanEqual, A, B);
        case 7: return Cmp4(spv::OpFUnordNotEqual, A, B);
        case 8: return b.Ext(tV4, GLSLstd450Fract, { A });
        case 9: return b.Ext(tV4, GLSLstd450Trunc, { A });
        case 10: return b.Ext(tV4, GLSLstd450Floor, { A });
        case 11: return Bin(spv::OpFAdd, tV4, Mul9(A, B, true), C);
        case 12: case 13: case 14:
        {
            const spv::Op op = a.vectorOp == 12 ? spv::OpFOrdEqual : a.vectorOp == 13 ? spv::OpFOrdGreaterThanEqual : spv::OpFOrdGreaterThan;
            return b.Op(spv::OpSelect, tV4, { Bin(op, tB4, A, Zero4()), B, C });
        }
        case 15: return Splat(Dot9(A, B));
        case 16:
        {
            const uint32_t z = Zero4();
            return Splat(Dot9(Shuffle(A, z, 0, 1, 2, 7), Shuffle(B, z, 0, 1, 2, 7)));
        }
        case 17:
        {
            const uint32_t z = Zero4();
            return Splat(Bin(spv::OpFAdd, tF, Dot9(Shuffle(A, z, 0, 1, 6, 7), Shuffle(B, z, 0, 1, 6, 7)), Comp(C, 0)));
        }
        case 19:
        {
            const uint32_t m0 = b.Ext(tF, GLSLstd450FMax, { Comp(A, 0), Comp(A, 1) });
            const uint32_t m1 = b.Ext(tF, GLSLstd450FMax, { Comp(A, 2), Comp(A, 3) });
            return Splat(b.Ext(tF, GLSLstd450FMax, { m0, m1 }));
        }
        case 28: return Vec4(F(1), Bin(spv::OpFMul, tF, Comp(A, 1), Comp(B, 1)), Comp(A, 2), Comp(B, 3));
        case 20: case 21: case 22: case 23:
        {
            // SETP_*_PUSH: p = src0.w == 0 && src1.w <op> 0; dest = 0 when src0.x == 0 && src1.x <op> 0, else src0.x + 1.
            static const spv::Op ops[] = { spv::OpFOrdEqual, spv::OpFUnordNotEqual, spv::OpFOrdGreaterThan, spv::OpFOrdGreaterThanEqual };
            const spv::Op op = ops[a.vectorOp - 20];
            Store(p, And(Bin(spv::OpFOrdEqual, tB, Comp(A, 3), F(0)), Bin(op, tB, Comp(B, 3), F(0))));
            const uint32_t c = And(Bin(spv::OpFOrdEqual, tB, Comp(A, 0), F(0)), Bin(op, tB, Comp(B, 0), F(0)));
            return Splat(SelF(c, F(0), Bin(spv::OpFAdd, tF, Comp(A, 0), F(1))));
        }
        case 24: case 25: case 26: case 27:
        {
            static const spv::Op ops[] = { spv::OpFOrdEqual, spv::OpFOrdGreaterThan, spv::OpFOrdGreaterThanEqual, spv::OpFUnordNotEqual };
            const uint32_t k = b.Op(spv::OpAny, tB, { Bin(ops[a.vectorOp - 24], tB4, A, B) });
            KillIf(k);
            return Splat(BoolToF(k));
        }
        }
        return Zero4();
    }

    uint32_t ScalarOp(const Alu& a)
    {
        const uint32_t sv = Src(a.src[2]);
        const uint32_t A = Comp(sv, 3), B = Comp(sv, 0);
        auto PS = [&] { return Load(tF, ps); };
        auto cmp0 = [&](spv::Op op) { return Bin(op, tB, A, F(0)); };
        auto isInf = [&](uint32_t x) { return b.Op(spv::OpIsInf, tB, { x }); };
        auto rcp = [&] { return Bin(spv::OpFDiv, tF, F(1), A); };
        auto clampMax = [&](uint32_t x) { return b.Ext(tF, GLSLstd450FClamp, { x, F(-FLT_MAX), F(FLT_MAX) }); };
        auto setA0 = [&](uint32_t x) {
            const uint32_t i = b.Op(spv::OpConvertFToS, tI, { b.Ext(tF, GLSLstd450Floor, { x }) });
            Store(a0, b.Ext(tI, GLSLstd450SClamp, { i, I(-256), I(255) }));
        };
        auto setp = [&](uint32_t c) {
            Store(p, c);
            return SelF(c, F(0), F(1));
        };
        switch (a.scalarOp)
        {
        case 0: return Bin(spv::OpFAdd, tF, A, B);
        case 1: return Bin(spv::OpFAdd, tF, A, PS());
        case 2: return Mul9(A, B, false);
        case 3: case 4: return Mul9(A, PS(), false);
        case 5: return b.Ext(tF, GLSLstd450FMax, { A, B });
        case 6: return b.Ext(tF, GLSLstd450FMin, { A, B });
        case 7: return BoolToF(cmp0(spv::OpFOrdEqual));
        case 8: return BoolToF(cmp0(spv::OpFOrdGreaterThan));
        case 9: return BoolToF(cmp0(spv::OpFOrdGreaterThanEqual));
        case 10: return BoolToF(cmp0(spv::OpFUnordNotEqual));
        case 11: return b.Ext(tF, GLSLstd450Fract, { A });
        case 12: return b.Ext(tF, GLSLstd450Trunc, { A });
        case 13: return b.Ext(tF, GLSLstd450Floor, { A });
        case 14: return b.Ext(tF, GLSLstd450Exp2, { A });
        case 15: { const uint32_t s = b.Ext(tF, GLSLstd450Log2, { A }); return SelF(isInf(s), F(-FLT_MAX), s); }
        case 16: return b.Ext(tF, GLSLstd450Log2, { A });
        case 17: return clampMax(rcp());
        case 18: { const uint32_t s = rcp(); return SelF(isInf(s), F(0), s); }
        case 19: return rcp();
        case 20: return clampMax(b.Ext(tF, GLSLstd450InverseSqrt, { A }));
        case 21: { const uint32_t s = b.Ext(tF, GLSLstd450InverseSqrt, { A }); return SelF(isInf(s), F(0), s); }
        case 22: return b.Ext(tF, GLSLstd450InverseSqrt, { A });
        case 23: case 24:
        {
            const uint32_t s = b.Ext(tF, GLSLstd450FMax, { A, B });
            setA0(a.scalarOp == 23 ? Bin(spv::OpFAdd, tF, A, F(0.5f)) : A);
            return s;
        }
        case 25: return Bin(spv::OpFSub, tF, A, B);
        case 26: return Bin(spv::OpFSub, tF, A, PS());
        case 27: return setp(cmp0(spv::OpFOrdEqual));
        case 28: return setp(cmp0(spv::OpFUnordNotEqual));
        case 29: return setp(cmp0(spv::OpFOrdGreaterThan));
        case 30: return setp(cmp0(spv::OpFOrdGreaterThanEqual));
        case 31:
        {
            const uint32_t c = Bin(spv::OpFOrdEqual, tB, A, F(1));
            Store(p, c);
            return SelF(c, F(0), SelF(cmp0(spv::OpFOrdEqual), F(1), A));
        }
        case 32:
        {
            const uint32_t t = Bin(spv::OpFSub, tF, A, F(1));
            const uint32_t c = Bin(spv::OpFOrdLessThanEqual, tB, t, F(0));
            Store(p, c);
            return SelF(c, F(0), t);
        }
        case 33: Store(p, b.ConstBool(false)); return F(FLT_MAX);
        case 34: Store(p, cmp0(spv::OpFOrdEqual)); return A;
        case 35: case 36: case 37: case 38: case 39:
        {
            static const spv::Op ops[] = { spv::OpFOrdEqual, spv::OpFOrdGreaterThan, spv::OpFOrdGreaterThanEqual, spv::OpFUnordNotEqual,
                spv::OpFOrdEqual };
            const uint32_t c = Bin(ops[a.scalarOp - 35], tB, A, F(a.scalarOp == 39 ? 1.0f : 0.0f));
            if (shader.pixel)
                KillIf(c);
            return BoolToF(c);
        }
        case 40: return b.Ext(tF, GLSLstd450Sqrt, { A });
        case 42: case 43: case 44: case 45: case 46: case 47:
        {
            // Constant operand (component W of src3's swizzle) with a temp whose index is spread over the opcode
            // and swizzle bits (hlsl.cpp ScalarCode).
            const uint32_t temp = (a.scalarOp & 1) | (uint32_t(a.src3IsTemp) << 1) | (a.src3RawSwizzle & 0x3C);
            uint32_t ca = Comp(Constant(a.src3RawReg, 0), ((a.src3RawSwizzle >> 6) + 3) & 3);
            uint32_t tb = Comp(Load(tV4, RegPtr(temp)), a.src3RawSwizzle & 3);
            if (a.src[2].negate)
            {
                ca = b.Op(spv::OpFNegate, tF, { ca });
                tb = b.Op(spv::OpFNegate, tF, { tb });
            }
            if (a.scalarOp <= 43)
                return Mul9(ca, tb, false);
            return Bin(a.scalarOp <= 45 ? spv::OpFAdd : spv::OpFSub, tF, ca, tb);
        }
        case 48: return b.Ext(tF, GLSLstd450Sin, { A });
        case 49: return b.Ext(tF, GLSLstd450Cos, { A });
        default: return PS(); // 50: retain
        }
    }

    void EmitAluBody(const Alu& a)
    {
        const bool scalarKept = a.scalarOp == 50 && !a.scalarClamp && !a.scalarMask;
        const bool vectorSideEffect = (a.vectorOp >= 20 && a.vectorOp <= 27) || a.vectorOp == 29;
        const bool vectorUnused = !a.vectorMask && !vectorSideEffect && !(a.exportData && a.scalarDestRelative);
        uint32_t v = Zero4(), A = 0;
        if (!vectorUnused)
        {
            A = Src(a.src[0]);
            const uint32_t B = Src(a.src[1]), C = Src(a.src[2]);
            v = VectorOp(a, A, B, C);
            if (a.vectorClamp)
                v = Saturate(v, true);
        }
        uint32_t s = scalarKept ? Load(tF, ps) : ScalarOp(a);
        if (a.vectorOp == 29 && !vectorUnused) // MAXAv: a0 from the first operand's W, after the reads
        {
            const uint32_t i = b.Op(spv::OpConvertFToS, tI, { b.Ext(tF, GLSLstd450Floor, { Bin(spv::OpFAdd, tF, Comp(A, 3), F(0.5f)) }) });
            Store(a0, b.Ext(tI, GLSLstd450SClamp, { i, I(-256), I(255) }));
        }
        if (a.scalarClamp)
            s = Saturate(s, false);
        if (!scalarKept)
            Store(ps, s);
        if (a.exportData)
        {
            uint32_t target = 0;
            if (a.vectorDest == 61)
                target = oD;
            else if (a.vectorDest < 4)
                target = oC[a.vectorDest];
            if (target)
            {
                const uint32_t vm = a.vectorMask & ~a.scalarMask, sm = a.scalarMask & ~a.vectorMask;
                const uint32_t one = a.vectorMask & a.scalarMask;
                const uint32_t zero = a.scalarDestRelative ? (~(a.vectorMask | a.scalarMask) & 15) : 0;
                WriteMasked(target, v, vm);
                if (sm)
                    WriteMasked(target, Splat(s), sm);
                WriteMasked(target, One4(), one);
                WriteMasked(target, Zero4(), zero);
            }
        }
        else
        {
            WriteMasked(RegPtr(a.vectorDest), v, a.vectorMask);
            if (a.scalarMask & 15)
                WriteMasked(RegPtr(a.scalarDest), Splat(s), a.scalarMask);
        }
    }

    void EmitAlu(const Alu& a)
    {
        const bool scalarKept = a.scalarOp == 50 && !a.scalarClamp && !a.scalarMask;
        const bool vectorSideEffect = (a.vectorOp >= 20 && a.vectorOp <= 27) || a.vectorOp == 29;
        const bool vectorUnused = !a.vectorMask && !vectorSideEffect && !(a.exportData && a.scalarDestRelative);
        if (scalarKept && vectorUnused && !a.exportData)
            return;
        if (a.predicated)
        {
            uint32_t c = Load(tB, p);
            if (!a.predCondition)
                c = Not(c);
            b.If(c, [&] { EmitAluBody(a); });
        }
        else
            EmitAluBody(a);
    }

    void EmitExec(const Cf& cf)
    {
        for (uint32_t k = 0; k < cf.count(); k++)
        {
            const uint32_t index = cf.address() + k;
            const uint32_t* w = &shader.words[index * 3];
            if (!((cf.sequence() >> (2 * k)) & 1))
                EmitAlu(DecodeAlu(w));
        }
    }

    uint32_t CfCondition(const Cf& cf)
    {
        // condjmp / ccall: bit 13 unconditional, bit 14 predicate, else a bool constant.
        if ((cf.lo >> 13) & 1)
            return b.ConstBool(true);
        if ((cf.lo >> 14) & 1)
        {
            const uint32_t c = Load(tB, p);
            return cf.condition() ? c : Not(c);
        }
        const uint32_t c = BoolConstant((cf.hi >> 2) & 0xFF);
        return cf.condition() ? c : Not(c);
    }

    uint32_t LoopConstant(uint32_t id) { return UniformU(2, id >> 2, id & 3); }
    uint32_t ArrU(uint32_t arr, uint32_t idx) { return b.Op(spv::OpAccessChain, pFnU, { arr, idx }); }
    uint32_t ArrI(uint32_t arr, uint32_t idx) { return b.Op(spv::OpAccessChain, pFnI, { arr, idx }); }

    void EmitCf(size_t i, bool linear)
    {
        const Cf& cf = shader.cf[i];
        const uint32_t next = cf.IsEnd() ? 0xFFFFu : uint32_t(i + 1);
        if (!linear)
            Store(cfpc, U(next)); // the jumps below overwrite it (nothing in a block reads cfpc before)
        switch (cf.op())
        {
        case CfOp::Nop:
        case CfOp::Alloc:
        case CfOp::MarkVsFetchDone:
            break;
        case CfOp::Exec:
        case CfOp::ExecEnd:
            EmitExec(cf);
            break;
        case CfOp::CondExec:
        case CfOp::CondExecEnd:
        case CfOp::CondExecPredClean:
        case CfOp::CondExecPredCleanEnd:
        {
            uint32_t c = BoolConstant((cf.hi >> 2) & 0xFF);
            if (!cf.condition())
                c = Not(c);
            b.If(c, [&] { EmitExec(cf); });
            break;
        }
        case CfOp::CondExecPred:
        case CfOp::CondExecPredEnd:
        {
            uint32_t c = Load(tB, p);
            if (!cf.condition())
                c = Not(c);
            b.If(c, [&] { EmitExec(cf); });
            break;
        }
        case CfOp::CondJmp:
            Store(cfpc, b.Op(spv::OpSelect, tU, { CfCondition(cf), U(cf.lo & 0x1FFF), U(next) }));
            break;
        case CfOp::LoopStart:
        {
            const uint32_t target = cf.lo & 0x1FFF, id = (cf.lo >> 16) & 0x1F;
            const uint32_t lk = LoopConstant(id);
            const uint32_t d = b.Ext(tU, GLSLstd450UMin, { Load(tU, loopDepth), U(3) });
            Store(ArrI(loopSaved, d), Load(tI, aL));
            const uint32_t count = Bin(spv::OpBitwiseAnd, tU, lk, U(0xFF));
            Store(ArrU(loopCount, d), count);
            if (!((cf.lo >> 13) & 1))
                Store(aL, b.Op(spv::OpBitcast, tI, { Bin(spv::OpBitwiseAnd, tU, Bin(spv::OpShiftRightLogical, tU, lk, U(8)), U(0xFF)) }));
            Store(loopDepth, b.Ext(tU, GLSLstd450UMin, { Bin(spv::OpIAdd, tU, Load(tU, loopDepth), U(1)), U(4) }));
            b.If(Bin(spv::OpIEqual, tB, count, U(0)), [&] {
                const uint32_t dd = Bin(spv::OpISub, tU, Load(tU, loopDepth), U(1));
                Store(loopDepth, dd);
                Store(aL, Load(tI, ArrI(loopSaved, b.Ext(tU, GLSLstd450UMin, { dd, U(3) }))));
                Store(cfpc, U(target));
            });
            break;
        }
        case CfOp::LoopEnd:
        {
            const uint32_t target = cf.lo & 0x1FFF, id = (cf.lo >> 16) & 0x1F;
            const uint32_t depth = Load(tU, loopDepth);
            const uint32_t d = b.Op(spv::OpSelect, tU, { Bin(spv::OpUGreaterThan, tB, depth, U(0)),
                Bin(spv::OpISub, tU, b.Ext(tU, GLSLstd450UMin, { depth, U(4) }), U(1)), U(0) });
            const uint32_t cnt = Bin(spv::OpISub, tU, Load(tU, ArrU(loopCount, d)), U(1));
            Store(ArrU(loopCount, d), cnt);
            const uint32_t step = Bin(spv::OpShiftRightArithmetic, tI, b.Op(spv::OpBitcast, tI, { Bin(spv::OpShiftLeftLogical, tU, LoopConstant(id), U(8)) }), I(24));
            Store(aL, Bin(spv::OpIAdd, tI, Load(tI, aL), step));
            uint32_t again = Bin(spv::OpINotEqual, tB, cnt, U(0));
            if ((cf.lo >> 21) & 1)
            {
                uint32_t brk = Load(tB, p);
                if (!cf.condition())
                    brk = Not(brk);
                again = And(again, Not(brk));
            }
            b.If(again, [&] { Store(cfpc, U(target)); },
                [&] {
                    Store(loopDepth, d);
                    Store(aL, Load(tI, ArrI(loopSaved, d)));
                });
            break;
        }
        case CfOp::CondCall:
        {
            const uint32_t target = cf.lo & 0x1FFF;
            const uint32_t c = And(CfCondition(cf), Bin(spv::OpULessThan, tB, Load(tU, callDepth), U(4)));
            b.If(c, [&] {
                const uint32_t depth = Load(tU, callDepth);
                Store(ArrU(callRet, depth), U(uint32_t(i + 1)));
                Store(callDepth, Bin(spv::OpIAdd, tU, depth, U(1)));
                Store(cfpc, U(target));
            });
            break;
        }
        case CfOp::Return:
            b.If(Bin(spv::OpUGreaterThan, tB, Load(tU, callDepth), U(0)), [&] {
                const uint32_t depth = Bin(spv::OpISub, tU, Load(tU, callDepth), U(1));
                Store(callDepth, depth);
                Store(cfpc, Load(tU, ArrU(callRet, depth)));
            });
            break;
        default:
            break;
        }
    }

    // Checks everything the program reaches before emitting anything (so a fallback costs no module).
    bool Scan(std::vector<bool>& reachable, bool& linear, bool& forward, bool& loops)
    {
        linear = true;
        loops = false;
        for (const Cf& cf : shader.cf)
        {
            const CfOp op = cf.op();
            linear = linear && op != CfOp::CondJmp && op != CfOp::LoopStart && op != CfOp::LoopEnd && op != CfOp::CondCall && op != CfOp::Return;
            loops = loops || op == CfOp::LoopStart;
        }
        forward = !linear;
        for (size_t i = 0; forward && i < shader.cf.size(); i++)
        {
            const CfOp op = shader.cf[i].op();
            if (op == CfOp::LoopStart || op == CfOp::LoopEnd || op == CfOp::CondCall || op == CfOp::Return)
                forward = false;
            else if (op == CfOp::CondJmp && (shader.cf[i].lo & 0x1FFF) <= i)
                forward = false;
        }
        reachable.assign(shader.cf.size(), true);
        if (linear)
        {
            // Linear: every block up to the first end.
            for (size_t i = 0; i < shader.cf.size(); i++)
                if (shader.cf[i].IsEnd())
                {
                    std::fill(reachable.begin() + i + 1, reachable.end(), false);
                    break;
                }
        }
        else
        {
            reachable.assign(shader.cf.size(), false);
            std::vector<size_t> pending{ 0 };
            while (!pending.empty())
            {
                const size_t i = pending.back();
                pending.pop_back();
                if (i >= reachable.size() || reachable[i])
                    continue;
                reachable[i] = true;
                const Cf& cf = shader.cf[i];
                const CfOp op = cf.op();
                if (op == CfOp::CondJmp || op == CfOp::CondCall || op == CfOp::LoopStart || op == CfOp::LoopEnd)
                    pending.push_back(cf.lo & 0x1FFF);
                const bool unconditionalJump = op == CfOp::CondJmp && ((cf.lo >> 13) & 1);
                if (!cf.IsEnd() && op != CfOp::Return && !unconditionalJump)
                    pending.push_back(i + 1);
            }
        }
        uint32_t maxReg = 0;
        auto useReg = [&](uint32_t n) { maxReg = std::max(maxReg, n & 127); };
        for (size_t i = 0; i < shader.cf.size(); i++)
        {
            if (!reachable[i])
                continue;
            const Cf& cf = shader.cf[i];
            switch (cf.op())
            {
            case CfOp::Nop: case CfOp::Alloc: case CfOp::MarkVsFetchDone: case CfOp::Exec: case CfOp::ExecEnd: case CfOp::CondExec:
            case CfOp::CondExecEnd: case CfOp::CondExecPredClean: case CfOp::CondExecPredCleanEnd: case CfOp::CondExecPred:
            case CfOp::CondExecPredEnd: case CfOp::CondJmp: case CfOp::LoopStart: case CfOp::LoopEnd: case CfOp::CondCall: case CfOp::Return:
                break;
            default:
                return why = "control flow op " + std::to_string(unsigned(cf.op())), false;
            }
            if (!cf.IsExec())
                continue;
            for (uint32_t k = 0; k < cf.count(); k++)
            {
                const uint32_t index = cf.address() + k;
                if ((index + 1) * 3 > shader.words.size())
                    return why = "instruction out of range", false;
                const uint32_t* w = &shader.words[index * 3];
                if ((cf.sequence() >> (2 * k)) & 1)
                {
                    const Fetch f = DecodeFetch(w);
                    return why = f.opcode == 0 ? "vertex fetch" : f.opcode == 1 ? "texture fetch" : "fetch op " + std::to_string(f.opcode), false;
                }
                const Alu a = DecodeAlu(w);
                if (!CheckAlu(a))
                    return false;
                for (const Source& s : a.src)
                    if (!s.isConstant)
                        useReg(s.reg);
                if (a.scalarOp >= 42 && a.scalarOp <= 47)
                    useReg((a.scalarOp & 1) | (uint32_t(a.src3IsTemp) << 1) | (a.src3RawSwizzle & 0x3C));
                if (a.exportData)
                    exportsDepth = exportsDepth || a.vectorDest == 61;
                else
                {
                    useReg(a.vectorDest);
                    useReg(a.scalarDest);
                }
            }
        }
        regCount = std::max<uint32_t>(maxReg + 1, 16);
        return true;
    }

    bool Run(std::vector<uint32_t>& words)
    {
        if (!shader.pixel)
            return why = "vertex shader", false;
        std::vector<bool> reachable;
        bool linear, forward, loops;
        if (!Scan(reachable, linear, forward, loops))
            return false;
        for (uint32_t i = 0, n = 0; i < 256; i++)
            rank[i] = uint8_t(n), n += (info.constantMask[i >> 6] >> (i & 63)) & 1;

        b.Capability(spv::CapabilityShader);
        tVoid = b.TypeVoid();
        tF = b.TypeFloat();
        tU = b.TypeUint();
        tI = b.TypeInt();
        tB = b.TypeBool();
        tV4 = b.TypeVector(tF, 4);
        tV2 = b.TypeVector(tF, 2);
        tUV4 = b.TypeVector(tU, 4);
        tB4 = b.TypeVector(tB, 4);
        pFnV4 = b.TypePointer(spv::StorageClassFunction, tV4);
        pFnF = b.TypePointer(spv::StorageClassFunction, tF);
        pFnU = b.TypePointer(spv::StorageClassFunction, tU);
        pFnI = b.TypePointer(spv::StorageClassFunction, tI);
        pFnB = b.TypePointer(spv::StorageClassFunction, tB);
        pUniV4 = b.TypePointer(spv::StorageClassUniform, tV4);
        pUniUV4 = b.TypePointer(spv::StorageClassUniform, tUV4);
        pUniU = b.TypePointer(spv::StorageClassUniform, tU);
        pInV4 = b.TypePointer(spv::StorageClassInput, tV4);

        // PSConstants (hlsl.cpp's preamble; DX layout = std140 for these members). Only the members up to pinfo2.
        const uint32_t aC = b.TypeArray(tV4, 256), aB = b.TypeArray(tUV4, 2), aL8 = b.TypeArray(tUV4, 8);
        b.Decorate(aC, spv::DecorationArrayStride, { 16 });
        b.Decorate(aB, spv::DecorationArrayStride, { 16 });
        b.Decorate(aL8, spv::DecorationArrayStride, { 16 });
        const uint32_t block = b.TypeStruct({ aC, aB, aL8, tUV4, tV4 });
        b.Decorate(block, spv::DecorationBlock);
        const uint32_t offsets[] = { 0, 4096, 4128, 4256, 4272 };
        for (uint32_t m = 0; m < 5; m++)
            b.MemberDecorate(block, m, spv::DecorationOffset, { offsets[m] });
        b.Name(block, "PSConstants");
        cb = b.GlobalVariable(b.TypePointer(spv::StorageClassUniform, block), spv::StorageClassUniform);
        b.Decorate(cb, spv::DecorationDescriptorSet, { 0 });
        b.Decorate(cb, spv::DecorationBinding, { 1 });

        // VSOut: SV_Position, o[16] (TEXCOORD0, location 0-15), pcoord (TEXCOORD16, location 16).
        inPos = b.GlobalVariable(pInV4, spv::StorageClassInput);
        b.Decorate(inPos, spv::DecorationBuiltIn, { spv::BuiltInFragCoord });
        const uint32_t aO = b.TypeArray(tV4, 16);
        inO = b.GlobalVariable(b.TypePointer(spv::StorageClassInput, aO), spv::StorageClassInput);
        b.Decorate(inO, spv::DecorationLocation, { 0 });
        inPcoord = b.GlobalVariable(b.TypePointer(spv::StorageClassInput, tV2), spv::StorageClassInput);
        b.Decorate(inPcoord, spv::DecorationLocation, { 16 });
        outColor = b.GlobalVariable(b.TypePointer(spv::StorageClassOutput, tV4), spv::StorageClassOutput);
        b.Decorate(outColor, spv::DecorationLocation, { 0 });
        std::vector<uint32_t> io{ inPos, inO, inPcoord, outColor };
        if (exportsDepth)
        {
            outDepth = b.GlobalVariable(b.TypePointer(spv::StorageClassOutput, tF), spv::StorageClassOutput);
            b.Decorate(outDepth, spv::DecorationBuiltIn, { spv::BuiltInFragDepth });
            io.push_back(outDepth);
        }

        const uint32_t entry = b.Id();
        b.Name(entry, "main");
        b.EntryPoint(spv::ExecutionModelFragment, entry, "main", io);
        b.ExecutionMode(entry, spv::ExecutionModeOriginUpperLeft);
        if (exportsDepth)
            b.ExecutionMode(entry, spv::ExecutionModeDepthReplacing);
        b.BeginFunction(entry, tVoid, b.TypeFunction(tVoid));

        r = b.Variable(b.TypePointer(spv::StorageClassFunction, b.TypeArray(tV4, regCount)));
        ps = b.Variable(pFnF);
        p = b.Variable(pFnB);
        a0 = b.Variable(pFnI);
        aL = b.Variable(pFnI);
        for (uint32_t& c : oC)
            c = b.Variable(pFnV4);
        oD = b.Variable(pFnV4);

        // Prologue: interpolators in r0-r15, the generated parameter in the register pinfo.y names.
        const uint32_t pos = Load(tV4, inPos);
        const uint32_t pcoord = Load(tV2, inPcoord);
        const uint32_t posXY = Shuffle(pos, pos, 0, 1, 0, 1);
        for (uint32_t n = 0; n < regCount; n++)
            Store(RegPtr(n), n < 16 ? Load(tV4, b.Op(spv::OpAccessChain, pInV4, { inO, I(int32_t(n)) })) : Zero4());
        {
            const uint32_t floorPos = b.Ext(tV4, GLSLstd450Floor, { posXY });
            const uint32_t xy = b.Op(spv::OpSelect, tV4, { Splat4B(PinfoBit(65536)), floorPos, posXY });
            const uint32_t scaled = Bin(spv::OpFMul, tV4, xy, Vec4(Pinfo2(1), Pinfo2(2), F(1), F(1)));
            const uint32_t gen = Vec4(Comp(scaled, 0), Comp(scaled, 1), Comp(pcoord, 0), Comp(pcoord, 1));
            const uint32_t which = Pinfo(1);
            for (uint32_t n = 0; n < 16; n++)
                b.If(Bin(spv::OpIEqual, tB, which, U(n)), [&] { Store(RegPtr(n), gen); });
        }
        for (uint32_t c : oC)
            Store(c, Zero4());
        Store(oD, Vec4(Comp(pos, 2), F(0), F(0), F(0)));
        Store(ps, F(0));
        Store(p, b.ConstBool(false));
        Store(a0, I(0));
        Store(aL, I(0));

        if (linear)
        {
            for (size_t i = 0; i < shader.cf.size(); i++)
                if (reachable[i])
                    EmitCf(i, true);
        }
        else
        {
            cfpc = b.Variable(pFnU);
            Store(cfpc, U(0));
            auto blocks = [&] {
                for (size_t i = 0; i < shader.cf.size(); i++)
                    if (reachable[i])
                        b.If(Bin(spv::OpIEqual, tB, Load(tU, cfpc), U(uint32_t(i))), [&] { EmitCf(i, false); });
            };
            if (forward)
                blocks();
            else
            {
                // do { blocks in order (a jump forward runs its target in the same pass, a jump back the next) }
                // while (cfpc != end && guard < limit).
                const uint32_t aU4 = b.TypePointer(spv::StorageClassFunction, b.TypeArray(tU, 4));
                const uint32_t aI4 = b.TypePointer(spv::StorageClassFunction, b.TypeArray(tI, 4));
                loopCount = b.Variable(aU4);
                loopSaved = b.Variable(aI4);
                callRet = b.Variable(aU4);
                loopDepth = b.Variable(pFnU);
                callDepth = b.Variable(pFnU);
                guard = b.Variable(pFnU);
                const uint32_t zu4 = b.ConstComposite(b.TypeArray(tU, 4), { U(0), U(0), U(0), U(0) });
                Store(loopCount, zu4);
                Store(callRet, zu4);
                Store(loopSaved, b.ConstComposite(b.TypeArray(tI, 4), { I(0), I(0), I(0), I(0) }));
                Store(loopDepth, U(0));
                Store(callDepth, U(0));
                Store(guard, U(0));
                const uint32_t header = b.Id(), body = b.Id(), cont = b.Id(), merge = b.Id();
                b.Label(header);
                b.OpNoResult(spv::OpLoopMerge, { merge, cont, uint32_t(spv::LoopControlMaskNone) });
                b.Branch(body);
                b.Label(body);
                blocks();
                b.Branch(cont);
                b.Label(cont);
                const uint32_t g = Bin(spv::OpIAdd, tU, Load(tU, guard), U(1));
                Store(guard, g);
                const uint32_t more = And(Bin(spv::OpULessThan, tB, g, U(loops ? 65536 : 1024)), Bin(spv::OpINotEqual, tB, Load(tU, cfpc), U(0xFFFF)));
                b.BranchConditional(more, header, merge);
                b.Label(merge);
            }
        }

        // Epilogue (hlsl.cpp TranslateShader): alpha test, alpha to mask, exponent bias, target clamps.
        {
            const uint32_t c0 = Load(tV4, oC[0]);
            const uint32_t alpha = Comp(c0, 3), ref = Pinfo2(0), fn = Pinfo(0);
            auto is = [&](uint32_t k) { return Bin(spv::OpIEqual, tB, fn, U(k)); };
            uint32_t fail = is(0);
            const spv::Op ops[] = { spv::OpFOrdLessThan, spv::OpFOrdEqual, spv::OpFOrdLessThanEqual, spv::OpFOrdGreaterThan, spv::OpFUnordNotEqual,
                spv::OpFOrdGreaterThanEqual };
            for (uint32_t k = 1; k <= 6; k++)
                fail = Or(fail, And(is(k), Not(Bin(ops[k - 1], tB, alpha, ref))));
            b.If(fail, [&] { b.Kill(); });
            b.If(PinfoBit(64), [&] {
                // Alpha to mask, dithered.
                const uint32_t qx = Bin(spv::OpBitwiseAnd, tU, b.Op(spv::OpConvertFToU, tU, { Comp(pos, 0) }), U(1));
                const uint32_t qy = Bin(spv::OpBitwiseAnd, tU, b.Op(spv::OpConvertFToU, tU, { Comp(pos, 1) }), U(1));
                const uint32_t q = Bin(spv::OpBitwiseOr, tU, Bin(spv::OpShiftLeftLogical, tU, qx, U(1)), qy);
                const uint32_t shift = Bin(spv::OpIAdd, tU, U(8), Bin(spv::OpIMul, tU, q, U(2)));
                const uint32_t bits = Bin(spv::OpBitwiseAnd, tU, Bin(spv::OpShiftRightLogical, tU, Pinfo(2), shift), U(3));
                const uint32_t thr = Bin(spv::OpFSub, tF, F(1), Bin(spv::OpFMul, tF, b.Op(spv::OpConvertUToF, tF, { bits }), F(0.25f)));
                b.If(Not(Bin(spv::OpFOrdGreaterThanEqual, tB, Comp(Load(tV4, oC[0]), 3), thr)), [&] { b.Kill(); });
            });
            const uint32_t bias = b.Ext(tF, GLSLstd450Exp2, { b.Op(spv::OpConvertSToF, tF, { b.Op(spv::OpBitcast, tI, { Pinfo(3) }) }) });
            Store(oC[0], b.Op(spv::OpVectorTimesScalar, tV4, { Load(tV4, oC[0]), bias }));
            b.If(PinfoBit(8), [&] { Store(oC[0], b.Ext(tV4, GLSLstd450FClamp, { Load(tV4, oC[0]), b.ConstVec4(-32, -32, -32, -32), b.ConstVec4(32, 32, 32, 32) })); });
            b.If(PinfoBit(32), [&] {
                const uint32_t a = Comp(Load(tV4, oC[0]), 3);
                const uint32_t rounded = Bin(spv::OpFDiv, tF, b.Ext(tF, GLSLstd450RoundEven, { Bin(spv::OpFMul, tF, Saturate(a, false), F(3)) }), F(3));
                WriteMasked(oC[0], Splat(rounded), 8);
            });
            const uint32_t ceiling = Pinfo2(3);
            b.If(Bin(spv::OpFOrdGreaterThan, tB, ceiling, F(0)),
                [&] {
                    const uint32_t c = Load(tV4, oC[0]);
                    const uint32_t clamped = b.Ext(tV4, GLSLstd450FClamp, { c, Zero4(), Splat(ceiling) });
                    WriteMasked(oC[0], clamped, 7);
                },
                [&] { b.If(Bin(spv::OpFOrdLessThan, tB, ceiling, F(0)), [&] { Store(oC[0], Saturate(Load(tV4, oC[0]), true)); }); });
            b.If(PinfoBit(0x80000u), [&] {
                const uint32_t c = Load(tV4, oC[0]);
                WriteMasked(oC[0], b.Op(spv::OpVectorTimesScalar, tV4, { c, Comp(c, 3) }), 7);
            });
            b.If(PinfoBit(0x100000u), [&] {
                const uint32_t c = Load(tV4, oC[0]);
                WriteMasked(oC[0], Splat(Bin(spv::OpFMul, tF, Comp(c, 3), Comp(c, 3))), 8);
            });
            Store(outColor, Load(tV4, oC[0]));
            if (exportsDepth)
                Store(outDepth, Saturate(Comp(Load(tV4, oD), 0), false));
        }
        b.EndFunction();
        words = b.Finish();
        return true;
    }

    uint32_t Splat4B(uint32_t c) { return b.Op(spv::OpCompositeConstruct, tB4, { c, c, c, c }); }
};
}

bool TranslateSpirv(const ucode::Shader& shader, const TranslatedShader& hlslInfo, std::vector<uint32_t>& words, std::string& why)
{
    words.clear();
    if (!shader.valid)
    {
        why = "invalid microcode";
        return false;
    }
    Translator t(shader, hlslInfo);
    if (!t.Run(words))
    {
        why = t.why;
        words.clear();
        return false;
    }
    return true;
}
