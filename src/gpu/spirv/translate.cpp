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
    uint32_t regLod = 0; // setTexLOD's value (hlsl.cpp's regLod)
    // Texture fetch resources, made at the first fetch (InitTextures).
    bool texInit = false;
    uint32_t tV3 = 0, tIV2 = 0, tSampler = 0, tImg2 = 0, tImgC = 0, tImg3 = 0, gtex = 0, gcube = 0, gvol = 0, gsamp = 0;
    bool exportsDepth = false;
    uint8_t rank[256] = {};

    const SpirvOptions& options;
    Translator(const Shader& s, const TranslatedShader& i, const SpirvOptions& o) : shader(s), info(i), options(o) {}
    const SpirvFetchSpec* Spec(uint32_t slot) const
    {
        for (const SpirvFetchSpec& s : options.specs)
            if (s.slot == slot)
                return &s;
        return nullptr;
    }

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
        if (a.vectorOp > 29)
            return why = "vector op " + std::to_string(a.vectorOp), false;
        if (a.scalarOp == 41 || a.scalarOp > 50)
            return why = "scalar op " + std::to_string(a.scalarOp), false;
        return true;
    }

    // CUBE (hlsl.cpp's CubeOp; the operands are v.zzxy and v.yxzz, the first carries v): (tc, sc, 2 * major axis, face).
    uint32_t CubeOp(uint32_t A)
    {
        const uint32_t vx = Comp(A, 2), vy = Comp(A, 3), vz = Comp(A, 0);
        const uint32_t mx = b.Ext(tF, GLSLstd450FAbs, { vx }), my = b.Ext(tF, GLSLstd450FAbs, { vy }), mz = b.Ext(tF, GLSLstd450FAbs, { vz });
        auto ge = [&](uint32_t x, uint32_t y) { return Bin(spv::OpFOrdGreaterThanEqual, tB, x, y); };
        auto neg = [&](uint32_t x) { return b.Op(spv::OpFNegate, tF, { x }); };
        auto lt0 = [&](uint32_t x) { return Bin(spv::OpFOrdLessThan, tB, x, F(0)); };
        const uint32_t zMajor = And(ge(mz, mx), ge(mz, my)), yMajor = ge(my, mx);
        const uint32_t ma = SelF(zMajor, vz, SelF(yMajor, vy, vx));
        const uint32_t face = SelF(zMajor, SelF(lt0(vz), F(5), F(4)), SelF(yMajor, SelF(lt0(vy), F(3), F(2)), SelF(lt0(vx), F(1), F(0))));
        const uint32_t sc = SelF(zMajor, SelF(lt0(vz), neg(vx), vx), SelF(yMajor, vx, SelF(lt0(vx), vz, neg(vz))));
        const uint32_t tc = SelF(zMajor, neg(vy), SelF(yMajor, SelF(lt0(vy), neg(vz), vz), neg(vy)));
        return Vec4(tc, sc, Bin(spv::OpFMul, tF, F(2), b.Ext(tF, GLSLstd450FAbs, { ma })), face);
    }

    uint32_t VectorOp(const Alu& a, uint32_t A, uint32_t B, uint32_t C)
    {
        switch (a.vectorOp)
        {
        case 18: return CubeOp(A);
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


    // ---- Texture fetches (hlsl.cpp's TFetch / TFetchRaw, the same results) ----
    uint32_t UBits(uint32_t v, uint32_t shift, uint32_t mask)
    {
        return Bin(spv::OpBitwiseAnd, tU, shift ? Bin(spv::OpShiftRightLogical, tU, v, U(shift)) : v, U(mask));
    }
    uint32_t UBit(uint32_t v, uint32_t shift) { return Bin(spv::OpINotEqual, tB, UBits(v, shift, 1), U(0)); }
    uint32_t UEq(uint32_t v, uint32_t c) { return Bin(spv::OpIEqual, tB, v, U(c)); }
    uint32_t SelU(uint32_t c, uint32_t x, uint32_t y) { return b.Op(spv::OpSelect, tU, { c, x, y }); }
    uint32_t Fract(uint32_t x) { return b.Ext(tF, GLSLstd450Fract, { x }); }
    uint32_t FAbs(uint32_t x) { return b.Ext(tF, GLSLstd450FAbs, { x }); }
    uint32_t FMax(uint32_t x, uint32_t y) { return b.Ext(tF, GLSLstd450FMax, { x, y }); }
    uint32_t FMin(uint32_t x, uint32_t y) { return b.Ext(tF, GLSLstd450FMin, { x, y }); }
    uint32_t FAdd(uint32_t x, uint32_t y) { return Bin(spv::OpFAdd, tF, x, y); }
    uint32_t FSub(uint32_t x, uint32_t y) { return Bin(spv::OpFSub, tF, x, y); }
    uint32_t FMulS(uint32_t x, uint32_t y) { return Bin(spv::OpFMul, tF, x, y); }
    uint32_t FDiv(uint32_t x, uint32_t y) { return Bin(spv::OpFDiv, tF, x, y); }
    uint32_t FLt(uint32_t x, uint32_t y) { return Bin(spv::OpFOrdLessThan, tB, x, y); }
    uint32_t FGt(uint32_t x, uint32_t y) { return Bin(spv::OpFOrdGreaterThan, tB, x, y); }
    uint32_t FGe(uint32_t x, uint32_t y) { return Bin(spv::OpFOrdGreaterThanEqual, tB, x, y); }
    uint32_t Vec2(uint32_t x, uint32_t y) { return b.Op(spv::OpCompositeConstruct, tV2, { x, y }); }
    uint32_t Vec3(uint32_t x, uint32_t y, uint32_t z) { return b.Op(spv::OpCompositeConstruct, tV3, { x, y, z }); }

    void InitTextures()
    {
        if (texInit)
            return;
        texInit = true;
        b.Capability(spv::CapabilityImageQuery);
        b.Capability(spv::CapabilityRuntimeDescriptorArrayEXT);
        b.Extension("SPV_EXT_descriptor_indexing");
        tV3 = b.TypeVector(tF, 3);
        tIV2 = b.TypeVector(tI, 2);
        tSampler = b.Type(spv::OpTypeSampler, {});
        auto image = [&](spv::Dim d) {
            return b.Type(spv::OpTypeImage, { tF, uint32_t(d), 0, 0, 0, 1, uint32_t(spv::ImageFormatUnknown) });
        };
        tImg2 = image(spv::Dim2D);
        tImgC = image(spv::DimCube);
        tImg3 = image(spv::Dim3D);
        // Bindless arrays (gpu/vk/bindings.h): set 1 gtex, set 2 gcube, set 3 gvol, binding 0 each; the samplers s0-s7 as
        // set 0 binding 200.
        auto array = [&](uint32_t type, uint32_t set, const char* name) {
            const uint32_t a = b.Type(spv::OpTypeRuntimeArray, { type });
            const uint32_t v = b.GlobalVariable(b.TypePointer(spv::StorageClassUniformConstant, a), spv::StorageClassUniformConstant);
            b.Decorate(v, spv::DecorationDescriptorSet, { set });
            b.Decorate(v, spv::DecorationBinding, { 0 });
            b.Name(v, name);
            return v;
        };
        gtex = array(tImg2, 1, "gtex");
        gcube = array(tImgC, 2, "gcube");
        gvol = array(tImg3, 3, "gvol");
        gsamp = b.GlobalVariable(b.TypePointer(spv::StorageClassUniformConstant, b.TypeArray(tSampler, 8)), spv::StorageClassUniformConstant);
        b.Decorate(gsamp, spv::DecorationDescriptorSet, { 0 });
        b.Decorate(gsamp, spv::DecorationBinding, { 200 });
        b.Name(gsamp, "gSamplers");
    }

    // A sampled image from a bindless array's element and sampler s (0 linear wrap, 1 linear clamp, 2 point clamp,
    // 3 point wrap, 4-7 anisotropic wrap).
    uint32_t SampledImage(uint32_t array, uint32_t imageType, uint32_t index, uint32_t s, uint32_t* imageOut = nullptr)
    {
        const uint32_t img = Load(imageType, b.Op(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniformConstant, imageType), { array, index }));
        if (imageOut)
            *imageOut = img;
        const uint32_t smp = Load(tSampler, b.Op(spv::OpAccessChain, b.TypePointer(spv::StorageClassUniformConstant, tSampler), { gsamp, s }));
        return b.Op(spv::OpSampledImage, b.Type(spv::OpTypeSampledImage, { imageType }), { img, smp });
    }
    // level: sampled at max(bias, 0) (the vertex path, a pixel fetch without computed LOD); else with the bias.
    uint32_t Sample(uint32_t si, uint32_t coord, bool level, uint32_t bias)
    {
        if (level)
            return b.Op(spv::OpImageSampleExplicitLod, tV4, { si, coord, uint32_t(spv::ImageOperandsLodMask), FMax(bias, F(0)) });
        return b.Op(spv::OpImageSampleImplicitLod, tV4, { si, coord, uint32_t(spv::ImageOperandsBiasMask), bias });
    }

    // Xenia's PWLGammaToLinear.
    uint32_t PwlGammaToLinear(uint32_t g)
    {
        g = Saturate(g, false);
        const uint32_t hi = FGe(g, F(96.0f / 255.0f)), top = FGe(g, F(192.0f / 255.0f)), mid = FGe(g, F(64.0f / 255.0f));
        const uint32_t scale = SelF(hi, SelF(top, F(8.0f / 1024.0f), F(4.0f / 1024.0f)), SelF(mid, F(2.0f / 1024.0f), F(1.0f / 1024.0f)));
        const uint32_t offset = SelF(hi, SelF(top, F(-1024.0f), F(-256.0f)), SelF(mid, F(-64.0f), F(0.0f)));
        uint32_t l = FAdd(FMulS(g, FMulS(F(255.0f * 1024.0f), scale)), offset);
        l = FAdd(l, b.Ext(tF, GLSLstd450Trunc, { FMulS(l, scale) }));
        return FDiv(l, F(1023.0f));
    }

    // hlsl.cpp's Fixed16Fetch, per component.
    uint32_t Fixed16(uint32_t v, uint32_t info)
    {
        const uint32_t copy = UBits(info, 4, 3), mode = UBits(info, 14, 3), comps = Bin(spv::OpShiftLeftLogical, tU, U(1), UBits(info, 2, 3));
        uint32_t c[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            const uint32_t x = Comp(v, i);
            const uint32_t stored = SelF(UEq(copy, 1), b.Ext(tF, GLSLstd450RoundEven, { FMulS(Saturate(x, false), F(65535.0f)) }),
                SelF(UEq(copy, 2), b.Ext(tF, GLSLstd450RoundEven, { FMulS(b.Ext(tF, GLSLstd450FClamp, { x, F(-1), F(1) }), F(32767.0f)) }), x));
            const uint32_t asSigned = FSub(stored, SelF(FGe(stored, F(32768.0f)), F(65536.0f), F(0)));
            const uint32_t asUnsigned = FAdd(stored, SelF(FLt(stored, F(0)), F(65536.0f), F(0)));
            const uint32_t r = SelF(UEq(mode, 1), FDiv(asUnsigned, F(65535.0f)),
                SelF(UEq(mode, 2), FMax(FDiv(asSigned, F(32767.0f)), F(-1)), SelF(UEq(mode, 3), asUnsigned, asSigned)));
            const uint32_t use = i == 0 ? b.ConstBool(true) : Bin(spv::OpUGreaterThanEqual, tB, comps, U(i == 1 ? 2 : 4));
            c[i] = SelF(use, r, x);
        }
        const uint32_t converted = Vec4(c[0], c[1], c[2], c[3]);
        return b.Op(spv::OpSelect, tV4, { Splat4B(UBit(info, 6)), converted, v });
    }

    // TFetch(slot, coord, denorm, level, bias, texel) with the instruction's filter override (filt 0 point, 1 linear,
    // -1 the constant's) and gLevelWrap = level.
    // lean: hlsl.cpp's lean TFetch2D (REGPU_LEAN_FETCH) for a 2D fetch: the constant's own filter (filt -1), no point
    // snap, its mirror rule (bit 30/31 only with 28/29); 3D and cube as the full path.
    uint32_t TexFetch(uint32_t slot, uint32_t coord, bool denorm, bool level, uint32_t bias, float ox, float oy, int filt, bool lean = false)
    {
        InitTextures();
        const uint32_t ti = Load(tUV4, b.Op(spv::OpAccessChain, pUniUV4, { cb, I(5), I(int32_t(slot)) }));
        const uint32_t tb = Load(tV4, b.Op(spv::OpAccessChain, pUniV4, { cb, I(6), I(int32_t(slot)) }));
        // (A specialised slot's words are literals: the driver folds every path its constant doesn't take.)
        const SpirvFetchSpec* spec = Spec(slot);
        const uint32_t info = spec ? U(spec->info) : Comp(ti, 0, tU), w = spec ? U(spec->w) : Comp(ti, 3, tU);
        const uint32_t index = b.Op(spv::OpBitcast, tU, { Comp(tb, 1) });
        const uint32_t dim = UBits(w, 0, 0xFF);
        const uint32_t result = b.Variable(pFnV4), border = b.Variable(pFnB);
        Store(border, b.ConstBool(false));
        const uint32_t cx = Comp(coord, 0), cy = Comp(coord, 1), cz = Comp(coord, 2);
        b.If(UEq(dim, 2), [&] {
            // 3D: the constant's own mode (no override), linear clamp at a level.
            const uint32_t vmode = UBits(info, 0, 3);
            Store(result, Sample(SampledImage(gvol, tImg3, index, level ? U(1) : vmode), Vec3(cx, cy, cz), level, bias));
        }, [&] {
            b.If(UEq(dim, 3), [&] {
                // Cube: (s, t, face) from the CUBE instruction back to a direction (CubeDirection).
                const uint32_t sc = FMulS(FSub(cx, F(1.5f)), F(2)), tc = FMulS(FSub(cy, F(1.5f)), F(2));
                const uint32_t face = b.Op(spv::OpConvertFToU, tU, { b.Ext(tF, GLSLstd450RoundEven, { cz }) });
                const uint32_t nsc = b.Op(spv::OpFNegate, tF, { sc }), ntc = b.Op(spv::OpFNegate, tF, { tc });
                auto pick = [&](uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4, uint32_t v5) {
                    return SelF(UEq(face, 0), v0, SelF(UEq(face, 1), v1, SelF(UEq(face, 2), v2, SelF(UEq(face, 3), v3, SelF(UEq(face, 4), v4, v5)))));
                };
                const uint32_t dx = pick(F(1), F(-1), sc, sc, sc, nsc), dy = pick(ntc, ntc, F(1), F(-1), ntc, ntc),
                               dz = pick(nsc, sc, tc, ntc, F(1), F(-1));
                Store(result, Sample(SampledImage(gcube, tImgC, index, U(1)), Vec3(dx, dy, dz), level, bias));
            }, [&] {
                // 2D.
                const uint32_t sx = b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 1, tU), 0, 0xFFFF), U(1) }) });
                const uint32_t sy = b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 2, tU), 0, 0xFFFF), U(1) }) });
                uint32_t u = denorm ? FDiv(FAdd(cx, F(ox)), sx) : FAdd(cx, FDiv(F(ox), sx));
                uint32_t v = denorm ? FDiv(FAdd(cy, F(oy)), sy) : FAdd(cy, FDiv(F(oy), sy));
                auto mirror = [&](uint32_t x) { return FSub(F(1), FAbs(FSub(FMulS(Fract(FMulS(x, F(0.5f))), F(2)), F(1)))); };
                if (lean)
                {
                    u = SelF(UBit(info, 28), SelF(UBit(info, 30), FAbs(u), mirror(u)), u);
                    v = SelF(UBit(info, 29), SelF(UBit(info, 31), FAbs(v), mirror(v)), v);
                }
                else
                {
                    u = SelF(UBit(info, 28), mirror(u), u);
                    v = SelF(UBit(info, 29), mirror(v), v);
                    u = SelF(UBit(info, 30), FAbs(u), u);
                    v = SelF(UBit(info, 31), FAbs(v), v);
                }
                const uint32_t outside = Or(Or(FLt(u, F(0)), FGt(u, F(1))), Or(FLt(v, F(0)), FGt(v, F(1))));
                b.If(And(UBit(info, 16), outside), [&] {
                    Store(border, b.ConstBool(true));
                    Store(result, b.Op(spv::OpSelect, tV4, { Splat4B(UBit(info, 17)), One4(), Zero4() }));
                }, [&] {
                    const uint32_t hx = FDiv(F(0.5f), sx), hy = FDiv(F(0.5f), sy);
                    uint32_t uu = SelF(UBit(info, 18), b.Ext(tF, GLSLstd450FClamp, { u, hx, FSub(F(1), hx) }), u);
                    uint32_t vv = SelF(UBit(info, 19), b.Ext(tF, GLSLstd450FClamp, { v, hy, FSub(F(1), hy) }), v);
                    const uint32_t sub = FGt(Comp(tb, 2), F(0));
                    uu = SelF(sub, FMulS(uu, Comp(tb, 2)), uu);
                    vv = SelF(sub, FMulS(vv, Comp(tb, 3)), vv);
                    // FetchMode: the constant's mode, its filter replaced by the instruction's.
                    uint32_t mode = UBits(info, 0, 3);
                    if (filt == 0 || filt == 1)
                    {
                        const uint32_t wrap = Or(UEq(mode, 0), UEq(mode, 3));
                        mode = filt == 0 ? SelU(wrap, U(3), U(2)) : SelU(wrap, U(0), U(1));
                    }
                    // The sampler: at a level the mode itself (gLevelWrap); else modes 1-3 theirs, mode 0 the anisotropy's
                    // (tinfo.x bits 20-22: 1-4 = s4-s7) or linear wrap.
                    uint32_t s = mode;
                    if (!level)
                    {
                        const uint32_t aniso = UBits(info, 20, 7);
                        const uint32_t anisoS = SelU(And(Bin(spv::OpUGreaterThanEqual, tB, aniso, U(1)), Bin(spv::OpULessThanEqual, tB, aniso, U(4))),
                            Bin(spv::OpIAdd, tU, aniso, U(3)), U(0));
                        s = SelU(UEq(mode, 0), anisoS, mode);
                    }
                    uint32_t img = 0;
                    const uint32_t si = SampledImage(gtex, tImg2, index, s, &img);
                    if (!level && !lean)
                    {
                        // A point fetch within 1/16 of a texel of a boundary goes onto it, 1/512 back (pinfo.z bit 17).
                        const uint32_t snap = And(Or(UEq(mode, 2), UEq(mode, 3)), PinfoBit(131072));
                        const uint32_t size = b.Op(spv::OpImageQuerySizeLod, tIV2, { img, I(0) });
                        const uint32_t dw = b.Op(spv::OpConvertSToF, tF, { Comp(size, 0, tI) }), dh = b.Op(spv::OpConvertSToF, tF, { Comp(size, 1, tI) });
                        auto snapAxis = [&](uint32_t x, uint32_t d) {
                            const uint32_t at = FMulS(x, d), edge = b.Ext(tF, GLSLstd450RoundEven, { at });
                            const uint32_t near = FLt(FAbs(FSub(at, edge)), F(1.0f / 16.0f));
                            return FDiv(SelF(near, FSub(edge, F(1.0f / 512.0f)), at), d);
                        };
                        uu = SelF(snap, snapAxis(uu, dw), uu);
                        vv = SelF(snap, snapAxis(vv, dh), vv);
                    }
                    Store(result, Sample(si, Vec2(uu, vv), level, bias));
                });
            });
        });
        // TFetch: the 16-bit fixed conversion (not of the border colour), gamma, exponent, the constant's swizzle.
        uint32_t t = Load(tV4, result);
        t = b.Op(spv::OpSelect, tV4, { Splat4B(Load(tB, border)), t, Fixed16(t, info) });
        const uint32_t gamma = UBits(info, 24, 15);
        uint32_t c[4];
        for (uint32_t i = 0; i < 4; i++)
            c[i] = SelF(UBit(gamma, i), PwlGammaToLinear(Comp(t, i)), Comp(t, i));
        t = Vec4(c[0], c[1], c[2], c[3]);
        const uint32_t expAdjust = Bin(spv::OpShiftRightArithmetic, tI, b.Op(spv::OpBitcast, tI, { Bin(spv::OpShiftLeftLogical, tU, info, U(18)) }), I(26));
        t = b.Op(spv::OpVectorTimesScalar, tV4, { t, b.Ext(tF, GLSLstd450Exp2, { b.Op(spv::OpConvertSToF, tF, { expAdjust }) }) });
        for (uint32_t i = 0; i < 4; i++)
        {
            const uint32_t sel = UBits(w, 8 + 3 * i, 7);
            const uint32_t picked = b.Op(spv::OpVectorExtractDynamic, tF, { t, b.Ext(tU, GLSLstd450UMin, { sel, U(3) }) });
            c[i] = SelF(Bin(spv::OpULessThan, tB, sel, U(4)), picked, SelF(UEq(sel, 5), F(1), F(0)));
        }
        const uint32_t swizzled = Vec4(c[0], c[1], c[2], c[3]);
        return b.Op(spv::OpSelect, tV4, { Splat4B(UBit(w, 31)), swizzled, t });
    }

    // FetchVariant's TFetchPlain: the constant's sampler (its filter replaced by the instruction's), the coordinates and
    // nothing else (the slot's constant is plain this draw: the full path would give the same).
    uint32_t TexFetchPlain(uint32_t slot, uint32_t coord, bool denorm, uint32_t bias, float ox, float oy, int filt)
    {
        InitTextures();
        const uint32_t ti = Load(tUV4, b.Op(spv::OpAccessChain, pUniUV4, { cb, I(5), I(int32_t(slot)) }));
        const uint32_t tb = Load(tV4, b.Op(spv::OpAccessChain, pUniV4, { cb, I(6), I(int32_t(slot)) }));
        const SpirvFetchSpec* spec = Spec(slot);
        const uint32_t info = spec ? U(spec->info) : Comp(ti, 0, tU);
        uint32_t mode = UBits(info, 0, 3);
        if (filt == 0 || filt == 1)
        {
            const uint32_t wrap = Or(UEq(mode, 0), UEq(mode, 3));
            mode = filt == 0 ? SelU(wrap, U(3), U(2)) : SelU(wrap, U(0), U(1));
        }
        const uint32_t aniso = UBits(info, 20, 7);
        const uint32_t anisoS = SelU(And(Bin(spv::OpUGreaterThanEqual, tB, aniso, U(1)), Bin(spv::OpULessThanEqual, tB, aniso, U(4))),
            Bin(spv::OpIAdd, tU, aniso, U(3)), U(0));
        const uint32_t s = SelU(UEq(mode, 0), anisoS, mode);
        const uint32_t sx = b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 1, tU), 0, 0xFFFF), U(1) }) });
        const uint32_t sy = b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 2, tU), 0, 0xFFFF), U(1) }) });
        const uint32_t cx = Comp(coord, 0), cy = Comp(coord, 1);
        const uint32_t u = denorm ? FDiv(FAdd(cx, F(ox)), sx) : FAdd(cx, FDiv(F(ox), sx));
        const uint32_t v = denorm ? FDiv(FAdd(cy, F(oy)), sy) : FAdd(cy, FDiv(F(oy), sy));
        const uint32_t index = b.Op(spv::OpBitcast, tU, { Comp(tb, 1) });
        return Sample(SampledImage(gtex, tImg2, index, s), Vec2(u, v), false, bias);
    }

    // hlsl.cpp's DestSwizzleWrite: per component 0-3 a value component, 4 zero, 5 one, 6-7 kept.
    void DestSwizzleWrite(uint32_t dest, uint32_t swizzle, uint32_t value)
    {
        const uint32_t ptr = RegPtr(dest), old = Load(tV4, ptr);
        uint32_t c[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            const uint32_t sel = (swizzle >> (3 * i)) & 7;
            c[i] = sel < 4 ? Comp(value, sel) : sel == 4 ? F(0) : sel == 5 ? F(1) : Comp(old, i);
        }
        Store(ptr, Vec4(c[0], c[1], c[2], c[3]));
    }

    void EmitFetchBody(const Fetch& f)
    {
        const uint32_t src = Load(tV4, RegPtr(f.src));
        auto comp = [&](uint32_t i) { return Comp(src, (f.srcSwizzle >> (2 * i)) & 3); };
        if (f.opcode == 1)
        {
            // A 1D fetch reads x at the row's centre (REGPU_NO_1D_TEXTURES: as 2D).
            static const bool no1d = getenv("REGPU_NO_1D_TEXTURES") != nullptr;
            InitTextures();
            const uint32_t coord = f.dimension == 0 && !no1d ? Vec3(comp(0), F(0.5f), F(0)) : Vec3(comp(0), comp(1), comp(2));
            static const bool noRegisterLod = getenv("REGPU_NO_REGISTER_LOD") != nullptr;
            const bool level = !noRegisterLod && !f.useComputedLod;
            const uint32_t tb = Load(tV4, b.Op(spv::OpAccessChain, pUniV4, { cb, I(6), I(int32_t(f.constIndex)) }));
            uint32_t bias = FAdd(F(f.lodBias), Comp(tb, 0));
            if (!noRegisterLod && f.useRegisterLod)
                bias = FAdd(Load(tF, regLod), bias);
            static const bool noOverride = getenv("REGPU_NO_FETCH_FILTER_OVERRIDE") != nullptr;
            const int filt = !noOverride && (f.magFilter == 0 || f.magFilter == 1) ? int(f.magFilter) : -1;
            // A 2D fetch at computed LOD without a register LOD is what the HLSL path makes a lean fetch (REGPU_LEAN_FETCH:
            // 1 TFetch2D, 2 TFetch2DV2) and what its fetch variants replace: the same rules here, so both paths draw alike.
            const bool leanCall = f.dimension == 1 && !level && !(!noRegisterLod && f.useRegisterLod);
            const int leanLevel = LeanFetchLevel();
            static const bool noOverrideLean = getenv("REGPU_NO_FETCH_FILTER_OVERRIDE") != nullptr;
            const int filtV2 = !noOverrideLean && (f.magFilter == 0 || f.magFilter == 1) ? int(f.magFilter) : -1;
            uint32_t t;
            if (((options.plainFetchMask >> f.constIndex) & 1) && leanCall)
                t = TexFetchPlain(f.constIndex, coord, f.denormalized, bias, f.offsetX, f.offsetY, leanLevel >= 2 ? filtV2 : -1);
            else if (leanCall && leanLevel >= 2)
            {
                // TFetch2DV2: a plain constant (tinfo.x bit 7) one sample with the instruction's filter, else TFetch2D.
                InitTextures();
                const SpirvFetchSpec* spec = Spec(f.constIndex);
                const uint32_t ti = Load(tUV4, b.Op(spv::OpAccessChain, pUniUV4, { cb, I(5), I(int32_t(f.constIndex)) }));
                const uint32_t info = spec ? U(spec->info) : Comp(ti, 0, tU);
                const uint32_t out = b.Variable(pFnV4);
                b.If(UBit(info, 7), [&] { Store(out, TexFetchPlain(f.constIndex, coord, f.denormalized, bias, f.offsetX, f.offsetY, filtV2)); },
                    [&] { Store(out, TexFetch(f.constIndex, coord, f.denormalized, false, bias, f.offsetX, f.offsetY, -1, true)); });
                t = Load(tV4, out);
            }
            else if (leanCall && leanLevel == 1)
                t = TexFetch(f.constIndex, coord, f.denormalized, false, bias, f.offsetX, f.offsetY, -1, true);
            else
                t = TexFetch(f.constIndex, coord, f.denormalized, level, bias, f.offsetX, f.offsetY, filt);
            DestSwizzleWrite(f.dest, f.destSwizzle, t);
        }
        else if (f.opcode == 24)
            Store(regLod, comp(0)); // setTexLOD
        else if (f.opcode == 18)
        {
            // getGradients: (ddx(a), ddy(a), ddx(b), ddy(b)).
            const uint32_t a = comp(0), c = comp(1);
            DestSwizzleWrite(f.dest, f.destSwizzle, Vec4(b.Op(spv::OpDPdx, tF, { a }), b.Op(spv::OpDPdy, tF, { a }), b.Op(spv::OpDPdx, tF, { c }),
                                                        b.Op(spv::OpDPdy, tF, { c })));
        }
        else if (f.opcode == 19 && !getenv("REGPU_NO_TEXTURE_WEIGHTS"))
        {
            // getWeights: the bilinear weights at these coordinates (0 for point filtering).
            uint32_t x = comp(0), y = comp(1);
            if (!f.denormalized)
            {
                const uint32_t ti = Load(tUV4, b.Op(spv::OpAccessChain, pUniUV4, { cb, I(5), I(int32_t(f.constIndex)) }));
                x = FMulS(x, b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 1, tU), 0, 0xFFFF), U(1) }) }));
                y = FMulS(y, b.Op(spv::OpConvertUToF, tF, { b.Ext(tU, GLSLstd450UMax, { UBits(Comp(ti, 2, tU), 0, 0xFFFF), U(1) }) }));
            }
            x = FSub(FAdd(x, F(f.offsetX)), F(0.5f));
            y = FSub(FAdd(y, F(f.offsetY)), F(0.5f));
            DestSwizzleWrite(f.dest, f.destSwizzle, f.magFilter == 0 ? Zero4() : Vec4(Fract(x), Fract(y), F(0), F(0)));
        }
        else
            DestSwizzleWrite(f.dest, f.destSwizzle, Zero4()); // other texture queries
    }

    void EmitFetch(const Fetch& f)
    {
        if (f.predicated)
        {
            uint32_t c = Load(tB, p);
            if (!f.predCondition)
                c = Not(c);
            b.If(c, [&] { EmitFetchBody(f); });
        }
        else
            EmitFetchBody(f);
    }

    void EmitExec(const Cf& cf)
    {
        for (uint32_t k = 0; k < cf.count(); k++)
        {
            const uint32_t index = cf.address() + k;
            const uint32_t* w = &shader.words[index * 3];
            if (!((cf.sequence() >> (2 * k)) & 1))
                EmitAlu(DecodeAlu(w));
            else
                EmitFetch(DecodeFetch(w));
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
                    if (f.opcode == 0)
                        return why = "vertex fetch", false;
                    if (f.opcode == 1 && f.dimension > 3)
                        return why = "texture fetch dimension " + std::to_string(f.dimension), false;
                    if (f.opcode == 1 || f.opcode == 24 || f.opcode == 19)
                        useReg(f.src), useReg(f.dest);
                    else
                        useReg(f.dest), useReg(f.src);
                    continue;
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

        // PSConstants (hlsl.cpp's preamble; DX layout = std140 for these members): pc, pb, pl, pinfo, pinfo2, tinfo[32],
        // tbias[32].
        const uint32_t aC = b.TypeArray(tV4, 256), aB = b.TypeArray(tUV4, 2), aL8 = b.TypeArray(tUV4, 8), aTi = b.TypeArray(tUV4, 32),
                       aTb = b.TypeArray(tV4, 32);
        for (uint32_t a : { aC, aB, aL8, aTi, aTb })
            b.Decorate(a, spv::DecorationArrayStride, { 16 });
        const uint32_t block = b.TypeStruct({ aC, aB, aL8, tUV4, tV4, aTi, aTb });
        b.Decorate(block, spv::DecorationBlock);
        const uint32_t offsets[] = { 0, 4096, 4128, 4256, 4272, 4288, 4800 };
        for (uint32_t m = 0; m < 7; m++)
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
        regLod = b.Variable(pFnF);
        Store(regLod, F(0));

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
            if (!options.noAlphaTest)
                b.If(fail, [&] { b.Kill(); });
            if (!options.noAlphaTest)
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

bool TranslateSpirv(const ucode::Shader& shader, const TranslatedShader& hlslInfo, std::vector<uint32_t>& words, std::string& why,
    const SpirvOptions& options)
{
    words.clear();
    if (!shader.valid)
    {
        why = "invalid microcode";
        return false;
    }
    Translator t(shader, hlslInfo, options);
    if (!t.Run(words))
    {
        why = t.why;
        words.clear();
        return false;
    }
    return true;
}
