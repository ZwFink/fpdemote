// Proteus JIT pass plugin (Prepend): keeps double storage, does double
// computation in float. Every rewrite has the shape fptrunc operands -> float
// operation -> fpext result; Proteus's O3 then removes the back-to-back
// ext/trunc pairs.
//
// Three module passes:
//   demote-fp       fadd/fsub/fmul/fdiv/frem/fneg/fcmp and
//                   llvm.{sqrt,fabs,fma,fmuladd}.f64.
//   demote-fp-math  calls to __ocml_<fn>_f64 and to the matching llvm.<fn>.f64
//                   intrinsics become llvm.<fn>.f32 intrinsics, which the
//                   AMDGPU backend lowers without a math library.
//   demote-fp-atomics  atomicrmw fadd/fsub on double becomes a cmpxchg loop
//                   that adds in float, so accumulators behave like float
//                   accumulators while memory stays double.
// Run all three with the pipeline "demote-fp,demote-fp-math,demote-fp-atomics".
//
// The plugin carries its own LLVM copy (libproteus.so hides its LLVM symbols),
// so it must never inspect, create or fold FP constants: all casts are built
// with IRBuilder<NoFolder> and constant folding is left to Proteus's O3.
//
// Environment:
//   QS_DEMOTE_ONLY  comma-separated substrings of (mangled) function names;
//                   when set, only matching functions are demoted.
//   QS_DEMOTE_SKIP  comma-separated substrings of function names to leave in
//                   double, in addition to the built-in exclusions.
//   QS_DEMOTE_MATH_SKIP  comma-separated math function names (e.g. "sin,pow")
//                   that demote-fp-math leaves in double.
//   QS_DEMOTE_VERBOSE=1  per-function counts on stderr.
#include <algorithm>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace {
using Builder = IRBuilder<NoFolder>;

std::vector<std::string> splitEnv(const char *Name) {
  std::vector<std::string> Out;
  const char *V = std::getenv(Name);
  if (!V)
    return Out;
  SmallVector<StringRef, 8> Parts;
  StringRef(V).split(Parts, ',', -1, false);
  for (StringRef P : Parts)
    Out.push_back(P.trim().str());
  return Out;
}

bool matchesAny(StringRef Name, const std::vector<std::string> &Patterns) {
  for (const std::string &P : Patterns)
    if (Name.contains(P))
      return true;
  return false;
}

bool isVerbose() {
  const char *V = std::getenv("QS_DEMOTE_VERBOSE");
  return V && V[0] == '1';
}

// The math library keeps its own double algorithms, and the RNG must produce
// the same stream in both runs.
bool isAlwaysExcluded(StringRef Name) {
  return Name.starts_with("__ocml") || Name.starts_with("__ockl") ||
         Name.contains("rngSample");
}

struct FunctionFilter {
  std::vector<std::string> Only = splitEnv("QS_DEMOTE_ONLY");
  std::vector<std::string> Skip = splitEnv("QS_DEMOTE_SKIP");

  bool isSelected(const Function &F) const {
    StringRef Name = F.getName();
    if (F.isDeclaration() || isAlwaysExcluded(Name) || matchesAny(Name, Skip))
      return false;
    return Only.empty() || matchesAny(Name, Only);
  }
};

Value *toFloat(Builder &B, Value *V) {
  return B.CreateFPTrunc(V, B.getFloatTy());
}

void copyFastMathFlags(Builder &B, const Instruction &I) {
  if (auto *FPOp = dyn_cast<FPMathOperator>(&I))
    B.setFastMathFlags(FPOp->getFastMathFlags());
}

// Replaces a double-returning call with the float overload of intrinsic ID:
// double arguments are truncated, other arguments (the i32 of ldexp) are
// passed through.
Value *callFloatIntrinsic(CallBase &Call, Intrinsic::ID ID) {
  Builder B(&Call);
  copyFastMathFlags(B, Call);
  SmallVector<Value *, 3> Args;
  SmallVector<Type *, 2> Overloads = {B.getFloatTy()};
  for (Value *A : Call.args()) {
    if (A->getType()->isDoubleTy()) {
      Args.push_back(toFloat(B, A));
    } else {
      Args.push_back(A);
      Overloads.push_back(A->getType());
    }
  }
  Function *F32 = Intrinsic::getDeclaration(Call.getModule(), ID, Overloads);
  return B.CreateFPExt(B.CreateCall(F32, Args), B.getDoubleTy());
}

void replace(Instruction &I, Value *New) {
  New->takeName(&I);
  I.replaceAllUsesWith(New);
}

void eraseAll(SmallVectorImpl<Instruction *> &Dead) {
  for (Instruction *I : Dead)
    I->eraseFromParent();
}

// --- demote-fp ------------------------------------------------------------

struct Counts {
  unsigned Arith = 0, Cmp = 0, Intr = 0;
  unsigned total() const { return Arith + Cmp + Intr; }
};

bool demoteInstruction(Instruction &I, Counts &C) {
  Builder B(&I);
  copyFastMathFlags(B, I);

  Value *New = nullptr;
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    if (!BO->getType()->isDoubleTy())
      return false;
    switch (BO->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::FRem:
      break;
    default:
      return false;
    }
    Value *R = B.CreateBinOp(BO->getOpcode(), toFloat(B, BO->getOperand(0)),
                             toFloat(B, BO->getOperand(1)));
    New = B.CreateFPExt(R, B.getDoubleTy());
    ++C.Arith;
  } else if (I.getOpcode() == Instruction::FNeg) {
    if (!I.getType()->isDoubleTy())
      return false;
    Value *R = B.CreateFNeg(toFloat(B, I.getOperand(0)));
    New = B.CreateFPExt(R, B.getDoubleTy());
    ++C.Arith;
  } else if (auto *Cmp = dyn_cast<FCmpInst>(&I)) {
    if (!Cmp->getOperand(0)->getType()->isDoubleTy())
      return false;
    New = B.CreateFCmp(Cmp->getPredicate(), toFloat(B, Cmp->getOperand(0)),
                       toFloat(B, Cmp->getOperand(1)));
    ++C.Cmp;
  } else if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    if (!II->getType()->isDoubleTy())
      return false;
    Intrinsic::ID ID = II->getIntrinsicID();
    switch (ID) {
    case Intrinsic::sqrt:
    case Intrinsic::fabs:
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
      break;
    default:
      return false;
    }
    New = callFloatIntrinsic(*II, ID);
    ++C.Intr;
  } else {
    return false;
  }

  replace(I, New);
  return true;
}

struct DemoteFP : PassInfoMixin<DemoteFP> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    FunctionFilter Filter;
    bool Verbose = isVerbose();

    Counts Total;
    unsigned Funcs = 0;
    for (Function &F : M) {
      if (!Filter.isSelected(F))
        continue;

      Counts C;
      SmallVector<Instruction *, 64> Dead;
      for (Instruction &I : instructions(F))
        if (demoteInstruction(I, C))
          Dead.push_back(&I);
      eraseAll(Dead);

      if (C.total() == 0)
        continue;
      ++Funcs;
      Total.Arith += C.Arith;
      Total.Cmp += C.Cmp;
      Total.Intr += C.Intr;
      if (Verbose)
        errs() << "[DemoteFP]   " << F.getName() << ": arith=" << C.Arith
               << " cmp=" << C.Cmp << " intr=" << C.Intr << "\n";
    }
    errs() << "[DemoteFP] module " << M.getName() << ": " << Funcs
           << " functions, arith=" << Total.Arith << " cmp=" << Total.Cmp
           << " intr=" << Total.Intr << "\n";
    return Total.total() ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// --- demote-fp-math -------------------------------------------------------

// Math functions whose f32 intrinsic the AMDGPU backend (LLVM 19, gfx942)
// lowers to plain instructions, keyed by both the ocml name and the intrinsic
// name. tan, asin, acos, atan, sinh, cosh, tanh and atan2 are absent because
// the backend would need a library call for them. pow, pown and powi are
// absent because the backend lowers them to exp2(y * log2(x)), which returns
// NaN for a negative base and for pow(0, 0).
std::optional<Intrinsic::ID> mathIntrinsic(StringRef Name) {
  static const std::map<std::string, Intrinsic::ID> Table = {
      {"sin", Intrinsic::sin},
      {"cos", Intrinsic::cos},
      {"log", Intrinsic::log},
      {"log2", Intrinsic::log2},
      {"log10", Intrinsic::log10},
      {"exp", Intrinsic::exp},
      {"exp2", Intrinsic::exp2},
      {"exp10", Intrinsic::exp10},
      {"sqrt", Intrinsic::sqrt},
      {"fabs", Intrinsic::fabs},
      {"floor", Intrinsic::floor},
      {"ceil", Intrinsic::ceil},
      {"trunc", Intrinsic::trunc},
      {"rint", Intrinsic::rint},
      {"nearbyint", Intrinsic::nearbyint},
      {"round", Intrinsic::round},
      {"roundeven", Intrinsic::roundeven},
      {"copysign", Intrinsic::copysign},
      {"fma", Intrinsic::fma},
      {"fmuladd", Intrinsic::fmuladd},
      {"fmin", Intrinsic::minnum},
      {"minnum", Intrinsic::minnum},
      {"fmax", Intrinsic::maxnum},
      {"maxnum", Intrinsic::maxnum},
      {"ldexp", Intrinsic::ldexp},
  };
  auto It = Table.find(Name.str());
  if (It == Table.end())
    return std::nullopt;
  return It->second;
}

// The math function name of a call: "log" for __ocml_log_f64 (or a
// linker-renamed __ocml_log_f64.46), "ldexp" for llvm.ldexp.f64.i32. Returns
// an empty name for calls that are not math functions returning double.
StringRef mathFunctionName(const CallBase &Call) {
  const Function *Callee = Call.getCalledFunction();
  if (!Callee || !Call.getType()->isDoubleTy())
    return {};
  StringRef Name = Callee->getName();
  if (Callee->isIntrinsic()) {
    Name.consume_front("llvm.");
    return Name.split('.').first;
  }
  Name = Name.split('.').first;
  if (!Name.consume_front("__ocml_") || !Name.consume_back("_f64"))
    return {};
  return Name;
}

bool hasDemotableArguments(const CallBase &Call) {
  return std::all_of(Call.arg_begin(), Call.arg_end(), [](const Use &A) {
    return A->getType()->isDoubleTy() || A->getType()->isIntegerTy(32);
  });
}

struct DemoteFPMath : PassInfoMixin<DemoteFPMath> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    FunctionFilter Filter;
    std::vector<std::string> MathSkip = splitEnv("QS_DEMOTE_MATH_SKIP");
    bool Verbose = isVerbose();

    std::map<std::string, unsigned> Demoted, Kept;
    for (Function &F : M) {
      if (!Filter.isSelected(F))
        continue;
      SmallVector<Instruction *, 16> Dead;
      for (Instruction &I : instructions(F)) {
        auto *Call = dyn_cast<CallBase>(&I);
        StringRef Fn = Call ? mathFunctionName(*Call) : StringRef();
        if (Fn.empty())
          continue;
        std::optional<Intrinsic::ID> ID = mathIntrinsic(Fn);
        bool Skipped = std::find(MathSkip.begin(), MathSkip.end(), Fn.str()) !=
                       MathSkip.end();
        if (!ID || Skipped || !hasDemotableArguments(*Call)) {
          ++Kept[Fn.str()];
          continue;
        }
        if (Verbose)
          errs() << "[DemoteFPMath]   " << F.getName() << ": "
                 << Call->getCalledFunction()->getName() << "\n";
        replace(I, callFloatIntrinsic(*Call, *ID));
        Dead.push_back(&I);
        ++Demoted[Fn.str()];
      }
      eraseAll(Dead);
    }

    errs() << "[DemoteFPMath] module " << M.getName() << ": demoted";
    for (const auto &[Fn, N] : Demoted)
      errs() << " " << Fn << "=" << N;
    errs() << "; kept in double";
    for (const auto &[Fn, N] : Kept)
      errs() << " " << Fn << "=" << N;
    errs() << "\n";
    return Demoted.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
  }
};

// --- demote-fp-atomics ----------------------------------------------------

std::optional<Instruction::BinaryOps> floatOpcode(const AtomicRMWInst &RMW) {
  switch (RMW.getOperation()) {
  case AtomicRMWInst::FAdd:
    return Instruction::FAdd;
  case AtomicRMWInst::FSub:
    return Instruction::FSub;
  default:
    return std::nullopt;
  }
}

bool isDoubleFPAtomic(const AtomicRMWInst &RMW) {
  return RMW.isFloatingPointOperation() && RMW.getType()->isDoubleTy();
}

// Replaces `atomicrmw fadd ptr %p, double %v` with
//
//   entry: %init = load atomic i64 %p monotonic
//   loop:  %loaded = phi i64 [%init, entry], [%current, loop]
//          %new = fpext(fptrunc(bitcast %loaded) + fptrunc(%v))
//          {%current, %ok} = cmpxchg %p, %loaded, bitcast %new
//          br %ok, exit, loop
//   exit:  result is bitcast %loaded, the value the update replaced
//
// The comparison is on the i64 bits, so a NaN in memory cannot loop forever.
void expandToFloatCmpXchg(AtomicRMWInst &RMW, Instruction::BinaryOps Op) {
  BasicBlock *Entry = RMW.getParent();
  Function *F = Entry->getParent();
  BasicBlock *Exit =
      Entry->splitBasicBlock(RMW.getIterator(), "demote.atomic.exit");
  BasicBlock *Loop =
      BasicBlock::Create(F->getContext(), "demote.atomic.loop", F, Exit);
  Entry->getTerminator()->eraseFromParent();

  Builder B(Entry);
  B.SetCurrentDebugLocation(RMW.getDebugLoc());
  Type *I64 = B.getInt64Ty();
  Value *Ptr = RMW.getPointerOperand();
  LoadInst *Init = B.CreateAlignedLoad(I64, Ptr, RMW.getAlign(),
                                       RMW.isVolatile(), "demote.atomic.init");
  Init->setAtomic(AtomicOrdering::Monotonic, RMW.getSyncScopeID());
  B.CreateBr(Loop);

  B.SetInsertPoint(Loop);
  PHINode *Loaded = B.CreatePHI(I64, 2, "demote.atomic.loaded");
  Value *Old = B.CreateBitCast(Loaded, B.getDoubleTy());
  Value *Sum =
      B.CreateBinOp(Op, toFloat(B, Old), toFloat(B, RMW.getValOperand()));
  Value *New = B.CreateBitCast(B.CreateFPExt(Sum, B.getDoubleTy()), I64);
  AtomicCmpXchgInst *CmpXchg = B.CreateAtomicCmpXchg(
      Ptr, Loaded, New, RMW.getAlign(), RMW.getOrdering(),
      AtomicCmpXchgInst::getStrongestFailureOrdering(RMW.getOrdering()),
      RMW.getSyncScopeID());
  CmpXchg->setVolatile(RMW.isVolatile());
  Value *Current = B.CreateExtractValue(CmpXchg, 0);
  Value *Succeeded = B.CreateExtractValue(CmpXchg, 1);
  B.CreateCondBr(Succeeded, Exit, Loop);
  Loaded->addIncoming(Init, Entry);
  Loaded->addIncoming(Current, Loop);

  B.SetInsertPoint(&RMW);
  replace(RMW, B.CreateBitCast(Loaded, B.getDoubleTy()));
  RMW.eraseFromParent();
}

struct DemoteFPAtomics : PassInfoMixin<DemoteFPAtomics> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    FunctionFilter Filter;
    bool Verbose = isVerbose();

    std::map<std::string, unsigned> Demoted, Kept;
    for (Function &F : M) {
      if (!Filter.isSelected(F))
        continue;
      SmallVector<AtomicRMWInst *, 4> Atomics;
      for (Instruction &I : instructions(F))
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          if (isDoubleFPAtomic(*RMW))
            Atomics.push_back(RMW);

      for (AtomicRMWInst *RMW : Atomics) {
        std::string OpName =
            AtomicRMWInst::getOperationName(RMW->getOperation()).str();
        std::optional<Instruction::BinaryOps> Op = floatOpcode(*RMW);
        if (!Op) {
          ++Kept[OpName];
          continue;
        }
        if (Verbose)
          errs() << "[DemoteFPAtomics]   " << F.getName() << ": atomicrmw "
                 << OpName << "\n";
        expandToFloatCmpXchg(*RMW, *Op);
        ++Demoted[OpName];
      }
    }

    errs() << "[DemoteFPAtomics] module " << M.getName() << ": demoted";
    for (const auto &[Op, N] : Demoted)
      errs() << " " << Op << "=" << N;
    errs() << "; kept in double";
    for (const auto &[Op, N] : Kept)
      errs() << " " << Op << "=" << N;
    errs() << "\n";
    return Demoted.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DemoteFP", "0.3", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "demote-fp") {
                    MPM.addPass(DemoteFP());
                    return true;
                  }
                  if (Name == "demote-fp-math") {
                    MPM.addPass(DemoteFPMath());
                    return true;
                  }
                  if (Name == "demote-fp-atomics") {
                    MPM.addPass(DemoteFPAtomics());
                    return true;
                  }
                  return false;
                });
          }};
}
