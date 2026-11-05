// Proteus JIT pass plugin (Prepend): keeps double storage, does double
// arithmetic in float. Each fadd/fsub/fmul/fdiv/frem/fneg/fcmp and each
// llvm.{sqrt,fabs,fma,fmuladd}.f64 call becomes fptrunc operands -> float op
// -> fpext result; Proteus's O3 then removes the back-to-back ext/trunc pairs.
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
//   QS_DEMOTE_VERBOSE=1  per-function counts on stderr.
#include <cstdlib>
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

// The math library keeps its own double algorithms, and the RNG must produce
// the same stream in both runs.
bool isAlwaysExcluded(StringRef Name) {
  return Name.starts_with("__ocml") || Name.starts_with("__ockl") ||
         Name.contains("rngSample");
}

struct Counts {
  unsigned Arith = 0, Cmp = 0, Intr = 0;
  unsigned total() const { return Arith + Cmp + Intr; }
};

Value *toFloat(Builder &B, Value *V) {
  return B.CreateFPTrunc(V, B.getFloatTy());
}

bool demoteInstruction(Instruction &I, Counts &C) {
  Builder B(&I);
  if (auto *FPOp = dyn_cast<FPMathOperator>(&I))
    B.setFastMathFlags(FPOp->getFastMathFlags());

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
    SmallVector<Value *, 3> Args;
    for (Value *A : II->args())
      Args.push_back(toFloat(B, A));
    Function *F32 =
        Intrinsic::getDeclaration(I.getModule(), ID, {B.getFloatTy()});
    Value *R = B.CreateCall(F32, Args);
    New = B.CreateFPExt(R, B.getDoubleTy());
    ++C.Intr;
  } else {
    return false;
  }

  New->takeName(&I);
  I.replaceAllUsesWith(New);
  return true;
}

struct DemoteFP : PassInfoMixin<DemoteFP> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    std::vector<std::string> Only = splitEnv("QS_DEMOTE_ONLY");
    std::vector<std::string> Skip = splitEnv("QS_DEMOTE_SKIP");
    const char *V = std::getenv("QS_DEMOTE_VERBOSE");
    bool Verbose = V && V[0] == '1';

    Counts Total;
    unsigned Funcs = 0;
    for (Function &F : M) {
      if (F.isDeclaration() || isAlwaysExcluded(F.getName()) ||
          matchesAny(F.getName(), Skip))
        continue;
      if (!Only.empty() && !matchesAny(F.getName(), Only))
        continue;

      Counts C;
      SmallVector<Instruction *, 64> Dead;
      for (Instruction &I : instructions(F))
        if (demoteInstruction(I, C))
          Dead.push_back(&I);
      for (Instruction *I : Dead)
        I->eraseFromParent();

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
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DemoteFP", "0.1", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "demote-fp")
                    return false;
                  MPM.addPass(DemoteFP());
                  return true;
                });
          }};
}
