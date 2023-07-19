#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// NOOP

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::NOOP)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// ADD

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// SUB

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// MULT

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// DIV

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
