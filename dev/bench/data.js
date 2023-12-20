window.BENCHMARK_DATA = {
  "lastUpdate": 1703110659999,
  "repoUrl": "https://github.com/luigifcruz/blade",
  "entries": {
    "Memory Benchmark": [
      {
        "commit": {
          "author": {
            "name": "luigifcruz",
            "username": "luigifcruz"
          },
          "committer": {
            "name": "luigifcruz",
            "username": "luigifcruz"
          },
          "id": "10e2ce1665ad94012cd5630eaaf58e07f6b73aa6",
          "message": "[V1.0] Development Work",
          "timestamp": "2023-12-10T14:45:40Z",
          "url": "https://github.com/luigifcruz/blade/pull/60/commits/10e2ce1665ad94012cd5630eaaf58e07f6b73aa6"
        },
        "date": 1703110658424,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5956.216397277489,
            "unit": "ns/iter",
            "extra": "iterations: 103811\ncpu: 12652.989827667589 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8598.448727835788,
            "unit": "ns/iter",
            "extra": "iterations: 82123\ncpu: 15467.867783690315 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37836.25165117437,
            "unit": "ns/iter",
            "extra": "iterations: 18379\ncpu: 44816.39098971652 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71124.97712626932,
            "unit": "ns/iter",
            "extra": "iterations: 8804\ncpu: 78052.176056338 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71513.97669347715,
            "unit": "ns/iter",
            "extra": "iterations: 8817\ncpu: 78495.68107065899 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138205.84809514205,
            "unit": "ns/iter",
            "extra": "iterations: 4616\ncpu: 145098.5359618719 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135469.70971228657,
            "unit": "ns/iter",
            "extra": "iterations: 4714\ncpu: 142384.31798896912 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270617.173442056,
            "unit": "ns/iter",
            "extra": "iterations: 2506\ncpu: 277318.99561053497 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71627.38695454356,
            "unit": "ns/iter",
            "extra": "iterations: 8724\ncpu: 78572.71824850974 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138173.18457110942,
            "unit": "ns/iter",
            "extra": "iterations: 4619\ncpu: 145041.57588222565 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37875.110778911585,
            "unit": "ns/iter",
            "extra": "iterations: 18122\ncpu: 44827.86546738767 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71097.7369519561,
            "unit": "ns/iter",
            "extra": "iterations: 8817\ncpu: 78033.28467732774 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71538.95140024016,
            "unit": "ns/iter",
            "extra": "iterations: 8781\ncpu: 78471.16740690122 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138157.6985757502,
            "unit": "ns/iter",
            "extra": "iterations: 4587\ncpu: 145078.2175713973 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135450.68844969693,
            "unit": "ns/iter",
            "extra": "iterations: 4726\ncpu: 142381.18112568752 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270733.5928090794,
            "unit": "ns/iter",
            "extra": "iterations: 2501\ncpu: 277417.2255097959 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71590.77528274863,
            "unit": "ns/iter",
            "extra": "iterations: 8697\ncpu: 78549.19834425657 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138520.33473953846,
            "unit": "ns/iter",
            "extra": "iterations: 4573\ncpu: 145387.81784386624 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37841.44771775928,
            "unit": "ns/iter",
            "extra": "iterations: 18107\ncpu: 44808.98089136799 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71081.63394279683,
            "unit": "ns/iter",
            "extra": "iterations: 8810\ncpu: 78031.48933030662 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71492.68536860064,
            "unit": "ns/iter",
            "extra": "iterations: 8772\ncpu: 78469.72412220688 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138097.06248900853,
            "unit": "ns/iter",
            "extra": "iterations: 4581\ncpu: 144928.10980135374 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135253.65431984936,
            "unit": "ns/iter",
            "extra": "iterations: 5040\ncpu: 142226.35476190466 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270727.2794334292,
            "unit": "ns/iter",
            "extra": "iterations: 2503\ncpu: 277330.41150619247 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71653.27972777304,
            "unit": "ns/iter",
            "extra": "iterations: 8718\ncpu: 78587.93748566178 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138278.40029972553,
            "unit": "ns/iter",
            "extra": "iterations: 4588\ncpu: 145029.77571926732 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37815.923963757974,
            "unit": "ns/iter",
            "extra": "iterations: 18343\ncpu: 44736.63768194937 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71058.38205492099,
            "unit": "ns/iter",
            "extra": "iterations: 8790\ncpu: 77978.99852104695 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71598.18679852453,
            "unit": "ns/iter",
            "extra": "iterations: 8773\ncpu: 78484.97081956016 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138170.35193375402,
            "unit": "ns/iter",
            "extra": "iterations: 4597\ncpu: 145121.60343702385 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147484.95746113517,
            "unit": "ns/iter",
            "extra": "iterations: 4382\ncpu: 154274.15426745772 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 292939.2123953838,
            "unit": "ns/iter",
            "extra": "iterations: 2314\ncpu: 299526.8686257545 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71750.96822597864,
            "unit": "ns/iter",
            "extra": "iterations: 8695\ncpu: 78618.56204715285 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138356.99352977774,
            "unit": "ns/iter",
            "extra": "iterations: 4592\ncpu: 145227.46341463478 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}