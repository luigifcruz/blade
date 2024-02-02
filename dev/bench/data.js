window.BENCHMARK_DATA = {
  "lastUpdate": 1706917978255,
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
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "066482a56dd5a89ebc6c8c75ae98a788b5dee039",
          "message": "Merge pull request #60 from luigifcruz/dev\n\n[V1.0] Development Work",
          "timestamp": "2023-12-20T19:37:32-03:00",
          "tree_id": "7facbc7f9032132eb8833d0aa465b7fa4e9c4971",
          "url": "https://github.com/luigifcruz/blade/commit/066482a56dd5a89ebc6c8c75ae98a788b5dee039"
        },
        "date": 1703112399625,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 6016.413413353688,
            "unit": "ns/iter",
            "extra": "iterations: 103442\ncpu: 12740.598006612403 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8674.478854472613,
            "unit": "ns/iter",
            "extra": "iterations: 81718\ncpu: 15601.020007831812 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37862.29002851533,
            "unit": "ns/iter",
            "extra": "iterations: 18369\ncpu: 44869.77456584463 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71135.50434031627,
            "unit": "ns/iter",
            "extra": "iterations: 8693\ncpu: 78125.72713677667 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71532.20991543523,
            "unit": "ns/iter",
            "extra": "iterations: 8765\ncpu: 78574.27027952082 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138538.47311765933,
            "unit": "ns/iter",
            "extra": "iterations: 4588\ncpu: 145540.36442894506 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135259.8252232883,
            "unit": "ns/iter",
            "extra": "iterations: 5047\ncpu: 142275.25678620973 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 271049.01788808964,
            "unit": "ns/iter",
            "extra": "iterations: 2503\ncpu: 277953.14103076287 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71648.45964343855,
            "unit": "ns/iter",
            "extra": "iterations: 8730\ncpu: 78678.65395188986 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138378.13971708293,
            "unit": "ns/iter",
            "extra": "iterations: 4578\ncpu: 145388.60659676685 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37910.47093899898,
            "unit": "ns/iter",
            "extra": "iterations: 18115\ncpu: 44879.32354402434 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71143.09582013133,
            "unit": "ns/iter",
            "extra": "iterations: 8800\ncpu: 78160.79204545452 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71583.27945351558,
            "unit": "ns/iter",
            "extra": "iterations: 8776\ncpu: 78570.83397903363 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138172.3774595431,
            "unit": "ns/iter",
            "extra": "iterations: 4588\ncpu: 145178.84677419357 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135517.3993901482,
            "unit": "ns/iter",
            "extra": "iterations: 4718\ncpu: 142531.89359898283 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270606.56733528076,
            "unit": "ns/iter",
            "extra": "iterations: 2503\ncpu: 277548.7419097085 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71627.01136286672,
            "unit": "ns/iter",
            "extra": "iterations: 8731\ncpu: 78624.3704043067 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138481.12699438448,
            "unit": "ns/iter",
            "extra": "iterations: 4575\ncpu: 145512.9357377051 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37849.96597906471,
            "unit": "ns/iter",
            "extra": "iterations: 18107\ncpu: 44881.48820898003 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71164.4150151128,
            "unit": "ns/iter",
            "extra": "iterations: 8816\ncpu: 78156.28516333942 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71567.44191709552,
            "unit": "ns/iter",
            "extra": "iterations: 8808\ncpu: 78579.92177565853 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138223.69349658414,
            "unit": "ns/iter",
            "extra": "iterations: 4583\ncpu: 145228.0931704122 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135540.0081495373,
            "unit": "ns/iter",
            "extra": "iterations: 4711\ncpu: 142582.98514115898 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270795.745934417,
            "unit": "ns/iter",
            "extra": "iterations: 2502\ncpu: 277713.2505995205 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71651.42252804496,
            "unit": "ns/iter",
            "extra": "iterations: 8727\ncpu: 78694.43657614289 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138401.28312493148,
            "unit": "ns/iter",
            "extra": "iterations: 4579\ncpu: 145338.68552085565 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37844.456720708964,
            "unit": "ns/iter",
            "extra": "iterations: 18361\ncpu: 44821.602309242495 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71074.09541892304,
            "unit": "ns/iter",
            "extra": "iterations: 8817\ncpu: 78062.53498922514 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71594.71933895934,
            "unit": "ns/iter",
            "extra": "iterations: 8733\ncpu: 78551.6213214243 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138307.57145339224,
            "unit": "ns/iter",
            "extra": "iterations: 4584\ncpu: 145321.96596858697 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147706.02708973092,
            "unit": "ns/iter",
            "extra": "iterations: 4360\ncpu: 154701.09747706394 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 294086.9713553589,
            "unit": "ns/iter",
            "extra": "iterations: 2305\ncpu: 301123.3375271139 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71664.28803305003,
            "unit": "ns/iter",
            "extra": "iterations: 9378\ncpu: 78665.05854126651 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138401.78179628876,
            "unit": "ns/iter",
            "extra": "iterations: 4572\ncpu: 145416.55730533705 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1946561f4d860df2ae1c23e94a80c1d1d3c45a91",
          "message": "Add minimal CUDA version.",
          "timestamp": "2024-01-18T21:59:18-03:00",
          "tree_id": "60dea43b99c4d915f05924c78f1e961afe5376ca",
          "url": "https://github.com/luigifcruz/blade/commit/1946561f4d860df2ae1c23e94a80c1d1d3c45a91"
        },
        "date": 1705626535711,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5998.075077032515,
            "unit": "ns/iter",
            "extra": "iterations: 104501\ncpu: 12717.678385852769 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8667.2664009386,
            "unit": "ns/iter",
            "extra": "iterations: 82466\ncpu: 15776.64893410618 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37822.737464287384,
            "unit": "ns/iter",
            "extra": "iterations: 18384\ncpu: 44857.630113141895 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71080.72057148907,
            "unit": "ns/iter",
            "extra": "iterations: 8819\ncpu: 78069.40265336209 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71503.66355440437,
            "unit": "ns/iter",
            "extra": "iterations: 8810\ncpu: 78542.22814982965 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138396.79983947103,
            "unit": "ns/iter",
            "extra": "iterations: 4604\ncpu: 145331.89400521273 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135423.6376342414,
            "unit": "ns/iter",
            "extra": "iterations: 4715\ncpu: 142358.36564156954 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270767.8901314731,
            "unit": "ns/iter",
            "extra": "iterations: 2500\ncpu: 277644.5292000005 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71539.74155821872,
            "unit": "ns/iter",
            "extra": "iterations: 8744\ncpu: 78585.84755260752 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138340.60861538173,
            "unit": "ns/iter",
            "extra": "iterations: 4594\ncpu: 145305.22899434046 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37845.55522691614,
            "unit": "ns/iter",
            "extra": "iterations: 18123\ncpu: 44858.300778016936 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71069.76234689301,
            "unit": "ns/iter",
            "extra": "iterations: 8833\ncpu: 78119.79859617357 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71581.61216625514,
            "unit": "ns/iter",
            "extra": "iterations: 8719\ncpu: 78580.94896203694 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138196.63112629607,
            "unit": "ns/iter",
            "extra": "iterations: 4589\ncpu: 145168.66441490516 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135425.90025871203,
            "unit": "ns/iter",
            "extra": "iterations: 4725\ncpu: 142353.57735449757 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270756.1857572696,
            "unit": "ns/iter",
            "extra": "iterations: 2503\ncpu: 277594.6679984023 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71571.75493985145,
            "unit": "ns/iter",
            "extra": "iterations: 8730\ncpu: 78583.49347079055 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138415.84389497348,
            "unit": "ns/iter",
            "extra": "iterations: 4586\ncpu: 145381.0691234192 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37850.95227872938,
            "unit": "ns/iter",
            "extra": "iterations: 18133\ncpu: 44881.97915402847 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71097.73903661377,
            "unit": "ns/iter",
            "extra": "iterations: 8779\ncpu: 78017.37885863993 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71526.96745740689,
            "unit": "ns/iter",
            "extra": "iterations: 8797\ncpu: 78541.6739797655 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138115.0494841829,
            "unit": "ns/iter",
            "extra": "iterations: 4613\ncpu: 145104.85605896375 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135466.25380431663,
            "unit": "ns/iter",
            "extra": "iterations: 4708\ncpu: 142419.89677145283 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270714.920521257,
            "unit": "ns/iter",
            "extra": "iterations: 2505\ncpu: 277963.4578842321 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71653.29024259294,
            "unit": "ns/iter",
            "extra": "iterations: 8743\ncpu: 78641.46299897067 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138288.05993872051,
            "unit": "ns/iter",
            "extra": "iterations: 4606\ncpu: 145175.6632653059 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37872.096894469694,
            "unit": "ns/iter",
            "extra": "iterations: 18384\ncpu: 44871.81206483912 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71016.89719368426,
            "unit": "ns/iter",
            "extra": "iterations: 8813\ncpu: 78041.94508113027 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71599.35527710487,
            "unit": "ns/iter",
            "extra": "iterations: 8741\ncpu: 78605.0267703923 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138308.30184930775,
            "unit": "ns/iter",
            "extra": "iterations: 4609\ncpu: 145298.64677804292 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147777.35127621357,
            "unit": "ns/iter",
            "extra": "iterations: 4359\ncpu: 154674.12594631768 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 293140.5357748606,
            "unit": "ns/iter",
            "extra": "iterations: 2309\ncpu: 299836.4486790812 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71755.6284039884,
            "unit": "ns/iter",
            "extra": "iterations: 8708\ncpu: 78733.15089572767 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138467.37952127302,
            "unit": "ns/iter",
            "extra": "iterations: 4579\ncpu: 145399.47106355158 ns\nthreads: 1"
          }
        ]
      },
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
          "id": "9e89758e3f332706120f1593390ba6ee5e2b0b81",
          "message": "V1.0.1",
          "timestamp": "2024-01-18T16:19:59Z",
          "url": "https://github.com/luigifcruz/blade/pull/66/commits/9e89758e3f332706120f1593390ba6ee5e2b0b81"
        },
        "date": 1705628204580,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 6003.284983789951,
            "unit": "ns/iter",
            "extra": "iterations: 104367\ncpu: 12744.792022382553 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8651.497802095075,
            "unit": "ns/iter",
            "extra": "iterations: 82440\ncpu: 15760.833709364386 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37799.39368913072,
            "unit": "ns/iter",
            "extra": "iterations: 18371\ncpu: 44834.23286701867 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 70463.52292354978,
            "unit": "ns/iter",
            "extra": "iterations: 8764\ncpu: 77473.59402099495 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71544.49171673502,
            "unit": "ns/iter",
            "extra": "iterations: 8859\ncpu: 78548.50016931933 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138322.11177310313,
            "unit": "ns/iter",
            "extra": "iterations: 4600\ncpu: 145270.89673913034 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135536.721659647,
            "unit": "ns/iter",
            "extra": "iterations: 4726\ncpu: 142466.41197630134 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270941.3623971481,
            "unit": "ns/iter",
            "extra": "iterations: 2507\ncpu: 277561.62824092497 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71544.32885995106,
            "unit": "ns/iter",
            "extra": "iterations: 8746\ncpu: 78578.97987651502 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138319.72849018135,
            "unit": "ns/iter",
            "extra": "iterations: 4585\ncpu: 145330.86608506 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37837.63247322538,
            "unit": "ns/iter",
            "extra": "iterations: 18123\ncpu: 44870.06941455609 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71069.14242083342,
            "unit": "ns/iter",
            "extra": "iterations: 8854\ncpu: 78081.98667269027 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71517.5931005597,
            "unit": "ns/iter",
            "extra": "iterations: 8823\ncpu: 78525.67981412217 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138263.45227358828,
            "unit": "ns/iter",
            "extra": "iterations: 4595\ncpu: 145237.77431991292 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135464.4175997703,
            "unit": "ns/iter",
            "extra": "iterations: 4720\ncpu: 142315.77902542386 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270762.44609832275,
            "unit": "ns/iter",
            "extra": "iterations: 2507\ncpu: 277588.0614280024 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71601.76446375986,
            "unit": "ns/iter",
            "extra": "iterations: 8721\ncpu: 78587.18839582638 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138443.9953239661,
            "unit": "ns/iter",
            "extra": "iterations: 4590\ncpu: 145406.9324618738 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37799.562375364905,
            "unit": "ns/iter",
            "extra": "iterations: 18139\ncpu: 44844.89966370805 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71037.81457842924,
            "unit": "ns/iter",
            "extra": "iterations: 8810\ncpu: 78020.49841089682 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71472.7747534282,
            "unit": "ns/iter",
            "extra": "iterations: 8793\ncpu: 78488.92005003996 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138250.40481165104,
            "unit": "ns/iter",
            "extra": "iterations: 4609\ncpu: 145167.92037318266 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135566.51125298493,
            "unit": "ns/iter",
            "extra": "iterations: 4705\ncpu: 142475.46695005355 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270584.3346479923,
            "unit": "ns/iter",
            "extra": "iterations: 2503\ncpu: 277201.72353176086 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71667.43760229884,
            "unit": "ns/iter",
            "extra": "iterations: 8754\ncpu: 78675.9994288325 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138310.95236510038,
            "unit": "ns/iter",
            "extra": "iterations: 4601\ncpu: 145254.04955444537 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37830.80914699403,
            "unit": "ns/iter",
            "extra": "iterations: 18355\ncpu: 44845.95592481608 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 70983.94500228467,
            "unit": "ns/iter",
            "extra": "iterations: 8826\ncpu: 78003.02741898927 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71608.32945826554,
            "unit": "ns/iter",
            "extra": "iterations: 8780\ncpu: 78593.29840546667 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138353.07189146712,
            "unit": "ns/iter",
            "extra": "iterations: 4601\ncpu: 145332.90371658295 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147836.9790060892,
            "unit": "ns/iter",
            "extra": "iterations: 4352\ncpu: 154692.05514705807 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 293601.6661466302,
            "unit": "ns/iter",
            "extra": "iterations: 2306\ncpu: 300417.0828274047 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71693.1242730206,
            "unit": "ns/iter",
            "extra": "iterations: 8742\ncpu: 78714.45515900315 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138311.83452363507,
            "unit": "ns/iter",
            "extra": "iterations: 4577\ncpu: 145292.43871531653 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df5f9185646fef36256e141c0a7818c6b931cba3",
          "message": "Merge pull request #66 from luigifcruz/v1.0.1\n\nV1.0.1\r\n\r\n- Remove unnecessary <span> include. \r\n- Change workdir place in Dockerfile.",
          "timestamp": "2024-01-18T22:22:46-03:00",
          "tree_id": "169d76a8efa9b55fe86bb795751030eeb8bcd7a6",
          "url": "https://github.com/luigifcruz/blade/commit/df5f9185646fef36256e141c0a7818c6b931cba3"
        },
        "date": 1705629829954,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5940.688750752018,
            "unit": "ns/iter",
            "extra": "iterations: 104368\ncpu: 12637.456308447034 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8654.937943754452,
            "unit": "ns/iter",
            "extra": "iterations: 82460\ncpu: 15737.179820519037 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37800.82261620891,
            "unit": "ns/iter",
            "extra": "iterations: 18377\ncpu: 44801.72666920609 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71069.1199334276,
            "unit": "ns/iter",
            "extra": "iterations: 8785\ncpu: 78058.92088787704 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71499.71532795078,
            "unit": "ns/iter",
            "extra": "iterations: 8780\ncpu: 78518.35649202728 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138401.77160134786,
            "unit": "ns/iter",
            "extra": "iterations: 4599\ncpu: 145372.95781691663 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135435.60519104608,
            "unit": "ns/iter",
            "extra": "iterations: 4725\ncpu: 142440.9595767196 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270842.8809769721,
            "unit": "ns/iter",
            "extra": "iterations: 2498\ncpu: 277742.6016813452 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71627.6981738982,
            "unit": "ns/iter",
            "extra": "iterations: 8719\ncpu: 78607.97557059297 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138369.38090057837,
            "unit": "ns/iter",
            "extra": "iterations: 4579\ncpu: 145383.6034068571 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37849.11439980002,
            "unit": "ns/iter",
            "extra": "iterations: 18131\ncpu: 44843.63057746399 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71058.55011049699,
            "unit": "ns/iter",
            "extra": "iterations: 8826\ncpu: 78105.9607976433 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71501.64414493693,
            "unit": "ns/iter",
            "extra": "iterations: 8749\ncpu: 78492.93724997136 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138233.18879934624,
            "unit": "ns/iter",
            "extra": "iterations: 4592\ncpu: 145216.05008710816 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135490.45465253896,
            "unit": "ns/iter",
            "extra": "iterations: 4701\ncpu: 142526.16741118894 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270845.0073571653,
            "unit": "ns/iter",
            "extra": "iterations: 2498\ncpu: 277782.43034427497 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71642.34448102542,
            "unit": "ns/iter",
            "extra": "iterations: 8768\ncpu: 78615.16434762761 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138399.24805665232,
            "unit": "ns/iter",
            "extra": "iterations: 4570\ncpu: 145436.52975929953 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37834.305840464534,
            "unit": "ns/iter",
            "extra": "iterations: 18120\ncpu: 44852.27461368645 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71084.60090075394,
            "unit": "ns/iter",
            "extra": "iterations: 8785\ncpu: 78077.36596471259 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71528.05687884145,
            "unit": "ns/iter",
            "extra": "iterations: 8799\ncpu: 78649.34435731321 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138261.32825028693,
            "unit": "ns/iter",
            "extra": "iterations: 4586\ncpu: 145248.34736153495 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135588.1351884067,
            "unit": "ns/iter",
            "extra": "iterations: 4702\ncpu: 142608.32475542312 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270818.8325512261,
            "unit": "ns/iter",
            "extra": "iterations: 2501\ncpu: 277711.4122351063 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71646.34257252727,
            "unit": "ns/iter",
            "extra": "iterations: 8758\ncpu: 78625.91687599904 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138405.1490583198,
            "unit": "ns/iter",
            "extra": "iterations: 4590\ncpu: 145338.29237472743 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37815.818813705984,
            "unit": "ns/iter",
            "extra": "iterations: 18381\ncpu: 44781.416136227585 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71025.41669217519,
            "unit": "ns/iter",
            "extra": "iterations: 8794\ncpu: 78004.24937457341 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71609.920802202,
            "unit": "ns/iter",
            "extra": "iterations: 8757\ncpu: 78562.80918122614 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138111.853876597,
            "unit": "ns/iter",
            "extra": "iterations: 4916\ncpu: 145104.7912937346 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147767.20367286846,
            "unit": "ns/iter",
            "extra": "iterations: 4350\ncpu: 154792.11103448295 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 294994.45330343023,
            "unit": "ns/iter",
            "extra": "iterations: 2295\ncpu: 301961.1032679745 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71751.99692340006,
            "unit": "ns/iter",
            "extra": "iterations: 8741\ncpu: 78685.13202150824 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138418.9540431623,
            "unit": "ns/iter",
            "extra": "iterations: 4578\ncpu: 145428.61533420617 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "code@radonn.co.za",
            "name": "Ross Donnachie",
            "username": "radonnachie"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b15b4b8a085107e5ab04f24504a4b8f77ac30a16",
          "message": "*^ memory/vector._refs host pointer (#67)\n\n* *^ memory/vector._refs host pointer\r\nfixing seg-faults seen under WSL2 docker runs by @radonnachie\r\n\r\n* @ minor version bump",
          "timestamp": "2024-01-29T16:45:01-03:00",
          "tree_id": "f69c2306cac2f3c40049af17abf65129c4c0bf2d",
          "url": "https://github.com/luigifcruz/blade/commit/b15b4b8a085107e5ab04f24504a4b8f77ac30a16"
        },
        "date": 1706558246339,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5940.655931235936,
            "unit": "ns/iter",
            "extra": "iterations: 105688\ncpu: 12615.15397206873 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8542.393125776845,
            "unit": "ns/iter",
            "extra": "iterations: 83061\ncpu: 15387.14285886276 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37799.016234946226,
            "unit": "ns/iter",
            "extra": "iterations: 18365\ncpu: 44781.00794990474 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71050.70544516375,
            "unit": "ns/iter",
            "extra": "iterations: 8795\ncpu: 78022.15440591236 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71505.65858293245,
            "unit": "ns/iter",
            "extra": "iterations: 8757\ncpu: 78496.16649537516 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138323.8486892173,
            "unit": "ns/iter",
            "extra": "iterations: 4599\ncpu: 145295.77995216358 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135507.2447299675,
            "unit": "ns/iter",
            "extra": "iterations: 4705\ncpu: 142490.54707757704 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270725.2381730673,
            "unit": "ns/iter",
            "extra": "iterations: 2501\ncpu: 277629.8996401438 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71535.23109721782,
            "unit": "ns/iter",
            "extra": "iterations: 8745\ncpu: 78512.98879359633 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138323.33210148942,
            "unit": "ns/iter",
            "extra": "iterations: 4601\ncpu: 145256.80134753295 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37836.329104108045,
            "unit": "ns/iter",
            "extra": "iterations: 18126\ncpu: 44765.04010813189 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71092.27309439956,
            "unit": "ns/iter",
            "extra": "iterations: 8813\ncpu: 78079.58810847612 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71593.24628966775,
            "unit": "ns/iter",
            "extra": "iterations: 8778\ncpu: 78508.44497607659 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138181.5817600901,
            "unit": "ns/iter",
            "extra": "iterations: 4607\ncpu: 145167.66008248285 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135554.10685930893,
            "unit": "ns/iter",
            "extra": "iterations: 4702\ncpu: 142532.975754998 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270736.5168203442,
            "unit": "ns/iter",
            "extra": "iterations: 2502\ncpu: 277654.8573141494 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71568.95578461938,
            "unit": "ns/iter",
            "extra": "iterations: 8718\ncpu: 78610.41970635449 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138400.86435500835,
            "unit": "ns/iter",
            "extra": "iterations: 4577\ncpu: 145403.95040419544 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37787.49417983737,
            "unit": "ns/iter",
            "extra": "iterations: 18105\ncpu: 44775.546313173174 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71043.68470647752,
            "unit": "ns/iter",
            "extra": "iterations: 8809\ncpu: 78085.0753774547 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71482.38794011081,
            "unit": "ns/iter",
            "extra": "iterations: 8836\ncpu: 78474.96842462668 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138173.0424690346,
            "unit": "ns/iter",
            "extra": "iterations: 4581\ncpu: 145162.8511242084 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135553.8741609026,
            "unit": "ns/iter",
            "extra": "iterations: 4706\ncpu: 142590.58074798144 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270541.3194821402,
            "unit": "ns/iter",
            "extra": "iterations: 2502\ncpu: 277442.09792166273 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71637.96834286558,
            "unit": "ns/iter",
            "extra": "iterations: 8755\ncpu: 78654.24328954848 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138323.68759733913,
            "unit": "ns/iter",
            "extra": "iterations: 4583\ncpu: 145277.6938686449 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37876.68162706228,
            "unit": "ns/iter",
            "extra": "iterations: 18398\ncpu: 44866.67360582662 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71063.23426329964,
            "unit": "ns/iter",
            "extra": "iterations: 8793\ncpu: 78055.78414648032 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71586.08092680696,
            "unit": "ns/iter",
            "extra": "iterations: 8756\ncpu: 78564.53517587922 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138384.39838874395,
            "unit": "ns/iter",
            "extra": "iterations: 4589\ncpu: 145357.78208760064 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147814.31498671227,
            "unit": "ns/iter",
            "extra": "iterations: 4339\ncpu: 154847.8711684719 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 293394.4256930278,
            "unit": "ns/iter",
            "extra": "iterations: 2313\ncpu: 300365.37051448284 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71671.55516454109,
            "unit": "ns/iter",
            "extra": "iterations: 8748\ncpu: 78653.89700503011 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138373.41205539936,
            "unit": "ns/iter",
            "extra": "iterations: 4586\ncpu: 145382.5388137804 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca24dbcf45a5ba9e8f4420dfd64ffb6f47c3bd2a",
          "message": "Merge pull request #68 from luigifcruz/v1.0.3\n\nBetter compatibility with older CUDA versions.",
          "timestamp": "2024-02-02T20:32:54-03:00",
          "tree_id": "2c50ecd650050f409cde9589479bcabb763cc4e9",
          "url": "https://github.com/luigifcruz/blade/commit/ca24dbcf45a5ba9e8f4420dfd64ffb6f47c3bd2a"
        },
        "date": 1706917976978,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5938.482899733287,
            "unit": "ns/iter",
            "extra": "iterations: 104521\ncpu: 12646.296619818024 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8514.134809037967,
            "unit": "ns/iter",
            "extra": "iterations: 82684\ncpu: 15335.177761114604 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37808.72547868171,
            "unit": "ns/iter",
            "extra": "iterations: 18401\ncpu: 44825.645943155294 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71104.21676978379,
            "unit": "ns/iter",
            "extra": "iterations: 8757\ncpu: 78099.82379810429 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71563.47186391639,
            "unit": "ns/iter",
            "extra": "iterations: 8785\ncpu: 78582.02982356286 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138406.71796670975,
            "unit": "ns/iter",
            "extra": "iterations: 4609\ncpu: 145369.9316554567 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135484.3063928433,
            "unit": "ns/iter",
            "extra": "iterations: 4725\ncpu: 142489.05502645505 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 269769.854007842,
            "unit": "ns/iter",
            "extra": "iterations: 2505\ncpu: 276610.98323353333 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71591.34583231498,
            "unit": "ns/iter",
            "extra": "iterations: 8714\ncpu: 78603.66972687632 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138330.6035346366,
            "unit": "ns/iter",
            "extra": "iterations: 4604\ncpu: 145313.37358818416 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37846.43736091189,
            "unit": "ns/iter",
            "extra": "iterations: 18125\ncpu: 44845.6290206897 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71130.43539029936,
            "unit": "ns/iter",
            "extra": "iterations: 8831\ncpu: 78104.4729928661 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71498.73573100574,
            "unit": "ns/iter",
            "extra": "iterations: 8824\ncpu: 78509.8074569358 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138085.33415318394,
            "unit": "ns/iter",
            "extra": "iterations: 4593\ncpu: 145074.58523840626 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135526.1322644174,
            "unit": "ns/iter",
            "extra": "iterations: 4715\ncpu: 142533.92216330866 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270404.3118907987,
            "unit": "ns/iter",
            "extra": "iterations: 2504\ncpu: 277403.2515974449 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71565.80258835478,
            "unit": "ns/iter",
            "extra": "iterations: 8732\ncpu: 78553.57638570764 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138361.1386619622,
            "unit": "ns/iter",
            "extra": "iterations: 4605\ncpu: 145372.86471226928 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37804.459473332405,
            "unit": "ns/iter",
            "extra": "iterations: 18129\ncpu: 44815.510342544985 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71115.85513139362,
            "unit": "ns/iter",
            "extra": "iterations: 8787\ncpu: 78070.09434391742 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71537.90500677068,
            "unit": "ns/iter",
            "extra": "iterations: 8842\ncpu: 78557.49400588112 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 137771.85313159475,
            "unit": "ns/iter",
            "extra": "iterations: 4942\ncpu: 144769.28369081378 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135254.84180343692,
            "unit": "ns/iter",
            "extra": "iterations: 5040\ncpu: 142270.35535714254 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270670.9361615383,
            "unit": "ns/iter",
            "extra": "iterations: 2499\ncpu: 277486.6350540227 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71562.4823695201,
            "unit": "ns/iter",
            "extra": "iterations: 8729\ncpu: 78542.26692633759 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138375.57995316826,
            "unit": "ns/iter",
            "extra": "iterations: 4583\ncpu: 145286.65786602636 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37760.35782480756,
            "unit": "ns/iter",
            "extra": "iterations: 18394\ncpu: 44734.57377405662 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71027.21354699846,
            "unit": "ns/iter",
            "extra": "iterations: 8816\ncpu: 78006.50533121618 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71487.86556903548,
            "unit": "ns/iter",
            "extra": "iterations: 9426\ncpu: 78433.554848292 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138302.8573621166,
            "unit": "ns/iter",
            "extra": "iterations: 4604\ncpu: 145256.91854908707 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147667.49645336083,
            "unit": "ns/iter",
            "extra": "iterations: 4345\ncpu: 154633.80529344064 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 293955.81994369533,
            "unit": "ns/iter",
            "extra": "iterations: 2314\ncpu: 300909.3232497831 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71699.93220222903,
            "unit": "ns/iter",
            "extra": "iterations: 8723\ncpu: 78652.18032786855 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138281.34648591373,
            "unit": "ns/iter",
            "extra": "iterations: 4583\ncpu: 145276.50469125024 ns\nthreads: 1"
          }
        ]
      }
    ],
    "Modules Benchmark": [
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
        "date": 1703111679683,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.002134326127882502,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009401766906738282 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.0021364257393879926,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.00942130908203125 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.0021262265228541577,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009359097167968749 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.17900414700822154,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1870554014892578 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.3302018099068391,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3384772383422852 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.285342585125363,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.294380816894532 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.12914123495066931,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.13730448791503952 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.9343403894988,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9425972155151378 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.701078573186578,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.7094807311401374 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.574038915459823,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.58254393109131 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.30932334229305525,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3175857186889647 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.3504284727948743,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.3588623284301753 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7484702236233431,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7566853065795915 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 6.004174569909537,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.0125989310913095 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.7951225987419264,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8033365812377957 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.350532077220805,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.35892049932861 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.36800358064326133,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3761415314941438 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8729711085162535,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.881439629638674 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.2775426938637082,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2856118197021515 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.136159899819745,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1430875451049767 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.1514018082220403,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1594841915893569 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1166544036385062,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.123956271728517 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.6457846047922544,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6526524266357442 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2920261892498388,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3000054089965787 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.3270527082861463,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3348775679321292 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2575380094214097,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.26546215075683166 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.577874691269244,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.5843909341430673 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.2922222444800013,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2997901137084962 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.5679368183280076,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.5762102799682576 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8415509807662147,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8498331372070317 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.2857845044393912,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2941757191772472 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3351751520715993,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3434463282470734 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4104929396268062,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4188894135131815 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.8281804771760335,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.836601595947264 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.7073871201963584,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.7158117067871137 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.336089628445336,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.344870121093751 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4099253356718577,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4182498086547854 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.0412420375777174,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.049676748291013 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "066482a56dd5a89ebc6c8c75ae98a788b5dee039",
          "message": "Merge pull request #60 from luigifcruz/dev\n\n[V1.0] Development Work",
          "timestamp": "2023-12-20T19:37:32-03:00",
          "tree_id": "7facbc7f9032132eb8833d0aa465b7fa4e9c4971",
          "url": "https://github.com/luigifcruz/blade/commit/066482a56dd5a89ebc6c8c75ae98a788b5dee039"
        },
        "date": 1703113421191,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.0021991698805831628,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009571587524414062 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.00217415425571299,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009501127197265627 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.002168226522936134,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009432270996093754 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.1789622422068149,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.18700734899902344 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.3308060327403837,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.339080637573242 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.2867154366254,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.295170730041505 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.12919128139632363,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.13735573474121104 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.9344713706518348,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9427263740844722 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.700129494262683,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.708564698059082 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.573985607218447,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.582514841247557 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.30809611168258755,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3162748090209978 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.346483502279284,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.3548618690795893 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7484152640522268,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7565446115112284 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 5.990264892147934,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.998658220092774 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.7952179892320999,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8033626756591798 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.350450481704684,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.358845424560548 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.3680722717884777,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.37616712084961174 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8728314709240976,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.8811748659667984 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.27806929499618604,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2856196989746082 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.136346740783779,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1432549176635742 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.1515873968296333,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.15970262036132943 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1170590816291792,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.123762100524908 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.6458319151825265,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6532468834838934 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.29201493571484605,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.30015582611083585 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.32704365354696563,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.33516602789306765 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2574062672442423,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.26557675976562195 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.5778839076195936,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.58638478704834 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.2922011571419034,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3005401023559529 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.567973364354657,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.5762079145507776 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8415400150099117,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8498098152465855 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.2857547468954067,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2940857887573238 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3353052348463734,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3435994215698286 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.410708431819785,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4190705290527337 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.8283689137774672,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.8367695114746077 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.7073832621198335,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.7158032401733423 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3361370623883317,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.344827714660643 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.415256637315565,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.423576533996579 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.0986550924817493,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1070123330688486 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1946561f4d860df2ae1c23e94a80c1d1d3c45a91",
          "message": "Add minimal CUDA version.",
          "timestamp": "2024-01-18T21:59:18-03:00",
          "tree_id": "60dea43b99c4d915f05924c78f1e961afe5376ca",
          "url": "https://github.com/luigifcruz/blade/commit/1946561f4d860df2ae1c23e94a80c1d1d3c45a91"
        },
        "date": 1705627557178,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.0021329081606805134,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009426470886230469 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.0021433105060406077,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009417599975585937 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.0021377577736070297,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009366099731445312 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.17878134648086075,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.18692953765869139 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.330242144838678,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3384792958984375 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.287065638299282,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.295593324523925 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.12952144528632914,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.13768792791747989 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.935977015231515,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9442494174804681 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.705926088855449,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.7144577705688473 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.573853883651282,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.582453645812988 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.30935542320520426,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.31758253900146527 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.3477166676286743,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.3561563915405275 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7489014782606773,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7570994836425781 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 6.0048990124528245,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.013326869018554 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.7957545603360927,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8039041563110347 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.354365208608215,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.362925565246581 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.3682656136163587,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3763413775024413 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8740931521440416,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.882499561706539 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.2781123123209994,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.28583170989990303 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.13630670737075,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1449575563964802 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.15143396156069855,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1595395664062485 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1172901218117204,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.1258067017822295 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.6458521938803585,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6541235072021478 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2918344257700767,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2999470270996091 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.32703061851613313,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3352113879394533 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.25756491370287904,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2657888348999021 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.577800513307693,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.586338812866212 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.29209158124155,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3004678051147454 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.5682096848289575,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.5764527570190427 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8416205589334425,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.84993720965576 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.285865658793739,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2942827788696292 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3354677539325621,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3438447775878937 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4104779551686875,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4188322340087864 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.8282382111536322,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.8366616242065439 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.7073792980019107,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.7158197055053708 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3362236725384946,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3448875379638707 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4096636949219032,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4180422027587902 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.051541440124538,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.060596436157226 ms\nthreads: 1"
          }
        ]
      },
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
          "id": "9e89758e3f332706120f1593390ba6ee5e2b0b81",
          "message": "V1.0.1",
          "timestamp": "2024-01-18T16:19:59Z",
          "url": "https://github.com/luigifcruz/blade/pull/66/commits/9e89758e3f332706120f1593390ba6ee5e2b0b81"
        },
        "date": 1705629226934,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.0021299257386905857,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009403759033203127 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.0021480995663764046,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009504697204589843 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.002126849569510092,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009407288146972656 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.17907469138744858,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1871401943359375 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.3304600653825105,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.338736373046875 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.287593029891013,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.296002837585449 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.1292532484029607,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.13740597467041052 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.9353395683646681,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9435194993286126 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.704000343347502,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.7125274277954103 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.574071634105394,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.582546407043458 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.3096797498169934,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.31787466625976535 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.357432958802974,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.365892679260257 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7490688360363151,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7573001718139648 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 6.0083422005732245,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.0168955236816375 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.7956532939878969,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8038391529541014 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.352069289590645,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.360522276123048 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.3682105251598955,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.37635300451660525 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8734825082068483,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.8819145363159215 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.2781502484996423,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.28586149511718867 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.136228047191935,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1448562426147415 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.15153590144301177,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.15985217993164103 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1167995286029964,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.125321770507809 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.6459089097567983,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6542345898437468 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2918954244730543,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.30007762658691706 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.3271503541242282,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.33544916778564804 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.25748046219131737,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2656175435180666 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.577881379494329,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.5864045684204084 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.2922403583033315,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.300659464904784 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.5681198833826784,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.5763746423339822 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8415834575004055,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8498901969604469 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.2859094018509154,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2943037228393544 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3353896631116413,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3437572736816408 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4110306413286366,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4194286642456033 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.828505950214776,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.836955618530277 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.707596093031839,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.716027201232908 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3364508056668,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.345015280944828 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.417317816446939,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4257392166748053 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.0991833822421313,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.1076256041259804 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df5f9185646fef36256e141c0a7818c6b931cba3",
          "message": "Merge pull request #66 from luigifcruz/v1.0.1\n\nV1.0.1\r\n\r\n- Remove unnecessary <span> include. \r\n- Change workdir place in Dockerfile.",
          "timestamp": "2024-01-18T22:22:46-03:00",
          "tree_id": "169d76a8efa9b55fe86bb795751030eeb8bcd7a6",
          "url": "https://github.com/luigifcruz/blade/commit/df5f9185646fef36256e141c0a7818c6b931cba3"
        },
        "date": 1705630851425,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.002118992144556353,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009346164367675783 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.002127060506507128,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009383046142578125 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.0021204491797294223,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009346262634277345 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.17905459313016792,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.18674365869140624 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.3303986840114135,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3376132013549804 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.288027767960557,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.287358732055664 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.12930167357749767,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1373609372558593 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.9366100462244731,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9435700615844731 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.7078646279837812,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.707893909423828 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.574078581422782,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.577863884582518 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.30898525057310167,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3171214963989255 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.348276266729954,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.3567653404541025 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7490037864030796,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7572761012573231 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 6.004594551797027,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.013208876220703 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.795813833324388,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8040896397705083 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.35329888217484,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.3618819873046855 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.36829290624851296,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3763906504516551 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8739484152566774,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.8824308455200205 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.27826299429989376,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2861642161865255 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.13620492790767,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.144908050964352 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.15162119610767633,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.15998609484863674 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1170461588321245,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.1255655427246103 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.6458329284306785,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6541707276611353 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.29203906003871793,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3003173615722665 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.32707178222679545,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3353516657714842 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.25757361436795634,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.26580881237792814 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.5778109743299638,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.5863605451660123 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.2921571051549563,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3005752605590866 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.5681686752971871,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.5764239795532228 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8417218901577428,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8500847319946256 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.2858950105538725,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2943424959106429 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3353823928312636,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3437894107665993 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4105712176970542,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4189870994262694 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.8286289285143198,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.837097895202637 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.707454036510626,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.715992927978514 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3363246821498365,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3449225681762682 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4099877918312131,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4183934177246087 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.050798020917455,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.059287238647463 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "code@radonn.co.za",
            "name": "Ross Donnachie",
            "username": "radonnachie"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b15b4b8a085107e5ab04f24504a4b8f77ac30a16",
          "message": "*^ memory/vector._refs host pointer (#67)\n\n* *^ memory/vector._refs host pointer\r\nfixing seg-faults seen under WSL2 docker runs by @radonnachie\r\n\r\n* @ minor version bump",
          "timestamp": "2024-01-29T16:45:01-03:00",
          "tree_id": "f69c2306cac2f3c40049af17abf65129c4c0bf2d",
          "url": "https://github.com/luigifcruz/blade/commit/b15b4b8a085107e5ab04f24504a4b8f77ac30a16"
        },
        "date": 1706559267662,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Cast_Compute_CF32_CF32/2/iterations:16384/manual_time",
            "value": 0.002122113237745804,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009413378112792968 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/16/iterations:16384/manual_time",
            "value": 0.002142035112402685,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009495061584472655 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CF32_CF32/64/iterations:16384/manual_time",
            "value": 0.0021217382416491293,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.009367104980468749 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/2/iterations:16384/manual_time",
            "value": 0.1788705960814596,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.18682393518066406 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/16/iterations:16384/manual_time",
            "value": 1.330390661323122,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3386229291381837 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF32/64/iterations:16384/manual_time",
            "value": 5.286006136316246,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 5.294504971923828 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/2/iterations:16384/manual_time",
            "value": 0.12945591354052155,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.13754823187255882 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/16/iterations:16384/manual_time",
            "value": 0.9351017168022224,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.9433267557983401 ms\nthreads: 1"
          },
          {
            "name": "BM_Cast_Compute_CI8_CF16/64/iterations:16384/manual_time",
            "value": 3.703104415748726,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 3.711595206359863 ms\nthreads: 1"
          },
          {
            "name": "BM_Channelizer_Compute/16/8192/iterations:16384/manual_time",
            "value": 6.5737189223114,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.5821974039916995 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/8/4/iterations:16384/manual_time",
            "value": 0.3098154174914214,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.31795593841552755 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/8/4/iterations:16384/manual_time",
            "value": 2.351374510368487,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.359866480407712 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/32/4/iterations:16384/manual_time",
            "value": 0.7507631810064197,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.7589091986083991 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/32/4/iterations:16384/manual_time",
            "value": 6.005603202964949,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.014092128601076 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/4/iterations:16384/manual_time",
            "value": 0.7956874194157137,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.8037753540649437 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/4/iterations:16384/manual_time",
            "value": 6.352782605972607,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 6.361270210510254 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/2/64/1/iterations:16384/manual_time",
            "value": 0.36812592437129865,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3761138843383846 ms\nthreads: 1"
          },
          {
            "name": "BM_Detector_Compute/16/64/1/iterations:16384/manual_time",
            "value": 2.8726196864283793,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.8809978765258757 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/2/5/iterations:16384/manual_time",
            "value": 0.27740693815125894,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.28560930804443635 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF32_CF32/16/5/iterations:16384/manual_time",
            "value": 2.1358726090738855,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.144470196777347 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/2/5/iterations:16384/manual_time",
            "value": 0.15134581807796366,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.1594662775878941 ms\nthreads: 1"
          },
          {
            "name": "BM_Polarizer_Compute_CF16_CF16/16/5/iterations:16384/manual_time",
            "value": 1.1163560503533176,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.1248433615722655 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.645800618706005,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.6540528172607405 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_CF32_CF32/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.29179655091304824,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.29998790649413826 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/192/8192/iterations:16384/manual_time",
            "value": 0.3270386480487275,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.3351647779541028 ms\nthreads: 1"
          },
          {
            "name": "BM_Gather_Compute_F16_F16/2/64/131072/1/iterations:16384/manual_time",
            "value": 0.2573914824495205,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.2655452612304668 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_CF32_CF32/iterations:16384/manual_time",
            "value": 2.577838878636385,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.586332678833007 ms\nthreads: 1"
          },
          {
            "name": "BM_Duplicate_Compute_F32_F32/iterations:16384/manual_time",
            "value": 1.2921747199428069,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3004956834106394 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/1/2/3/iterations:16384/manual_time",
            "value": 0.5680142506463426,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.576216965454103 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/1/3/iterations:16384/manual_time",
            "value": 0.8415907401513323,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 0.849867992431641 ms\nthreads: 1"
          },
          {
            "name": "BM_Permutation_Compute/0/2/3/1/iterations:16384/manual_time",
            "value": 1.2856893741144404,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.2940508862915034 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3355813882967027,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.3439012787475544 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4107424206599717,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4190514302978505 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/8/20/iterations:16384/manual_time",
            "value": 1.828523877620114,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.836910322631835 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerATA_Compute/1/42/iterations:16384/manual_time",
            "value": 2.707579232264834,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.7159704015502912 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/1/20/iterations:16384/manual_time",
            "value": 1.3363010644980022,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.344820441711421 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/2/20/iterations:16384/manual_time",
            "value": 1.4112501760124019,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 1.4195765661010746 ms\nthreads: 1"
          },
          {
            "name": "BM_BeamformerMEERKAT_Compute/8/20/iterations:16384/manual_time",
            "value": 2.048172548782645,
            "unit": "ms/iter",
            "extra": "iterations: 16384\ncpu: 2.0565349686279264 ms\nthreads: 1"
          }
        ]
      }
    ],
    "Bundles Benchmark": [
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
        "date": 1703111740470,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.332753275390626,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3156.7090345 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 88.3077777109375,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 22311.702942 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.008572763671875,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2817.935093999999 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "066482a56dd5a89ebc6c8c75ae98a788b5dee039",
          "message": "Merge pull request #60 from luigifcruz/dev\n\n[V1.0] Development Work",
          "timestamp": "2023-12-20T19:37:32-03:00",
          "tree_id": "7facbc7f9032132eb8833d0aa465b7fa4e9c4971",
          "url": "https://github.com/luigifcruz/blade/commit/066482a56dd5a89ebc6c8c75ae98a788b5dee039"
        },
        "date": 1703113481973,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.36511054296875,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3165.1257434999998 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 88.25702022265625,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 22308.408436 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.008122259765624,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2817.923417500001 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1946561f4d860df2ae1c23e94a80c1d1d3c45a91",
          "message": "Add minimal CUDA version.",
          "timestamp": "2024-01-18T21:59:18-03:00",
          "tree_id": "60dea43b99c4d915f05924c78f1e961afe5376ca",
          "url": "https://github.com/luigifcruz/blade/commit/1946561f4d860df2ae1c23e94a80c1d1d3c45a91"
        },
        "date": 1705627617742,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.358883833984375,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3159.3790299999996 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 88.23610683398438,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 22284.2425295 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.011184380859376,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2818.744893000002 ms\nthreads: 1"
          }
        ]
      },
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
          "id": "9e89758e3f332706120f1593390ba6ee5e2b0b81",
          "message": "V1.0.1",
          "timestamp": "2024-01-18T16:19:59Z",
          "url": "https://github.com/luigifcruz/blade/pull/66/commits/9e89758e3f332706120f1593390ba6ee5e2b0b81"
        },
        "date": 1705629287601,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.367783230468751,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3165.8476505 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 88.44291044921874,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 22368.268203 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.025843082031251,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2822.464824499999 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "luigifcruz@gmail.com",
            "name": "Luigi Cruz",
            "username": "luigifcruz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df5f9185646fef36256e141c0a7818c6b931cba3",
          "message": "Merge pull request #66 from luigifcruz/v1.0.1\n\nV1.0.1\r\n\r\n- Remove unnecessary <span> include. \r\n- Change workdir place in Dockerfile.",
          "timestamp": "2024-01-18T22:22:46-03:00",
          "tree_id": "169d76a8efa9b55fe86bb795751030eeb8bcd7a6",
          "url": "https://github.com/luigifcruz/blade/commit/df5f9185646fef36256e141c0a7818c6b931cba3"
        },
        "date": 1705630912012,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.382897142578125,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3165.4459144999996 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 88.17277755078125,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 22275.5423785 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.026673548828127,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2822.709089 ms\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "code@radonn.co.za",
            "name": "Ross Donnachie",
            "username": "radonnachie"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b15b4b8a085107e5ab04f24504a4b8f77ac30a16",
          "message": "*^ memory/vector._refs host pointer (#67)\n\n* *^ memory/vector._refs host pointer\r\nfixing seg-faults seen under WSL2 docker runs by @radonnachie\r\n\r\n* @ minor version bump",
          "timestamp": "2024-01-29T16:45:01-03:00",
          "tree_id": "f69c2306cac2f3c40049af17abf65129c4c0bf2d",
          "url": "https://github.com/luigifcruz/blade/commit/b15b4b8a085107e5ab04f24504a4b8f77ac30a16"
        },
        "date": 1706559318937,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_BundleATAModeB/iterations:2/manual_time",
            "value": 12.365083328124998,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 3160.8362165 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeBH/iterations:2/manual_time",
            "value": 69.1484728515625,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 17700.744611 ms\nthreads: 1"
          },
          {
            "name": "BM_BundleATAModeH/iterations:2/manual_time",
            "value": 11.023394400390625,
            "unit": "ms/iter",
            "extra": "iterations: 2\ncpu: 2821.695034000001 ms\nthreads: 1"
          }
        ]
      }
    ]
  }
}