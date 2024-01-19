window.BENCHMARK_DATA = {
  "lastUpdate": 1705627558240,
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
      }
    ]
  }
}