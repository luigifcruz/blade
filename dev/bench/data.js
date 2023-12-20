window.BENCHMARK_DATA = {
  "lastUpdate": 1703109724304,
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
          "id": "82fa0599bce126b499f0052318ecedc73f06134f",
          "message": "[V1.0] Development Work",
          "timestamp": "2023-12-10T14:45:40Z",
          "url": "https://github.com/luigifcruz/blade/pull/60/commits/82fa0599bce126b499f0052318ecedc73f06134f"
        },
        "date": 1703108742447,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5955.562298798582,
            "unit": "ns/iter",
            "extra": "iterations: 103676\ncpu: 12621.222346541148 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8683.677685178209,
            "unit": "ns/iter",
            "extra": "iterations: 82132\ncpu: 15741.058637315538 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37818.24412862437,
            "unit": "ns/iter",
            "extra": "iterations: 18365\ncpu: 44750.36046828208 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71077.75467157764,
            "unit": "ns/iter",
            "extra": "iterations: 8810\ncpu: 77974.18626560725 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71533.2638072002,
            "unit": "ns/iter",
            "extra": "iterations: 8802\ncpu: 78465.04192229043 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138250.4136555362,
            "unit": "ns/iter",
            "extra": "iterations: 4602\ncpu: 145146.66058235554 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135468.640413015,
            "unit": "ns/iter",
            "extra": "iterations: 4723\ncpu: 142391.28583527412 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 270412.877126746,
            "unit": "ns/iter",
            "extra": "iterations: 2573\ncpu: 277289.7038476488 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 70915.40349302234,
            "unit": "ns/iter",
            "extra": "iterations: 9453\ncpu: 77887.25780175606 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138366.62751318474,
            "unit": "ns/iter",
            "extra": "iterations: 4603\ncpu: 145273.15033673673 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37866.955100347295,
            "unit": "ns/iter",
            "extra": "iterations: 18121\ncpu: 44775.9243971083 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 71112.2093935446,
            "unit": "ns/iter",
            "extra": "iterations: 8799\ncpu: 78046.45402886678 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71550.88125606538,
            "unit": "ns/iter",
            "extra": "iterations: 8770\ncpu: 78462.20877993165 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138199.96267233847,
            "unit": "ns/iter",
            "extra": "iterations: 4608\ncpu: 145123.9759114586 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135461.37906963512,
            "unit": "ns/iter",
            "extra": "iterations: 4709\ncpu: 142371.35527712878 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270601.87173557933,
            "unit": "ns/iter",
            "extra": "iterations: 2533\ncpu: 277522.8436636402 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71521.7561485081,
            "unit": "ns/iter",
            "extra": "iterations: 8735\ncpu: 78449.72386949063 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138397.09522588132,
            "unit": "ns/iter",
            "extra": "iterations: 4587\ncpu: 145332.25703073863 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37839.4152918716,
            "unit": "ns/iter",
            "extra": "iterations: 18105\ncpu: 44761.85230599281 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71098.25625538244,
            "unit": "ns/iter",
            "extra": "iterations: 8823\ncpu: 78016.46492122862 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71572.8155719649,
            "unit": "ns/iter",
            "extra": "iterations: 8794\ncpu: 78523.88560382058 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138155.24940874393,
            "unit": "ns/iter",
            "extra": "iterations: 4593\ncpu: 145005.2146745051 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135544.10629431836,
            "unit": "ns/iter",
            "extra": "iterations: 4688\ncpu: 142490.9667235488 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 270555.0368975135,
            "unit": "ns/iter",
            "extra": "iterations: 2499\ncpu: 277395.0904361757 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71676.80284787738,
            "unit": "ns/iter",
            "extra": "iterations: 8744\ncpu: 78619.25194419014 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138250.86614509713,
            "unit": "ns/iter",
            "extra": "iterations: 4587\ncpu: 145132.18683235254 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37863.25116824059,
            "unit": "ns/iter",
            "extra": "iterations: 18357\ncpu: 44804.087759437636 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 71115.7629316867,
            "unit": "ns/iter",
            "extra": "iterations: 8864\ncpu: 77973.94629963925 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71592.57913835044,
            "unit": "ns/iter",
            "extra": "iterations: 8737\ncpu: 78504.62160924834 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138337.97621781752,
            "unit": "ns/iter",
            "extra": "iterations: 4576\ncpu: 145254.6700174827 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147412.6440863043,
            "unit": "ns/iter",
            "extra": "iterations: 4364\ncpu: 154337.69546287804 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 294084.0272292406,
            "unit": "ns/iter",
            "extra": "iterations: 2314\ncpu: 300985.2130509944 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71697.14084574077,
            "unit": "ns/iter",
            "extra": "iterations: 8777\ncpu: 78571.88002734419 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138347.0228941792,
            "unit": "ns/iter",
            "extra": "iterations: 4603\ncpu: 145311.29545948407 ns\nthreads: 1"
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
          "id": "b7520474578bbe390bbeacdd17cd14b97aa995fc",
          "message": "[V1.0] Development Work",
          "timestamp": "2023-12-10T14:45:40Z",
          "url": "https://github.com/luigifcruz/blade/pull/60/commits/b7520474578bbe390bbeacdd17cd14b97aa995fc"
        },
        "date": 1703109722734,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/1048576/manual_time",
            "value": 5982.407122758742,
            "unit": "ns/iter",
            "extra": "iterations: 104297\ncpu: 12684.287457932634 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::NOOP>/2097152/manual_time",
            "value": 8603.507242253241,
            "unit": "ns/iter",
            "extra": "iterations: 82407\ncpu: 15571.426019634255 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 37837.05162413793,
            "unit": "ns/iter",
            "extra": "iterations: 18346\ncpu: 44848.34879537772 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 71045.046009672,
            "unit": "ns/iter",
            "extra": "iterations: 8760\ncpu: 78043.24155251136 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 71559.4190711759,
            "unit": "ns/iter",
            "extra": "iterations: 8723\ncpu: 78545.72383354344 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 138593.09491512636,
            "unit": "ns/iter",
            "extra": "iterations: 4576\ncpu: 145553.9449300698 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/1048576/manual_time",
            "value": 135628.46207294805,
            "unit": "ns/iter",
            "extra": "iterations: 4752\ncpu: 142486.57344276094 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::ADD>/2097152/manual_time",
            "value": 271096.93329550157,
            "unit": "ns/iter",
            "extra": "iterations: 2516\ncpu: 278016.197138315 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/1048576/manual_time",
            "value": 10511360168.457031,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 10525274.000000807 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::ADD>/2097152/manual_time",
            "value": 16762880325.317383,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16755927.00000017 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 37822.67413561551,
            "unit": "ns/iter",
            "extra": "iterations: 18254\ncpu: 44783.06710857895 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 70877.15527655938,
            "unit": "ns/iter",
            "extra": "iterations: 8793\ncpu: 77872.25952462171 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 71485.40492024513,
            "unit": "ns/iter",
            "extra": "iterations: 8738\ncpu: 78488.87800411983 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 138124.4216438228,
            "unit": "ns/iter",
            "extra": "iterations: 4600\ncpu: 145161.68347826108 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/1048576/manual_time",
            "value": 135538.21432153886,
            "unit": "ns/iter",
            "extra": "iterations: 4727\ncpu: 142491.82906706145 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::SUB>/2097152/manual_time",
            "value": 270972.4180239297,
            "unit": "ns/iter",
            "extra": "iterations: 2498\ncpu: 277970.58927141735 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/1048576/manual_time",
            "value": 9908320426.940918,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9880174.99999927 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::SUB>/2097152/manual_time",
            "value": 17157119750.976562,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 17181435.99999955 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 37835.1692031376,
            "unit": "ns/iter",
            "extra": "iterations: 18361\ncpu: 45088.48935243178 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 71093.86156324757,
            "unit": "ns/iter",
            "extra": "iterations: 8805\ncpu: 78338.16013628645 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 71582.87412402914,
            "unit": "ns/iter",
            "extra": "iterations: 8800\ncpu: 78858.64681818149 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 138326.62518078106,
            "unit": "ns/iter",
            "extra": "iterations: 4589\ncpu: 145506.2482022228 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/1048576/manual_time",
            "value": 135574.40183944107,
            "unit": "ns/iter",
            "extra": "iterations: 4709\ncpu: 142577.55213421097 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::MULT>/2097152/manual_time",
            "value": 271107.22015579726,
            "unit": "ns/iter",
            "extra": "iterations: 2498\ncpu: 278094.53202562017 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/1048576/manual_time",
            "value": 9833472251.89209,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9810012.999999173 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::MULT>/2097152/manual_time",
            "value": 16928800582.885742,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16950866.000001952 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 37823.68014637244,
            "unit": "ns/iter",
            "extra": "iterations: 18339\ncpu: 44746.296853699794 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F16, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 70761.78577705074,
            "unit": "ns/iter",
            "extra": "iterations: 8768\ncpu: 77735.76151916049 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 71576.54355829902,
            "unit": "ns/iter",
            "extra": "iterations: 8736\ncpu: 78485.58333333334 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F32, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 138545.41124047598,
            "unit": "ns/iter",
            "extra": "iterations: 4572\ncpu: 145759.82895887972 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/1048576/manual_time",
            "value": 147641.6452697045,
            "unit": "ns/iter",
            "extra": "iterations: 4359\ncpu: 154806.5627437489 ns\nthreads: 1"
          },
          {
            "name": "OpsComplexKernelBenchmark<F64, ArithmeticOp::DIV>/2097152/manual_time",
            "value": 292919.59457537264,
            "unit": "ns/iter",
            "extra": "iterations: 2312\ncpu: 299888.3520761235 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/1048576/manual_time",
            "value": 9495552062.988281,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 9510924.000000643 ns\nthreads: 1"
          },
          {
            "name": "CuComplexKernelBenchmark<ArithmeticOp::DIV>/2097152/manual_time",
            "value": 16812032699.58496,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 16781907.99999868 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}