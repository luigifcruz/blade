subdir('collection')

benchmark('memory-ops-benchmark', executable(
    'blade-memory-ops-benchmark', ['ops.cu'],
    dependencies: lib_blade_dep,
    cuda_args: cuda_args,
), timeout: 0)
