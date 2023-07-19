if cfg_lst.get('BLADE_LOADER_GTEST', false)
    test(
        'test-memory-device-ops',
        executable(
            'test-memory-device-ops',
            'ops/base.cu',
            dependencies: lib_blade_dep,
            cuda_args: cuda_args,
        ),
        is_parallel: false,
        timeout: 0,
    )
endif