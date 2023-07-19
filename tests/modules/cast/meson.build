if cfg_lst.get('BLADE_MODULE_CAST', false)
    test(
        'test-modules-cast',
        executable(
            'test-modules-cast',
            'basic.cc',
            dependencies: lib_blade_dep,
        ),
        is_parallel: false,
        timeout: 0,
    )
endif

# TODO: Convert Cast test to Python.