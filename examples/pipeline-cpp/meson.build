if cfg_lst.get('BLADE_MODULE_CAST', false) and \
   cfg_lst.get('BLADE_MODULE_GATHER', false)
    test(
        'test-examples-cpp',
         executable(
            'test-examples-cpp', 
            ['pipeline.cc'],
            dependencies: lib_blade_dep,
        ),
        is_parallel: false,
        timeout: 0,
    )
endif