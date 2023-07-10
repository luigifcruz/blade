if cfg_lst.get('BLADE_MODULE_CAST', false)
    test('cast', executable(
        'blade-cast', 'basic.cc',
        dependencies: lib_blade_dep,
    ), is_parallel: false, timeout: 0)
endif

# TODO: Convert Cast test to Python.