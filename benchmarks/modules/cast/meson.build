if cfg_lst.get('BLADE_MODULE_CAST', false)
    benchmark('cast-benchmark', executable(
        'blade-cast-benchmark', ['generic.cc'],
        dependencies: lib_blade_dep,
    ), timeout: 0)
endif
