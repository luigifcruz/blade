if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    benchmark('detector-benchmark', executable(
        'blade-detector-benchmark', ['generic.cc'],
        dependencies: lib_blade_dep,
    ), timeout: 0)
endif
