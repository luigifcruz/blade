if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    benchmark('channelizer-benchmark', executable(
        'blade-channelizer-benchmark', ['generic.cc'],
        dependencies: lib_blade_dep,
    ), timeout: 0)
endif
