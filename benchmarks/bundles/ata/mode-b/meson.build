if cfg_lst.get('BLADE_BUNDLE_ATA_MODE_B', false)
    benchmark('blade-ata-mode-b-benchmark', executable(
        'blade-ata-mode-b-benchmark', ['mode_b_stub.c', 'mode_b.cc', 'benchmark.cc'],
        dependencies: lib_blade_dep,
    ), timeout: 0)
endif
