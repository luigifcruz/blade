if cfg_lst.get('BLADE_BUNDLE_ATA_MODE_B', false)
    benchmark(
        'ata-mode-b-benchmark', 
        executable(
            'ata-mode-b-benchmark', 
            ['benchmark.cc'],
            dependencies: lib_blade_dep,
        ), 
        timeout: 0,
    )
endif
