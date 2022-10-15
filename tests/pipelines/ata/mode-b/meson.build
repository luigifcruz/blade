if cfg_lst.get('BLADE_PIPELINE_ATA_MODE_B', false)
    test('blade-ata-mode-b', executable(
        'blade-ata-mode-b', ['mode_b.c', 'mode_b.cc'],
        dependencies: lib_blade_dep,
    ), is_parallel: false, timeout: 0)
endif
