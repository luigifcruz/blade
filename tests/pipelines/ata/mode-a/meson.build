if cfg_lst.get('BLADE_PIPELINE_ATA_MODE_A', false)
    test('blade-ata-mode-a', executable(
        'blade-ata-mode-a', ['mode_a.c', 'mode_a.cc'],
        dependencies: lib_blade_dep,
    ), is_parallel: false, timeout: 0)
endif
