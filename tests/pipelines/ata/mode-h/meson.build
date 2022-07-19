if cfg_lst.get('BLADE_PIPELINE_ATA_MODE_H', false)
    test('blade-ata-mode-h', executable(
        'blade-ata-mode-h', ['mode_h.c', 'mode_h.cc'],
        dependencies: lib_blade_dep,
    ), is_parallel: false, timeout: 0)
endif
