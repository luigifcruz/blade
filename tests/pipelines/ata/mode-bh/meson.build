if cfg_lst.get('BLADE_PIPELINE_GENERIC_MODE_H', false)  
   if cfg_lst.get('BLADE_PIPELINE_ATA_MODE_B', false)
        test('blade-ata-mode-bh', executable(
            'blade-ata-mode-bh', ['mode_bh.c', 'mode_bh.cc'],
            dependencies: lib_blade_dep,
        ), is_parallel: false, timeout: 0)
    endif
endif
