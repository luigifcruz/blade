if cfg_lst.get('BLADE_PIPELINE_GENERIC_MODE_H', false)  
   if cfg_lst.get('BLADE_PIPELINE_ATA_MODE_B', false)
        benchmark('blade-ata-mode-bh-benchmark', executable(
            'blade-ata-mode-bh-benchmark', ['mode_bh_stub.c', 'mode_bh.cc', 'benchmark.cc'],
            dependencies: lib_blade_dep,
        ), timeout: 0)
    endif
endif
