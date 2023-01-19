if cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false)
    benchmark('beamformer-ata-benchmark', executable(
        'blade-beamformer-ata-benchmark', ['ata.cc'],
        dependencies: lib_blade_dep,
    ), timeout: 0)
endif

