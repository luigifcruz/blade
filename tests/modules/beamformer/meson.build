if cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false)
    if py.found()
        test(
            'test-modules-beamformer-ata',
            py,
            args: files('ata.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
