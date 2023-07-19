if cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false)
    if has_python
        test(
            'test-modules-beamformer-ata',
            find_program('python', 'python3'),
            args: files('ata.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
