if cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false)
    test('beamformer-ata', executable(
        'blade-beamformer-ata', ['ata.cc'],
        dependencies: lib_blade_dep,
    ), is_parallel: false, timeout: 0)

    if has_python
        test(
            'blade-beamformer-ata-python',
            find_program('python', 'python3'),
            args: files('ata.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
