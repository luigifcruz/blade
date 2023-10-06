if cfg_lst.get('BLADE_MODULE_ATA_PHASOR', false)
    if py.found()
        test(
            'test-modules-phasor',
            py,
            args: files('advanced.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
