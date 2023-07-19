if cfg_lst.get('BLADE_MODULE_ATA_PHASOR', false)
    if has_python
        test(
            'test-modules-phasor',
            find_program('python', 'python3'),
            args: files('advanced.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
