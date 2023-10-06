if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    if py.found()
        test(
            'test-modules-polarizer',
            py,
            args: files('polarizer.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
