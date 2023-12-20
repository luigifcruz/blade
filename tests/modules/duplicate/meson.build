if cfg_lst.get('BLADE_MODULE_DUPLICATE', false)
    if py.found()
        test(
            'test-modules-duplicate',
            py,
            args: files('duplicate.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
