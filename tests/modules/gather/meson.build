if cfg_lst.get('BLADE_MODULE_GATHER', false)
    if py.found()
        test(
            'test-modules-gather',
            py,
            args: files('gather.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
