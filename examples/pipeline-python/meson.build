if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    if py.found()
        test(
            'test-examples-python-async',
            py,
            args: files('pipeline_async.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-examples-python-sync',
            py,
            args: files('pipeline_sync.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif