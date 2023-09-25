if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    if has_python
        test(
            'test-examples-python-async',
            find_program('python', 'python3'),
            args: files('pipeline_async.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-examples-python-sync',
            find_program('python', 'python3'),
            args: files('pipeline_sync.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif