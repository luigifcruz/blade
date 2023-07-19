if cfg_lst.get('BLADE_MODULE_GATHER', false)
    if has_python
        test(
            'test-modules-gather',
            find_program('python', 'python3'),
            args: files('gather.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
