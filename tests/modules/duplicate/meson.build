if cfg_lst.get('BLADE_MODULE_DUPLICATE', false)
    if has_python
        test(
            'test-modules-duplicate',
            find_program('python', 'python3'),
            args: files('duplicate.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
