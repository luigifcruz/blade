if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    if has_python
        test(
            'test-modules-polarizer',
            find_program('python', 'python3'),
            args: files('polarizer.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
