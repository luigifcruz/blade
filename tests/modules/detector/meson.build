if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    if has_python
        test(
            'test-modules-detector-cross-pol',
            find_program('python', 'python3'),
            args: files('cross_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-modules-detector-single-pol',
            find_program('python', 'python3'),
            args: files('single_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
