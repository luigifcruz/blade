if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    if has_python
        test(
            'test-modules-channelizer-weird',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '5', '100', '200', '1', '200'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-modules-channelizer-hires',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '2', '1024'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-modules-channelizer-hires-single-pol',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '1', '1024'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-modules-channelizer-fail-case-1',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '1', '512'],
            is_parallel: false,
            should_fail: true,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )

        test(
            'test-modules-channelizer-fail-case-2',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1025', '1', '1025'],
            is_parallel: false,
            should_fail: true,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path),
        )
    endif
endif
