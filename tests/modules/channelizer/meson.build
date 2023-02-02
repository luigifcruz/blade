if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    if has_python
        test(
            'blade-channelize-hires-weird-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '5', '100', '200', '1', '200'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-hires-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '2', '1024'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-hires-single-pol-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '1', '1024'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-fail-case-1-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1024', '1', '512'],
            is_parallel: false,
            should_fail: true,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-fail-case-2-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '8', '128', '1025', '1', '1025'],
            is_parallel: false,
            should_fail: true,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
