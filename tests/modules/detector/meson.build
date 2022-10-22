if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    if has_python
        test(
            'blade-detector-cross-pol-python',
            find_program('python', 'python3'),
            args: files('cross_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-detector-single-pol-python',
            find_program('python', 'python3'),
            args: files('single_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    
        test(
            'blade-detector-stepped-python',
            find_program('python', 'python3'),
            args: files('stepped.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
