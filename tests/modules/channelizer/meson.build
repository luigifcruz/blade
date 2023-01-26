if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    if has_python
        test(
            'blade-channelize-hires-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '5', '100', '200', '1', '200'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-std-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '20', '96', '35000', '2', '4'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))

        test(
            'blade-channelize-std-beams-python',
            find_program('python', 'python3'),
            args: [files('advanced.py'), '20', '96', '35000', '2', '4'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
