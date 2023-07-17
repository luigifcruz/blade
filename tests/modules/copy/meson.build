if cfg_lst.get('BLADE_MODULE_COPY', false)
    if has_python
        test(
            'blade-copy-python',
            find_program('python', 'python3'),
            args: files('copy.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
