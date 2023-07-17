if cfg_lst.get('BLADE_MODULE_PERMUTATION', false)
    if has_python
        test(
            'blade-permutation-python',
            find_program('python', 'python3'),
            args: files('permutation.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(python_path))
    endif
endif
