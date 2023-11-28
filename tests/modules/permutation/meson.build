if cfg_lst.get('BLADE_MODULE_PERMUTATION', false)
    if py.found()
        test(
            'test-modules-permutation-copy',
            py,
            args: [files('permutation.py'), '0', '1', '2', '3'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutation-single',
            py,
            args: [files('permutation.py'), '0', '2', '1', '3'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutation-reverse',
            py,
            args: [files('permutation.py'), '3', '2', '1', '0'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-permutation-random',
            py,
            args: [files('permutation.py'), '2', '0', '3', '1'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif