if cfg_lst.get('BLADE_MODULE_PERMUTATION', false)
    if py.found()
        test(
            'test-modules-permutation',
            py,
            args: files('permutation.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
