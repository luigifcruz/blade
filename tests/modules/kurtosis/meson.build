if cfg_lst.get('BLADE_MODULE_KURTOSIS', false)
    if py.found()
        test(
            'test-modules-kurtosis',
            py,
            args: files('kurtosis.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
