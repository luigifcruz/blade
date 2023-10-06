if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    if py.found()
        test(
            'test-modules-detector-cross-pol',
            py,
            args: files('cross_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-detector-single-pol',
            py,
            args: files('single_pol.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
