if cfg_lst.get('BLADE_MODULE_CAST', false)
    if py.found()
        test(
            'test-modules-cast',
            py,
            args: files('cast.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
