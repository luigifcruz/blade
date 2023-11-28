if cfg_lst.get('BLADE_BUNDLE_ATA_MODE_B', false)
    if py.found()
        test(
            'test-bundles-ata-modeb',
            py,
            args: files('mode_b.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif

if cfg_lst.get('BLADE_BUNDLE_ATA_MODE_B', false) and \
   cfg_lst.get('BLADE_BUNDLE_GENERIC_MODE_H', false)
    if py.found()
        test(
            'test-bundles-ata-modebh',
            py,
            args: files('mode_bh.py'),
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif