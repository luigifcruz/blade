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

# TODO: Add Mode-HB test.