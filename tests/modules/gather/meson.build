if cfg_lst.get('BLADE_MODULE_GATHER', false)
    if py.found()
        # ATA-like accumulation.
        test(
            'test-modules-gather-ata',
            py,
            args: [files('gather.py'), '2', '192', '8192', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # VLA-like accumulation.
        test(
            'test-modules-gather-vla',
            py,
            args: [files('gather.py'), '27', '131072', '3', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # Miscellaneous configurations.

        test(
            'test-modules-gather-misc-1',
            py,
            args: [files('gather.py'), '3', '7', '3', '1', '0', '5'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gather-misc-2',
            py,
            args: [files('gather.py'), '1', '2', '5', '6', '1', '6'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gather-misc-3',
            py,
            args: [files('gather.py'), '11', '1', '13', '9', '2', '7'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gather-misc-4',
            py,
            args: [files('gather.py'), '9', '2', '1', '3', '3', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gather-misc-5',
            py,
            args: [files('gather.py'), '9', '2', '5555', '3', '2', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
