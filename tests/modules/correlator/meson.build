if cfg_lst.get('BLADE_MODULE_CORRELATOR', false)
    if py.found()
        # A, F, T, P, I, S, C, B

        #
        # ATA Time Domain Optimized
        #

        test(
            'test-modules-correlator-ata-time-fp32',
            py,
            args: [files('correlator.py'), '28', '192', '4096', '2',  '8', '0', '1', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-time-shamem-fp32',
            py,
            args: [files('correlator.py'), '28', '192', '4096', '2',  '8', '1', '1', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-time-fp64',
            py,
            args: [files('correlator.py'), '28', '192', '4096', '2',  '8', '0', '2', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-time-shamem-fp64',
            py,
            args: [files('correlator.py'), '28', '192', '4096', '2',  '8', '1', '2', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        #
        # ATA Frequency Domain Optimized
        #

        test(
            'test-modules-correlator-ata-frequency-fp32',
            py,
            args: [files('correlator.py'), '28', '4096', '1', '2',  '8', '0', '1', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-frequency-shamem-fp32',
            py,
            args: [files('correlator.py'), '28', '4096', '1', '2',  '8', '1', '1', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-frequency-fp64',
            py,
            args: [files('correlator.py'), '28', '4096', '1', '2',  '8', '0', '2', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-correlator-ata-frequency-shamem-fp64',
            py,
            args: [files('correlator.py'), '28', '4096', '1', '2',  '8', '1', '2', '32'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
