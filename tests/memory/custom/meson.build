if py.found()
    test(
        'test-memory-custom-array-tensor',
        py,
        args: files('array_tensor.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )

    test(
        'test-memory-custom-delay-tensor',
        py,
        args: files('delay_tensor.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )

    test(
        'test-memory-custom-phasor-tensor',
        py,
        args: files('phasor_tensor.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )

    test(
        'test-memory-custom-tensor',
        py,
        args: files('tensor.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )
endif