if py.found()
    test(
        'test-memory-copy',
        py,
        args: files('copy.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )
endif