test(
    'test-memory-vector-reference',
    executable(
        'test-memory-vector-reference', 
        'reference.cc',
        dependencies: lib_blade_dep,
        cuda_args: cuda_args,
    ), 
    is_parallel: false, 
    timeout: 0,
)

if py.found()
    test(
        'test-memory-vector-matrix',
        py,
        args: files('matrix.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(py_build_path),
    )
endif