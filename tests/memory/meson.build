test('memory', executable(
    'blade-memory-reference', 'reference.cc',
    dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

if has_python
    test(
        'blade-memory-matrix',
        find_program('python', 'python3'),
        args: files('matrix.py'),
        is_parallel: false,
        timeout: 0,
        env: 'PYTHONPATH=@0@'.format(python_path))
endif
