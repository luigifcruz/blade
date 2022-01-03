test('channelizer', executable(
  'blade-channelizer', 'advanced.cc',
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

if has_python
  test(
    'blade-channelize-python',
    find_program('python', 'python3'),
    args: files('advanced.py'),
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))
endif
