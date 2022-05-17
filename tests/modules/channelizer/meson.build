test('channelizer', executable(
  'blade-channelizer', 'advanced.cc',
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

if has_python
  test(
    'blade-channelize-hires-python',
    find_program('python', 'python3'),
    args: [files('trial.py'), '1', '5', '100', '200', '1', '200'],
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))

  test(
    'blade-channelize-std-python',
    find_program('python', 'python3'),
    args: [files('trial.py'), '1', '20', '96', '35000', '2', '4'],
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))

  test(
    'blade-channelize-std-beams-python',
    find_program('python', 'python3'),
    args: [files('trial.py'), '2', '20', '96', '35000', '2', '4'],
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))

  test(
    'blade-channelize-weird-python',
    find_program('python', 'python3'),
    args: [files('trial.py'), '2', '10', '20', '30', '4', '5'],
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))
endif
