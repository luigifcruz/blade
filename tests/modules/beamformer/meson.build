test('beamformer-ata', executable(
  'blade-beamformer-ata', ['pipeline_ata.hh', 'ata.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

test('beamformer-vla', executable(
  'blade-beamformer-vla', ['pipeline_vla.hh', 'vla.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

if has_python
  test(
    'blade-beamformer-vla-python',
    find_program('python', 'python3'),
    args: files('vla.py'),
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))
endif
