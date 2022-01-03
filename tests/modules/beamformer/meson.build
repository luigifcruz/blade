test('beamformer-ata', executable(
  'blade-beamformer-ata', ['pipeline.hh', 'ata.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

test('beamformer-ata-runner', executable(
  'blade-beamformer-ata-runner', ['pipeline.hh', 'ata_runner.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

if has_python
  test(
    'blade-beamformer-ata-python',
    find_program('python', 'python3'),
    args: files('ata.py'),
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))
endif
