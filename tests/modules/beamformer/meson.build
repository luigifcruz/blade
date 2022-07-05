test('beamformer-ata', executable(
  'blade-beamformer-ata', ['pipeline_ata.hh', 'ata.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

executable(
  'blade-beamformer-ata-benchmark', ['pipeline_ata.hh', 'ata_benchmark.cc'],
  dependencies: lib_blade_dep,
)

test('beamformer-vla', executable(
  'blade-beamformer-vla', ['pipeline_vla.hh', 'vla.cc'],
  dependencies: lib_blade_dep,
), is_parallel: false, timeout: 0)

executable(
  'blade-beamformer-vla-benchmark', ['pipeline_vla.hh', 'vla_benchmark.cc'],
  dependencies: lib_blade_dep,
)

if has_python
  test(
    'blade-beamformer-vla-python',
    find_program('python', 'python3'),
    args: files('vla.py'),
    is_parallel: false,
    timeout: 0,
    env: 'PYTHONPATH=@0@'.format(python_path))
endif
