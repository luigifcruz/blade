
executable(
  'blade_guppiraw', ['guppiraw.cc'],
  dependencies: [
    subproject('guppirawc99').get_variable('lib_guppiraw_dep'),
    lib_blade_dep
  ],
  install: true
)