project(
  'blade',
  ['cpp', 'c', 'cuda'],
  version: '0.7.0',
  default_options: [
    'buildtype=release',
    'cpp_std=c++20',
    'b_ndebug=if-release',
  ]
)

cfg_lst = configuration_data()
cfg_lst.set('version', meson.project_version())
cfg_lst.set('build_type', get_option('buildtype'))

src_lst = []
dep_lst = [
  subproject('radiointerferometryc99').get_variable('lib_radiointerferometry_dep'),
  dependency('cuda', version : '>=11', modules : ['nvrtc', 'cudart', 'cuda', 'cufft']),
  dependency('fmt'),
  dependency('benchmark'),
]
inc_lst = [
  include_directories('include'),
  include_directories('.'),
]
jit_lst = []

subdir('src')
subdir('include')
subdir('tools')

lib_blade = library(
  'blade',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  gnu_symbol_visibility: 'hidden',
  install: true,
)

lib_blade_dep = declare_dependency(
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lib_blade,
)

subdir('python')
subdir('tests')
subdir('docs')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
         'python bindings': has_python,
        }, section: 'General', bool_yn: true)

pkg = import('pkgconfig')
pkg.generate(
  lib_blade,
  libraries: dep_lst,
)
