project(
  'blade',
  ['cpp', 'c', 'cuda'],
  version: '0.1',
  default_options: [
    'buildtype=release',
    'cpp_std=c++20',
    'b_ndebug=if-release',
  ]
)

cfg_lst = configuration_data()
cfg_lst.set('VERSION', meson.project_version())
cfg_lst.set('BUILD_TYPE', get_option('prefix'))

src_lst = []
dep_lst = [
  dependency('spdlog'),
  dependency('cuda', version : '>=11', modules : ['nvrtc', 'cudart', 'cuda']),
]
inc_lst = [
  include_directories('include'),
  include_directories('.'),
]
jit_lst = []

subdir('src')
subdir('include')
subdir('tools')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

lib_blade = library(
  'blade',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
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

pkg = import('pkgconfig')
pkg.generate(
  lib_blade,
  libraries: dep_lst,
)
