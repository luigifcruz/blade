project(
  'blade',
  ['cpp', 'c', 'cuda'],
  version: '0.0.0.3',
  default_options: [
    'buildtype=release',
    'cpp_std=c++20',
    'b_ndebug=if-release',
  ]
)

py = import('python').find_installation('python3')

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
  dependency('spdlog'),
  dependency('pybind11'),
  py.dependency(embed: true),
  dependency('cuda', version : '>=11', modules : ['nvrtc', 'cudart', 'cuda']),
]
inc_lst = [
  include_directories('include'),
  include_directories('.'),
]
lnk_lst = []
jit_lst = []
pyt_lst = []

subdir('src')
subdir('include')
subdir('python')
subdir('tools')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

lnk_lst += py.extension_module(
  'blade-python',
  sources: pyt_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  gnu_symbol_visibility: 'hidden',
)

lnk_lst += library(
  'blade',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lnk_lst,
  install: true,
  gnu_symbol_visibility: 'hidden',
)

lib_blade_dep = declare_dependency(
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lnk_lst,
)

subdir('tests')

# TODO: Add pkgconfig module.
