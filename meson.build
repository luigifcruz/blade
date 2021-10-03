project(
  'bl',
  ['cpp','cuda'],
  version: '0.0.0',
  default_options: [
    'cpp_std=c++20',
    'b_ndebug=if-release',
    'b_lto=true',
  ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
  dependency('cuda', version : '>=11', modules : ['nvrtc', 'cudart', 'cuda']),
  dependency('spdlog'),
]
inc_lst = [
  include_directories('include'),
  include_directories('.'),
]
lnk_lst = []
jit_lst = []

py = import('python').find_installation('python3')

subdir('src')
subdir('include')
subdir('python')
subdir('tools')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

lnk_lst += library(
  'bl-beamformer',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lnk_lst,
  install: true,
  gnu_symbol_visibility: 'hidden',
)

libbl_dep = declare_dependency(
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lnk_lst,
)

subdir('tests')

executable(
  'bl-beamformer', 'main.cc',
  dependencies: libbl_dep,
  install: true,
)

