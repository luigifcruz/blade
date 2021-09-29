project(
  'bl',
  ['cpp','cuda'],
  version: '0.0.0',
  default_options: [
    'cpp_std=c++20',
  ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
  dependency('cuda', version : '>=11', modules : ['cufft', 'nvrtc', 'cudart', 'cuda']),
  dependency('spdlog'),
]
inc_lst = [
  include_directories('include'),
  include_directories('.'),
]
lnk_lst = []
jit_lst = []

subdir('src')
subdir('include')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

jitify_comp = executable(
  'jitify-preprocess',
  'tools/jitify.cc',
  include_directories: inc_lst,
  dependencies: dep_lst,
)

jitify_gen = generator(
  jitify_comp,
  output: '@BASENAME@.jit.hh',
  arguments : ['-I/opt/cuda/include', '-i', '--minify', '-o', '@OUTPUT@', '@INPUT@']
)

src_lst += jitify_gen.process(jit_lst)

lnk_lst += library(
  'bl-beamformer',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lnk_lst,
  install: true,
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
