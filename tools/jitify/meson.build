jitify_comp = executable(
  'jitify-preprocess',
  'base.cc',
  include_directories: inc_lst,
  dependencies: dep_lst,
)

jitify_gen = generator(
  jitify_comp,
  output: '@BASENAME@.jit.hh',
  # TODO: Fix this hardcoded CUDA path.
  # TODO: Add -lineinfo only when debug enabled.
  arguments : ['-I/usr/local/cuda/include', '-I/opt/cuda/include', '-lineinfo',
      '-dlto', '-i', '--minify', '-o', '@OUTPUT@', '@INPUT@']
)

src_lst += jitify_gen.process(jit_lst)
