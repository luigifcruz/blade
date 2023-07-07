jitify_comp = executable(
  'jitify-preprocess',
  'base.cc',
  include_directories: inc_lst,
  dependencies: dep_lst,
)

foreach jit_file: jit_lst
  jitify_gen = custom_target(
      output: '@BASENAME@.jit.hh',
      input : jit_file,
      install : true,
      install_dir : get_option('includedir') + '/blade/kernels',
      # TODO: Fix this hardcoded CUDA path.
      # TODO: Add -lineinfo only when debug enabled.
      command : [
        jitify_comp,
        '--std=c++20',
        '-I/usr/local/cuda/include',
        '-I/opt/cuda/include',
        '-I/home/luigi/sandbox/blade/include',  # TODO: Automatically figure out this path.
        '-lineinfo',
        '-i', '--minify', '-o', '@OUTPUT@', '@INPUT@'],
  )

  src_lst += jitify_gen
endforeach