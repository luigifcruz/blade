jitify_comp = executable(
    'jitify-preprocess',
    'base.cc',
    include_directories: inc_lst,
    dependencies: dep_lst,
)

jitify_command_opts = [
    jitify_comp,
    '--std=c++17',
    '-I/usr/cuda/include',
    '-I/usr/local/cuda/include',
    '-I/opt/cuda/include',
    '-I' + join_paths(meson.source_root(), 'include'),
    '-i', '-o', '@OUTPUT@', '@INPUT@'
]

if get_option('buildtype') != 'release'
    jitify_command_opts += [
        '-lineinfo',
    ]
else
    jitify_command_opts += [
        '--minify',
    ]
endif

foreach jit_file: jit_lst
    jitify_gen = custom_target(
        output: '@BASENAME@.jit.hh',
        input: jit_file,
        command: jitify_command_opts,
    )

    src_lst += jitify_gen
endforeach