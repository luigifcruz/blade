project(
    'blade',
    ['cpp', 'c', 'cuda'],
    version: '1.0.0',
    default_options: [
        'buildtype=release',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ]
)

cuda_args = ['-arch', 'compute_70', '--expt-relaxed-constexpr', '-diag-suppress', '815']

cfg_lst = configuration_data(
    {
        'commit': run_command('git', 'describe', '--always', '--dirty', '--abbrev=9', check: true).stdout().strip()
    }
)
src_lst = []
dep_lst = [
    dependency('cuda', version : '>=11', modules : ['nvrtc', 'cudart', 'cuda', 'culibos']),
    dependency('fmt'),
]
static_dep_lst = [
    dependency('cuda', version : '>=11', modules : ['cufft_static']),
]
inc_lst = [
    include_directories('include'),
    include_directories('kernels'),
    include_directories('.'),
]
jit_lst = []

subdir('meson')
subdir('src')
subdir('include')
subdir('kernels')

lib_blade = library(
    'blade',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst + static_dep_lst,
    gnu_symbol_visibility: 'hidden',
    cuda_args: ['-dc'],
    install: true,
)

lib_blade_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_blade,
)

pkg = import('pkgconfig')
pkg.generate(
    lib_blade,
    libraries: dep_lst,
)

subdir('python')
subdir('docs')
subdir('tests')
subdir('benchmarks')
subdir('examples')

summary({'cpp_std': get_option('cpp_std'),
            'prefix': get_option('prefix'),
            'buildtype': get_option('buildtype'),
            'python bindings': has_python,
        }, section: 'General', bool_yn: true)