project(
    'blade',
    ['cpp'],
    version: '2.0.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

src_lst = []
dep_lst = [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cudart']),
    dependency('jetstream'),
]
inc_lst = [
    include_directories('include'),
]

subdir('src')

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

pkg = import('pkgconfig')
pkg.generate(
    lib_blade,
    libraries: dep_lst,
)
