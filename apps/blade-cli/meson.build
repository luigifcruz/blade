deps = [
    subproject('CLI11').get_variable('CLI11_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    executable(
        'blade-cli', 'main.cc',
        include_directories: inc_lst,
        dependencies: dep_lst + deps,
        install: true,
    )
endif
