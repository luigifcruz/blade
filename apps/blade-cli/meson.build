deps = [
    dependency('cli11', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    executable(
        'blade-cli', 'main.cc',
        dependencies: dep_lst + cli_dep_lst,
        install: true,
    )
endif