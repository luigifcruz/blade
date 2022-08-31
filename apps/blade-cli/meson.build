deps = [
    'BLADE_LOADER_CLI11',
    'BLADE_PIPELINE_GENERIC_FILE_READER'
]

inc_lst += [
    include_directories('include'),
]
src_lst = files([
    'main.cc',
])

subdir('src')

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'blade-cli', 
        src_lst,
        include_directories: inc_lst,
        dependencies: dep_lst + lib_blade_dep,
        install: true,
    )
endif

ldr_lst += {'CLI': all_deps_found}
