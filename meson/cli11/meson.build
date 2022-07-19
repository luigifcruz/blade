deps = [
]

if subproject('CLI11').found()
    deps += subproject('CLI11').get_variable('CLI11_dep')
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_CLI11', true)
    dep_lst += deps
endif

ldr_lst += {'CLI11': all_deps_found}
