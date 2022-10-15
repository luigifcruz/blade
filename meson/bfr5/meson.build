deps = [
]

if subproject('bfr5c99').found()
    deps += subproject('bfr5c99').get_variable('lib_bfr5c99_dep')
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_BFR5', true)
    dep_lst += deps
endif

ldr_lst += {'BFR5': all_deps_found}
