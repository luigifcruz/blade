deps = [
]

if subproject('seticore').found()
    deps += subproject('seticore').get_variable('lib_seticore_dep')
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_SETICORE', true)
    dep_lst += deps
endif

ldr_lst += {'Seticore': all_deps_found}
