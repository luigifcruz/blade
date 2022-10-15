deps = [
]

if subproject('guppirawc99').found()
    deps += subproject('guppirawc99').get_variable('lib_guppiraw_dep')
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_GUPPIRAW', true)
    dep_lst += deps
endif

ldr_lst += {'GUPPI RAW': all_deps_found}
