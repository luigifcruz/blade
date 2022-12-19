deps = [
]

if subproject('filterbankc99').found()
    deps += subproject('filterbankc99').get_variable('lib_filterbankc99_dep')
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_FILTERBANK', true)
    dep_lst += deps
endif

ldr_lst += {'Filterbank': all_deps_found}
