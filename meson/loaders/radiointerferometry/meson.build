deps = [
    subproject(
        'radiointerferometryc99',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
        ],
    ).get_variable('lib_radiointerferometry_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_RADIOINTERFEROMETRY_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Radio Interferometry': all_deps_found}
