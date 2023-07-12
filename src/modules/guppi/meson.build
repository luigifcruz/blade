deps = [
    'BLADE_LOADER_RADIOINTERFEROMETRY',
    'BLADE_LOADER_GUPPIRAW',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'reader.cc',
        'writer.cc',
    ])

    jit_lst += files([
        'guppi.cu',
    ])
endif

sum_lst += {'[GEN] GUPPI': all_deps_found}

cfg_lst.set('BLADE_MODULE_GUPPI', all_deps_found)
