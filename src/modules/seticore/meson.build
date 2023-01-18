deps = [
    'BLADE_LOADER_SETICORE',
    'BLADE_LOADER_RADIOINTERFEROMETRY',
    'BLADE_LOADER_GUPPIRAW',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'dedoppler.cc',
        'hits_raw_writer.cc',
        'hits_stamp_writer.cc',
    ])

    jit_lst += files([
        'dedoppler.cu',
        'hits_writer.cu',
    ])
endif

sum_lst += {'Seticore': all_deps_found}

cfg_lst.set('BLADE_MODULE_SETICORE', all_deps_found)
