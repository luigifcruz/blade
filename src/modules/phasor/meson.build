deps = [
    'BLADE_LOADER_RADIOINTERFEROMETRY',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'ata.cc',
        'generic.cc',
        'vla.cc',
    ])

    jit_lst += files([
        'phasor.cu',
    ])
endif

sum_lst += {'Phasor ATA': all_deps_found}

cfg_lst.set('BLADE_MODULE_ATA_PHASOR', all_deps_found)
