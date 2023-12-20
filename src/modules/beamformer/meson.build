deps = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'ata.cc',
        'meerkat.cc',
        'generic.cc',
    ])

    jit_lst += files([
        'beamformer.cu',
    ])
endif

sum_lst += {'[ATA] Beamformer': all_deps_found}
sum_lst += {'[MKT] Beamformer': all_deps_found}

cfg_lst.set('BLADE_MODULE_ATA_BEAMFORMER', all_deps_found)
cfg_lst.set('BLADE_MODULE_MEERKAT_BEAMFORMER', all_deps_found)
