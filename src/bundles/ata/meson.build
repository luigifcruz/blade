sum_lst = {}

deps = [
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_ATA_PHASOR',
    'BLADE_MODULE_ATA_BEAMFORMER',
    'BLADE_MODULE_DETECTOR',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'Mode B': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_ATA_MODE_B', all_deps_found)

summary(sum_lst, section: 'ATA Bundles', bool_yn: true)
