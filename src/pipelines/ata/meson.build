sum_lst = {}

#
#  Mode A
#

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

if all_deps_found
    src_lst += files([
        'mode_a.cc',
    ])
endif

sum_lst += {'Mode A': all_deps_found}

cfg_lst.set('BLADE_PIPELINE_ATA_MODE_A', all_deps_found)

#
#  Mode B
#

deps = [
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_ATA_PHASOR',
    'BLADE_MODULE_ATA_BEAMFORMER',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
         'mode_b.cc',
    ])
endif

sum_lst += {'Mode B': all_deps_found}

cfg_lst.set('BLADE_PIPELINE_ATA_MODE_B', all_deps_found)

summary(sum_lst, section: 'ATA Pipelines', bool_yn: true)
