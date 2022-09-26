sum_lst = {}

#
#  Mode B
#

deps = [
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_MEERKAT_BEAMFORMER',
    'BLADE_MODULE_DETECTOR',
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

cfg_lst.set('BLADE_PIPELINE_VLA_MODE_B', all_deps_found)

summary(sum_lst, section: 'VLA Pipelines', bool_yn: true)
