#
#  Mode H
#

deps = [
    'BLADE_MODULE_CAST',
    'BLADE_MODULE_CHANNELIZER',
    'BLADE_MODULE_DETECTOR'
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'[GEN] Mode H': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_MODE_H', all_deps_found)