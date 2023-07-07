sum_lst = {}

#
#  File Reader 
#

deps = [
    'BLADE_MODULE_GUPPI',
    'BLADE_MODULE_BFR5',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'File Reader': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_FILE_READER', all_deps_found)

#
#  File Writer
#

deps = [
    'BLADE_MODULE_GUPPI',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'File Writer': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_FILE_WRITER', all_deps_found)

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

sum_lst += {'Mode H': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_GENERIC_MODE_H', all_deps_found)

summary(sum_lst, section: 'Generic Bundles', bool_yn: true)
