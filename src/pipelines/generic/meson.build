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

if all_deps_found
    src_lst += files([
        'file_reader.cc',
    ])
endif

sum_lst += {'File Reader': all_deps_found}

cfg_lst.set('BLADE_PIPELINE_GENERIC_FILE_READER', all_deps_found)

#
#  Accumulate
#

deps = [
    'BLADE_MODULE_GUPPI',
    'BLADE_MODULE_FILTERBANK',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'accumulate.cc',
    ])
endif

sum_lst += {'Accumulate': all_deps_found}

cfg_lst.set('BLADE_PIPELINE_GENERIC_ACCUMULATE', all_deps_found)

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

if all_deps_found
    src_lst += files([
        'mode_h.cc',
    ])
endif

sum_lst += {'Mode H': all_deps_found}

cfg_lst.set('BLADE_PIPELINE_GENERIC_MODE_H', all_deps_found)

summary(sum_lst, section: 'Generic Pipelines', bool_yn: true)
