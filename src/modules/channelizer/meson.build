deps = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'base.cc',
        'callback.cu',
    ])

    jit_lst += files([
        'channelizer.cu',
    ])
endif

sum_lst += {'Channelizer': all_deps_found}

cfg_lst.set('BLADE_MODULE_CHANNELIZER', all_deps_found)
