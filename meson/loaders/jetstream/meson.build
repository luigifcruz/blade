deps = [
    dependency(
        'jetstream',
        version: '>=' + minimum_jetstream_version,
        required: true,
        default_options: [
            'remote=enabled',
        ],
    ),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_JETSTREAM_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Jetstream': all_deps_found}
