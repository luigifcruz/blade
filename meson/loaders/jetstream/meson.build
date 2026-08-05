deps = [
    dependency(
        'jetstream',
        version: '>=' + minimum_jetstream_version,
        fallback: ['cyberether', 'libjetstream_dep'],
        required: true,
        default_options: [
            'tests=false',
            'examples=false',
            'remote=disabled',
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
