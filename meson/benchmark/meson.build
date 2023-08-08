deps = [
    dependency('benchmark', required: false)
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_BENCHMARK', true)
    dep_lst += deps
endif

ldr_lst += {'Google Benchmark': all_deps_found}
