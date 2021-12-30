deps = [
  dependency('pybind11', required: false),
]

py = import('python').find_installation('python3', required: false)

deps += py.dependency()

all_deps_found = py.found()
foreach x_dep : deps
  all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
  py.extension_module(
    'blade',
    sources: 'bindings.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py.get_install_dir()
  )
endif
