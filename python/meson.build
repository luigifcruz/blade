deps = [
  dependency('fmt', version: '>=6.1', required: false),
  dependency('pybind11', version: '>=2.4', required: false),
]

py = import('python').find_installation('python3', required: false)

deps += py.dependency()

has_python = py.found()
foreach x_dep : deps
  has_python = has_python and x_dep.found()
endforeach

if has_python
  py.extension_module(
    'blade',
    sources: 'bindings.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: lib_blade,
    install: true,
    install_dir: py.get_install_dir()
  )
  python_path = meson.current_build_dir()
endif
