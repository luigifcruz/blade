dep_lst += [
  dependency('pybind11'),
  py.dependency(embed: true)
]

python_src_lst = []

subdir('ata')
#subdir('meerkat')

lnk_lst += [
  py.extension_module(
    'blade-python',
    sources: python_src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    gnu_symbol_visibility: 'hidden',
  )
]
