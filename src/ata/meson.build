pymod = import('python')
py = pymod.find_installation('python3')

dep_lst += [
  dependency('pybind11'),
  py.dependency(embed: true)
]

ata_src_lst = []

subdir('beamformer')

lnk_lst += [
  py.extension_module(
    'ata',
    sources: ata_src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
  )
]
