cfg_lst.set('version', meson.project_version())
cfg_lst.set('build_type', get_option('buildtype'))

ldr_lst = {}

subdir('radiointerferometry')
subdir('guppiraw')
subdir('bfr5')
subdir('filterbank')
subdir('cli11')

summary(ldr_lst, section: 'Dependency Loaders', bool_yn: true)
