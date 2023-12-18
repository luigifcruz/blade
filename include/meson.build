configure_file(
    input: 'blade/config.hh.in',
    output: 'blade_config.hh',
    configuration: cfg_lst,
    install: not is_static,
    install_dir: get_option('includedir') + '/blade',
)

if not is_static
    install_subdir('blade', install_dir: get_option('includedir'))
endif