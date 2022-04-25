configure_file(
   input : 'blade/config.hh.in',
   output : 'blade_config.hh',
   configuration : cfg_lst,
   install : true,
   install_dir : get_option('includedir') + '/blade'
)

install_subdir('blade', install_dir : get_option('includedir'))
