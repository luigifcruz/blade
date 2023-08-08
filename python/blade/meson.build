py = import('python').find_installation('python3', required: false)

has_python = false
if py.found()
    has_python = true
endif

if has_python
    cmake = import('cmake')
    sub_proj = cmake.subproject('nanobind')

    deps = []
    deps += sub_proj.dependency('nanobind-static')
    deps += py.dependency()

    python_path = meson.current_build_dir()
    python_ext_path = join_paths(py.get_install_dir(), 'blade')

    if get_option('buildtype') == 'debugoptimized'
        run_command(
            find_program('bash'), 
            '-c',
            '@0@ -fs @1@/* @2@'.format(
                find_program('ln').full_path(), 
                meson.current_source_dir(), 
                meson.current_build_dir()
            ),
            check: true
        )
    endif

    py.extension_module(
        '_blade_mem_impl',
        sources: 'ext_mem.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )

if cfg_lst.get('BLADE_MODULE_CAST', false)
    py.extension_module(
        '_blade_cast_impl',
        sources: 'ext_cast.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_CHANNELIZER', false)
    py.extension_module(
        '_blade_channelizer_impl',
        sources: 'ext_channelizer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_POLARIZER', false)
    py.extension_module(
        '_blade_polarizer_impl',
        sources: 'ext_polarizer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_PERMUTATION', false)
    py.extension_module(
        '_blade_permutation_impl',
        sources: 'ext_permutation.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_GUPPI', false)
    py.extension_module(
        '_blade_guppi_impl',
        sources: 'ext_guppi.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_GATHER', false)
    py.extension_module(
        '_blade_gather_impl',
        sources: 'ext_gather.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_DETECTOR', false)
    py.extension_module(
        '_blade_detector_impl',
        sources: 'ext_detector.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_COPY', false)
    py.extension_module(
        '_blade_copy_impl',
        sources: 'ext_copy.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_BFR5', false)
    py.extension_module(
        '_blade_bfr5_impl',
        sources: 'ext_bfr5.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if cfg_lst.get('BLADE_MODULE_ATA_PHASOR', false)
    py.extension_module(
        '_blade_phasor_impl',
        sources: 'ext_phasor.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

if (cfg_lst.get('BLADE_MODULE_ATA_BEAMFORMER', false) or
    cfg_lst.get('BLADE_MODULE_VLA_BEAMFORMER', false) or
    cfg_lst.get('BLADE_MODULE_MEERKAT_BEAMFORMER', false))
    py.extension_module(
        '_blade_beamformer_impl',
        sources: 'ext_beamformer.cc',
        include_directories: inc_lst,
        dependencies: deps + dep_lst,
        link_with: lib_blade,
        install: true,
        install_dir: python_ext_path,
    )
endif

endif