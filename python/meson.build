subdir('blade')

if has_python
    install_subdir(
        'blade',
        install_dir: py.get_install_dir(),
        exclude_files: [
            'ext_beamformer.cc',
            'ext_bfr5.cc',
            'ext_cast.cc',
            'ext_channelizer.cc',
            'ext_const.cc',
            'ext_copy.cc',
            'ext_duplicate.cc',
            'ext_detector.cc',
            'ext_gather.cc',
            'ext_guppi.cc',
            'ext_hidden.cc',
            'ext_mem.cc',
            'ext_modeb.cc',
            'ext_modeh.cc',
            'ext_permutation.cc',
            'ext_phasor.cc',
            'ext_pipeline.cc',
            'ext_polarizer.cc',
            'ext_runner.cc',
        ]
    )
endif
