cuda_deps = []
cuda_found = false

if blade_device_cuda_requested
    cuda_deps += dependency(
        'cuda',
        version: '>=11.7',
        modules: ['cuda', 'nvrtc', 'cufft'],
        required: not blade_devices_auto,
    )

    cuda_found = true
    foreach cuda_dep : cuda_deps
        cuda_found = cuda_found and cuda_dep.found()
    endforeach
endif

if cuda_found
    cfg_lst.set('BLADE_LOADER_CUDA_AVAILABLE', true)
    cfg_lst.set('BLADE_DEVICE_CUDA_AVAILABLE', true)
    dep_lst += cuda_deps
    blade_device_cuda_enabled = true
endif

ldr_lst += {'CUDA': cuda_found}
