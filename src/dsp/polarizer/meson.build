src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
])

if blade_device_cpu_enabled
    src_lst += files(['module_impl_native_cpu.cc'])
endif

if blade_device_cuda_enabled
    src_lst += files(['module_impl_native_cuda.cc'])
endif
