project(
    'blade',
    ['cpp'],
    version: '2.0.0',
    meson_version: '>=1.11.0',
    default_options: [
        'cpp_std=c++20',
        'warning_level=3',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

#
# Global Variables
#

blade_is_linux = target_machine.system().startswith('linux')

if not blade_is_linux
    error('BLADE only supports Linux targets for now.')
endif

minimum_jetstream_version = '1.8.0'

#
# Load Python
#

pymod = import('python')
py = pymod.find_installation(required: false)

#
# Handle Options
#

cfg_lst = configuration_data()
src_lst = []
dep_lst = []
inc_lst = [
    include_directories('include'),
]

blade_requested_devices = get_option('devices')
if blade_requested_devices.length() == 0
    error('The devices option must contain auto, cpu, or cuda.')
endif

blade_devices_auto = 'auto' in blade_requested_devices
if blade_devices_auto and blade_requested_devices.length() != 1
    error('The auto device cannot be combined with explicit devices.')
endif

blade_device_cpu_enabled = blade_devices_auto or 'cpu' in blade_requested_devices
blade_device_cuda_requested = blade_devices_auto or 'cuda' in blade_requested_devices
blade_device_cuda_enabled = false

if blade_device_cpu_enabled
    cfg_lst.set('BLADE_DEVICE_CPU_AVAILABLE', true)
endif

#
# Handle Configuration Data
#

cfg_lst.set('version', meson.project_version())
version_parts = meson.project_version().split('.')
cfg_lst.set('version_major', version_parts[0])
cfg_lst.set('version_minor', version_parts[1])
cfg_lst.set('version_patch', version_parts[2])

#
# Call Subdirectories
#

subdir('meson')
subdir('src')

#
# Library
#

lib_blade = library(
    'blade',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    gnu_symbol_visibility: 'hidden',
    link_args: ['-Wl,-Bsymbolic', '-Wl,--no-as-needed'],
    install: true,
)

lib_blade_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_blade,
)

if get_option('tests')
    subdir('tests')
endif

#
# Install Library
#

subdir('include')

pkg = import('pkgconfig')
pkg.generate(
    lib_blade,
    libraries: dep_lst,
)

#
# Packaging
#

blade_examples = files(
    'examples/blade_phasor.yml',
    'examples/blade_stacker_integrator.yml',
)

blade_package_device = cfg_lst.get('BLADE_LOADER_CUDA_AVAILABLE', false) ? 'cuda' : 'cpu'

if py.found()
    custom_target(
        'blade_cep',
        input: [lib_blade] + blade_examples,
        output: 'blade.cep',
        command: [
            py,
            files('tools/bundler.py'),
            '--output', '@OUTPUT@',
            '--name', meson.project_name(),
            '--version', meson.project_version(),
            '--minimum-jetstream-version', minimum_jetstream_version,
            '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=' + blade_package_device + ',arch=' + host_machine.cpu_family(),
            '--example', '@INPUT1@',
            '--example', '@INPUT2@',
        ],
        build_by_default: true,
        install: false,
    )
endif

summary({'CEP bundle': py.found()}, section: 'Packaging', bool_yn: true)

#
# Print Summary
#

summary({
    'cpp_std': get_option('cpp_std'),
    'prefix': get_option('prefix'),
    'buildtype': get_option('buildtype'),
}, section: 'General', bool_yn: true)

summary({
    'Selection': ','.join(blade_requested_devices),
    'CPU': blade_device_cpu_enabled,
    'CUDA': blade_device_cuda_enabled,
}, section: 'Devices', bool_yn: true)
