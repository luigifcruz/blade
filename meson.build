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

plugin_is_linux = target_machine.system().startswith('linux')

if not plugin_is_linux
    error('BLADE only supports Linux targets for now.')
endif

minimum_jetstream_version = '1.10.0'

#
# Load Python
#

pymod = import('python')
py = pymod.find_installation(required: true)

#
# Handle Options
#

cfg_lst = configuration_data()
src_lst = []
plugin_test_lst = []
dep_lst = []
inc_lst = [
    include_directories('include'),
]

plugin_requested_devices = get_option('devices')
if plugin_requested_devices.length() == 0
    error('The devices option must contain auto, cpu, or cuda.')
endif

plugin_devices_auto = 'auto' in plugin_requested_devices
if plugin_devices_auto and plugin_requested_devices.length() != 1
    error('The auto device cannot be combined with explicit devices.')
endif

plugin_device_cpu_enabled = plugin_devices_auto or 'cpu' in plugin_requested_devices
plugin_device_cuda_requested = plugin_devices_auto or 'cuda' in plugin_requested_devices
plugin_device_cuda_enabled = false

if plugin_device_cpu_enabled
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
subdir('include')
subdir('src')

#
# Library
#

plugin_library = static_library(
    'blade_code',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    gnu_symbol_visibility: 'hidden',
    pic: true,
    install: false,
)

plugin_code_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_whole: plugin_library,
)

plugin_module = shared_module(
    'blade',
    [],
    dependencies: plugin_code_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: false,
    install: false,
)

if get_option('tests')
    subdir('tests')
endif

#
# Packaging
#

plugin_examples = files(
    'examples/blade_phasor.yml',
    'examples/blade_stacker_integrator.yml',
)

plugin_package_device = plugin_device_cuda_enabled ? 'cuda' : 'cpu'

custom_target(
    'blade_cep',
    input: [plugin_module] + plugin_examples,
    output: 'blade.cep',
    command: [
        py,
        files('tools/bundler.py'),
        '--output', '@OUTPUT@',
        '--name', meson.project_name(),
        '--version', meson.project_version(),
        '--minimum-jetstream-version', minimum_jetstream_version,
        '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=' + plugin_package_device + ',arch=' + host_machine.cpu_family(),
        '--example', '@INPUT1@',
        '--example', '@INPUT2@',
    ],
    build_by_default: true,
    install: false,
)

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
    'Selection': ','.join(plugin_requested_devices),
    'CPU': plugin_device_cpu_enabled,
    'CUDA': plugin_device_cuda_enabled,
}, section: 'Devices', bool_yn: true)
