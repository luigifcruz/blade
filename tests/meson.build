catch2_dep = dependency(
    'catch2',
    version: '>=3',
    fallback: ['catch2', 'catch2_dep'],
    default_options: [
        'default_library=static',
        'tests=false',
        'cpp_std=c++20',
    ],
)

plugin_test_dependencies = [plugin_code_dep, catch2_dep]
plugin_test_link_args = ['-Wl,--no-as-needed']

plugin_test_main = static_library(
    'blade-test-main',
    'main.cc',
    dependencies: plugin_test_dependencies,
)

plugin_test_fs = import('fs')
plugin_test_targets = []

foreach test_source : plugin_test_lst
    test_component_path = plugin_test_fs.parent(test_source)
    test_domain = plugin_test_fs.name(
        plugin_test_fs.parent(test_component_path)
    )
    test_component = plugin_test_fs.name(test_component_path)
    test_filename = plugin_test_fs.name(test_source)

    if test_filename == 'module_tests.cc'
        test_kind = 'module'
    elif test_filename == 'block_tests.cc'
        test_kind = 'block'
    else
        error('Unsupported plugin test source: ' + test_filename)
    endif

    test_name = test_domain + '-' + test_component + '-' + test_kind
    test_target = executable(
        'blade-' + test_name + '-tests',
        test_source,
        dependencies: plugin_test_dependencies,
        link_with: plugin_test_main,
        link_args: plugin_test_link_args,
    )
    plugin_test_targets += [test_target]

    test(
        test_name,
        test_target,
        suite: 'blade',
        is_parallel: false,
        timeout: 180,
    )
endforeach

plugin_flowgraph_tests = executable(
    'blade-flowgraph-tests',
    'flowgraphs/base.cc',
    dependencies: plugin_test_dependencies,
    link_with: plugin_test_main,
    cpp_args: [
        '-DBLADE_TEST_FLOWGRAPH_DIR="@0@"'.format(
            meson.current_source_dir() / 'flowgraphs'
        ),
    ],
    link_args: plugin_test_link_args,
)
plugin_test_targets += [plugin_flowgraph_tests]

test(
    'blade-flowgraphs',
    plugin_flowgraph_tests,
    suite: 'blade',
    is_parallel: false,
    timeout: 0,
)

alias_target(
    'blade-tests',
    plugin_test_targets,
)
