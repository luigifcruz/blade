catch2_dep = dependency(
    'catch2',
    version: '>=3',
    fallback: ['catch2', 'catch2_dep'],
)

plugin_flowgraph_tests = executable(
    'blade-flowgraph-tests',
    [
        'main.cc',
        'flowgraphs.cc',
    ],
    dependencies: [plugin_code_dep, catch2_dep],
    cpp_args: [
        '-DBLADE_TEST_FLOWGRAPH_DIR="@0@"'.format(
            meson.current_source_dir() / 'flowgraphs'
        ),
    ],
    link_args: ['-Wl,--no-as-needed'],
)

test(
    'blade-flowgraphs',
    plugin_flowgraph_tests,
    suite: 'blade',
    is_parallel: false,
    timeout: 0,
)
