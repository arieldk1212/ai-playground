file(GLOB TEST_SOURCES CONFIGURE_DEPENDS test_*.cc)

foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})

    target_link_libraries(${test_name} PRIVATE my_lib)

    target_include_directories(${test_name} PRIVATE include)
endforeach()

enable_testing()