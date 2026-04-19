add_executable(HpSharkFloatTest
    MainTestCuda.cpp
)

target_include_directories(HpSharkFloatTest PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(HpSharkFloatTest PRIVATE HpSharkFloatTestLib)
