add_executable(HpSharkInstantiate)

target_sources(HpSharkInstantiate PRIVATE
    HpSharkInstantiate.cpp
)

target_include_directories(HpSharkInstantiate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
