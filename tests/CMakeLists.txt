set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_gate_search test_gate_search.cpp)
target_link_libraries(test_gate_search ${PROJECT_NAME})

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_gate_cos_navigate test_gate_cos_navigate.cpp)
target_link_libraries(test_gate_cos_navigate ${PROJECT_NAME} -ltcmalloc)