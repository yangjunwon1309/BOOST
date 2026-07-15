cmake_minimum_required(VERSION 3.16)

project(g1_agx_lcm_split VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_path(LCM_INCLUDE_DIR lcm/lcm.h)
find_library(LCM_LIBRARY lcm)

if(NOT LCM_INCLUDE_DIR OR NOT LCM_LIBRARY)
  message(FATAL_ERROR
      "LCM was not found. Install it with: sudo apt install liblcm-dev")
endif()

message(STATUS "LCM include: ${LCM_INCLUDE_DIR}")
message(STATUS "LCM library: ${LCM_LIBRARY}")

add_library(g1_lcm_split INTERFACE)
target_include_directories(g1_lcm_split INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/include"
  "${LCM_INCLUDE_DIR}"
)
target_link_libraries(g1_lcm_split INTERFACE
  "${LCM_LIBRARY}"
  Threads::Threads
)

add_executable(g1_lcm_state_listener examples/state_listener.cpp)
target_link_libraries(g1_lcm_state_listener PRIVATE g1_lcm_split)

add_executable(g1_lcm_action_smoke_sender examples/action_smoke_sender.cpp)
target_link_libraries(g1_lcm_action_smoke_sender PRIVATE g1_lcm_split)

add_executable(g1_lcm_bidirectional_smoke examples/bidirectional_smoke.cpp)
target_link_libraries(g1_lcm_bidirectional_smoke PRIVATE g1_lcm_split)

target_compile_options(g1_lcm_state_listener PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(g1_lcm_action_smoke_sender PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(g1_lcm_bidirectional_smoke PRIVATE -Wall -Wextra -Wpedantic)
