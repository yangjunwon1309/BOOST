cmake_minimum_required(VERSION 3.16)
project(path_g1_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED IMPORTED_TARGET lcm)

set(UNITREE_SDK2_ROOT "" CACHE PATH "Unitree SDK2 source/install root")
if (UNITREE_SDK2_ROOT AND EXISTS "${UNITREE_SDK2_ROOT}/CMakeLists.txt")
  set(BUILD_EXAMPLES OFF CACHE BOOL "Disable external Unitree SDK examples" FORCE)
  add_subdirectory("${UNITREE_SDK2_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2" EXCLUDE_FROM_ALL)
else ()
  find_package(unitree_sdk2 CONFIG REQUIRED
    HINTS "${UNITREE_SDK2_ROOT}/lib/cmake/unitree_sdk2"
         "${UNITREE_SDK2_ROOT}/lib64/cmake/unitree_sdk2"
  )
endif ()

if (NOT TARGET unitree_sdk2)
  message(FATAL_ERROR "Unitree SDK2 did not provide the unitree_sdk2 target")
endif ()

find_package(Threads REQUIRED)
add_executable(g1_lowlevel_bridge src/g1_lowlevel_bridge.cpp)
target_include_directories(g1_lowlevel_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_definitions(g1_lowlevel_bridge PRIVATE G1_HAS_LCM=1)
target_link_libraries(g1_lowlevel_bridge PRIVATE
  PkgConfig::LCM
  unitree_sdk2
  Threads::Threads
)
target_compile_options(g1_lowlevel_bridge PRIVATE -Wall -Wextra -Wpedantic)
