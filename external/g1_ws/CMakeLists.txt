cmake_minimum_required(VERSION 3.16)

project(g1_deploy_real VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

find_path(LCM_INCLUDE_DIR lcm/lcm.h)
find_library(LCM_LIBRARY lcm)

if(LCM_INCLUDE_DIR AND LCM_LIBRARY)
  set(G1_HAS_LCM TRUE)
  message(STATUS "LCM found: ${LCM_LIBRARY}")
else()
  set(G1_HAS_LCM FALSE)
  message(STATUS "LCM not found. LCM action/state bridge will be disabled.")
endif()

set(DEFAULT_UNITREE_SDK2_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../thirdparty/unitree_sdk2")
set(UNITREE_SDK2_PATH
    "${DEFAULT_UNITREE_SDK2_PATH}"
    CACHE PATH "Path to unitree_sdk2 source tree")

if(NOT EXISTS "${UNITREE_SDK2_PATH}/CMakeLists.txt")
  message(FATAL_ERROR
      "unitree_sdk2 was not found at '${UNITREE_SDK2_PATH}'.\n"
      "Pass -DUNITREE_SDK2_PATH=/path/to/unitree_sdk2 or copy the SDK next to this package.")
endif()

set(BUILD_EXAMPLES OFF CACHE BOOL "Disable unitree_sdk2 examples" FORCE)
add_subdirectory("${UNITREE_SDK2_PATH}" "${CMAKE_BINARY_DIR}/unitree_sdk2")

add_executable(g1_lowlevel_bridge
  src/g1_lowlevel_bridge.cpp
)

target_include_directories(g1_lowlevel_bridge PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_link_libraries(g1_lowlevel_bridge PRIVATE
  unitree_sdk2
  Threads::Threads
)

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  message(STATUS "✅ OpenCV found — camera streaming enabled")
  target_compile_definitions(g1_lowlevel_bridge PRIVATE G1_HAS_OPENCV=1)
  target_include_directories(g1_lowlevel_bridge PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(g1_lowlevel_bridge PRIVATE ${OpenCV_LIBS})
else()
  message(STATUS "⚠️  OpenCV not found — camera streaming disabled")
endif()

if(G1_HAS_LCM)
  target_include_directories(g1_lowlevel_bridge PRIVATE "${LCM_INCLUDE_DIR}")
  target_link_libraries(g1_lowlevel_bridge PRIVATE "${LCM_LIBRARY}")
  target_compile_definitions(g1_lowlevel_bridge PRIVATE G1_HAS_LCM=1)
endif()

target_compile_options(g1_lowlevel_bridge PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

add_executable(lowcmd_monitor src/lowcmd_monitor.cpp)
target_include_directories(lowcmd_monitor PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(lowcmd_monitor PRIVATE unitree_sdk2 Threads::Threads)


add_executable(front_video_probe
  src/front_video_probe.cpp
)

target_link_libraries(front_video_probe PRIVATE
  unitree_sdk2
  Threads::Threads
)

add_executable(videohub_rpc_probe
  src/videohub_rpc_probe.cpp
)

target_link_libraries(videohub_rpc_probe PRIVATE
  unitree_sdk2
  Threads::Threads
)

enable_testing()

add_executable(videohub_lcm_adapter_test
  tests/videohub_lcm_adapter_test.cpp
)

target_include_directories(videohub_lcm_adapter_test PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

add_test(
  NAME videohub_lcm_adapter_test
  COMMAND videohub_lcm_adapter_test
)

if(OpenCV_FOUND)
  add_executable(jpeg_resize_test
    tests/jpeg_resize_test.cpp
  )

  target_include_directories(jpeg_resize_test PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(jpeg_resize_test PRIVATE
    ${OpenCV_LIBS}
  )

  add_test(
    NAME jpeg_resize_test
    COMMAND jpeg_resize_test
  )
endif()

if(OpenCV_FOUND AND G1_HAS_LCM)
  add_executable(g1_videohub_lcm_bridge
    src/g1_videohub_lcm_bridge.cpp
  )

  target_include_directories(g1_videohub_lcm_bridge PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${LCM_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(g1_videohub_lcm_bridge PRIVATE
    unitree_sdk2
    Threads::Threads
    "${LCM_LIBRARY}"
    ${OpenCV_LIBS}
  )

  target_compile_options(g1_videohub_lcm_bridge PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
endif()
