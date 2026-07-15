cmake_minimum_required(VERSION 3.20)
project(g1_ee_qp LANGUAGES CXX)

option(EE_QP_BUILD_TESTS "Build EE QP unit tests" ${PROJECT_IS_TOP_LEVEL})

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(proxsuite CONFIG QUIET)
if(NOT proxsuite_FOUND)
  file(GLOB _LOCAL_PROX_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python*/site-packages/cmeel.prefix")
  if(_LOCAL_PROX_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH ${_LOCAL_PROX_PREFIX})
  endif()
  find_package(proxsuite CONFIG REQUIRED)
endif()

add_library(g1_ee_qp STATIC
  src/ee_qp_corrector.cpp
  src/ee_reference_provider.cpp
  src/robot_fk_adapter.cpp
)
target_compile_features(g1_ee_qp PUBLIC cxx_std_20)
target_include_directories(g1_ee_qp PUBLIC include)
target_include_directories(g1_ee_qp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../g1_deploy_onnx_ref/include)
target_link_libraries(g1_ee_qp PUBLIC Eigen3::Eigen yaml-cpp proxsuite::proxsuite)

if(EE_QP_BUILD_TESTS)
  enable_testing()
  add_executable(test_ee_qp tests/test_ee_qp.cpp)
  target_link_libraries(test_ee_qp PRIVATE g1_ee_qp)
  add_test(NAME ee_qp_smoke COMMAND test_ee_qp)

  add_executable(test_robot_fk_adapter
    tests/test_robot_fk_adapter.cpp
    ../g1_deploy_onnx_ref/src/fk.cpp)
  target_include_directories(test_robot_fk_adapter PRIVATE
    ../g1_deploy_onnx_ref/include)
  target_link_libraries(test_robot_fk_adapter PRIVATE g1_ee_qp)
  target_compile_definitions(test_robot_fk_adapter PRIVATE
    G1_EE_QP_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../g1/g1_29dof.xml")
  add_test(NAME robot_fk_adapter_jacobian COMMAND test_robot_fk_adapter)
endif()
