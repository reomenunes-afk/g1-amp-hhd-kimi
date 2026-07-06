cmake_minimum_required(VERSION 3.14)

project(bitbot)

set(BITBOT_DEPENDENCY_USE_PROXY OFF)
set(BITBOT_DEPENDENCY_USE_LOCAL_FILE TRUE)
set(BITBOT_DEPENDENCY_LOCAL_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_subdirectory(CtrlZ)
include(bitbot.cmake)

option(BUILD_SIMULATION "Build simulation" ON)

if(BUILD_SIMULATION)
  add_definitions(-DBUILD_SIMULATION)
  BitbotAddDependency(
    NAME bitbot_mujoco
    FILE_NAME "bitbot_mujoco-main.zip"
    FILE_PATH ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
    USE_LOCAL_FILE TRUE
  )
else()
  add_subdirectory(Bitbot_Unitree)
  BitbotAddDependency(
    NAME bitbot_kernel
    FILE_NAME "bitbot_kernel-main.zip"
    FILE_PATH ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
    USE_LOCAL_FILE TRUE
  )
endif()


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
message("PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}")

add_executable(main_app
  main.cpp
  types.hpp
  user_func.h
  user_func.cpp
  user_command.h
  user_command.cpp
)

if(BUILD_SIMULATION)
  set(RAYCASTER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mujoco_ray_caster)
  set(RAYCASTER_SHIM_INCLUDE ${CMAKE_BINARY_DIR}/raycaster_shim_include)

  file(MAKE_DIRECTORY
    ${RAYCASTER_SHIM_INCLUDE}/engine
    ${RAYCASTER_SHIM_INCLUDE}/mujoco
  )

  file(TOUCH
    ${RAYCASTER_SHIM_INCLUDE}/engine/engine_name.h
    ${RAYCASTER_SHIM_INCLUDE}/engine/engine_util_errmem.h
    ${RAYCASTER_SHIM_INCLUDE}/engine/engine_util_spatial.h
  )

  file(WRITE ${RAYCASTER_SHIM_INCLUDE}/mujoco/mjthread.h
  "#pragma once

  struct mjThreadPool {};

  struct mjTask {
    void* (*func)(void*) = nullptr;
    void* args = nullptr;
  };

  static inline void mju_defaultTask(mjTask* task) {
    if (task) {
      task->func = nullptr;
      task->args = nullptr;
    }
  }

  static inline mjThreadPool* mju_threadPoolCreate(int) {
    return nullptr;
  }

  static inline void mju_threadPoolDestroy(mjThreadPool*) {}

  static inline void mju_threadPoolEnqueue(mjThreadPool*, mjTask* task) {
    if (task && task->func) {
      task->func(task->args);
    }
  }

  static inline void mju_taskJoin(mjTask*) {}
  ")

  file(WRITE ${RAYCASTER_SHIM_INCLUDE}/mujoco_237_compat.h
  [=[
#pragma once

#include <mujoco/mujoco.h>

#ifndef mj_markStack
#define mj_markStack(d) int _raycaster_mj_stack_mark = (d)->pstack
#endif

#ifndef mj_freeStack
#define mj_freeStack(d) ((d)->pstack = _raycaster_mj_stack_mark)
#endif

static inline void mju_mulMatVec3(mjtNum* res, const mjtNum* mat, const mjtNum* vec) {
  mju_mulMatVec(res, mat, vec, 3, 3);
}
]=])

  file(GLOB RAYCASTER_CORE_SRC
    ${RAYCASTER_ROOT}/raycaster_src/*.cpp
  )

  add_library(sensor_raycaster SHARED
    ${RAYCASTER_ROOT}/register.cc
    ${RAYCASTER_ROOT}/ray_plugin.cc
    ${RAYCASTER_ROOT}/ray_caster_plugin.cc
    ${RAYCASTER_ROOT}/ray_caster_camera_plugin.cc
    ${RAYCASTER_ROOT}/ray_caster_lidar_plugin.cc
    ${RAYCASTER_CORE_SRC}
  )

  target_include_directories(sensor_raycaster PRIVATE
    ${RAYCASTER_ROOT}
    ${RAYCASTER_ROOT}/raycaster_src
    ${RAYCASTER_SHIM_INCLUDE}
  )

  target_compile_options(sensor_raycaster PRIVATE
    -include ${RAYCASTER_SHIM_INCLUDE}/mujoco_237_compat.h
  )

  target_link_libraries(sensor_raycaster PRIVATE mujoco)

  set_target_properties(sensor_raycaster PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/mujoco_plugin
  )

  add_dependencies(main_app sensor_raycaster)
endif()

if(BUILD_SIMULATION)
  target_link_libraries(main_app BitbotMujoco)
else()
  target_link_libraries(main_app bitbotUnitree)
endif()

target_link_libraries(main_app
  BitbotKernel
  CtrlZ
)

if(BUILD_SIMULATION)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

  target_sources(main_app PRIVATE
    mujoco_camera/src/mujoco_camera.cpp
    mujoco_camera/include/mujoco_camera.h
  )
  target_include_directories(main_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_camera/include
  )
  target_link_libraries(main_app ${OpenCV_LIBS})
endif()

target_compile_options(main_app PUBLIC)
target_compile_features(main_app PUBLIC cxx_std_20)
