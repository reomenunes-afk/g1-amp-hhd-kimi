cmake_minimum_required(VERSION 3.16)
project(orbbec_mujoco_raycaster_plugin LANGUAGES CXX)

if(NOT DEFINED RAYCASTER_SOURCE_DIR)
  message(FATAL_ERROR "RAYCASTER_SOURCE_DIR is required")
endif()
if(NOT DEFINED MUJOCO_INCLUDE_DIR)
  message(FATAL_ERROR "MUJOCO_INCLUDE_DIR is required")
endif()
if(NOT DEFINED MUJOCO_LIBRARY)
  message(FATAL_ERROR "MUJOCO_LIBRARY is required")
endif()
if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR is required")
endif()

file(GLOB RAYCASTER_CORE_SRC
  "${RAYCASTER_SOURCE_DIR}/raycaster_src/*.cpp"
)

add_library(sensor_raycaster SHARED
  "${RAYCASTER_SOURCE_DIR}/register.cc"
  "${RAYCASTER_SOURCE_DIR}/ray_plugin.cc"
  "${RAYCASTER_SOURCE_DIR}/ray_caster_plugin.cc"
  "${RAYCASTER_SOURCE_DIR}/ray_caster_camera_plugin.cc"
  "${RAYCASTER_SOURCE_DIR}/ray_caster_lidar_plugin.cc"
  ${RAYCASTER_CORE_SRC}
)

target_compile_features(sensor_raycaster PRIVATE cxx_std_17)
target_include_directories(sensor_raycaster PRIVATE
  "${RAYCASTER_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/shim_include"
  "${MUJOCO_INCLUDE_DIR}"
)
target_link_libraries(sensor_raycaster PRIVATE "${MUJOCO_LIBRARY}")
target_compile_options(sensor_raycaster PRIVATE -O3 -fPIC)
set_target_properties(sensor_raycaster PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
