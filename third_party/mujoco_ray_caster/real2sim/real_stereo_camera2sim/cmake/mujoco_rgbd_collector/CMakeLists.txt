cmake_minimum_required(VERSION 3.16)
project(orbbec_mujoco_rgbd_collector LANGUAGES CXX)

if(NOT DEFINED RAYCASTER_SOURCE_DIR)
  message(FATAL_ERROR "RAYCASTER_SOURCE_DIR is required")
endif()
if(NOT DEFINED MUJOCO_INCLUDE_DIR)
  message(FATAL_ERROR "MUJOCO_INCLUDE_DIR is required")
endif()
if(NOT DEFINED MUJOCO_LIBRARY)
  message(FATAL_ERROR "MUJOCO_LIBRARY is required")
endif()
if(NOT DEFINED COLLECTOR_SOURCE)
  message(FATAL_ERROR "COLLECTOR_SOURCE is required")
endif()
if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR is required")
endif()
if(NOT DEFINED MUJOCO_LIBRARY_DIR)
  get_filename_component(MUJOCO_LIBRARY_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

set(RAYCASTER_CORE_SRC
  "${RAYCASTER_SOURCE_DIR}/raycaster_src/RayCaster.cpp"
  "${RAYCASTER_SOURCE_DIR}/raycaster_src/RayCasterCamera.cpp"
  "${RAYCASTER_SOURCE_DIR}/raycaster_src/RayCasterLidar.cpp"
)

add_executable(mujoco_rgbd_collector
  "${COLLECTOR_SOURCE}"
  ${RAYCASTER_CORE_SRC}
)

target_compile_features(mujoco_rgbd_collector PRIVATE cxx_std_17)
target_include_directories(mujoco_rgbd_collector PRIVATE
  "${RAYCASTER_SOURCE_DIR}"
  "${MUJOCO_INCLUDE_DIR}"
  "${OpenCV_INCLUDE_DIRS}"
  "${GLFW3_INCLUDE_DIRS}"
)
target_link_directories(mujoco_rgbd_collector PRIVATE ${GLFW3_LIBRARY_DIRS})
target_link_libraries(mujoco_rgbd_collector PRIVATE
  "${MUJOCO_LIBRARY}"
  ${OpenCV_LIBS}
  ${GLFW3_LIBRARIES}
)
target_compile_options(mujoco_rgbd_collector PRIVATE -O3)
set_target_properties(mujoco_rgbd_collector PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  BUILD_RPATH "$ORIGIN:${MUJOCO_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN:${MUJOCO_LIBRARY_DIR}"
)
