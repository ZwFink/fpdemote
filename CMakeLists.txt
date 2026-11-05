cmake_minimum_required(VERSION 3.20)
project(fpdemote LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point LLVM_DIR at the LLVM that Proteus was built with, e.g.
# /opt/rocm-6.4.0/llvm/lib/cmake/llvm. The plugin must use the same LLVM
# version and headers as Proteus.
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

add_library(DemoteFP MODULE DemoteFP.cpp)
target_include_directories(DemoteFP SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(_llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(DemoteFP PRIVATE ${_llvm_definitions})

# A static-LLVM Proteus hides its LLVM symbols, so the plugin links its own
# copy and exports only the plugin entry point.
if(LLVM_LINK_LLVM_DYLIB)
  set(_llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(_llvm_libs Core Passes Support)
endif()
target_link_libraries(DemoteFP PRIVATE ${_llvm_libs})
target_link_options(DemoteFP PRIVATE
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(DemoteFP PRIVATE -fno-rtti)
endif()

install(TARGETS DemoteFP LIBRARY DESTINATION lib)
