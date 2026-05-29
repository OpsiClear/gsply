cmake_minimum_required(VERSION 3.15)
project(gsply_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# nanobind needs Python's Development.Module component + the installed nanobind's
# CMake config (located via `python -m nanobind --cmake_dir`).
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# zlib for SPZ gzip (de)compression: prefer a system install, else fetch a known-
# good source from the official GitHub mirror (zlib.net tarballs 404 over time).
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  message(STATUS "ZLIB not found; fetching zlib 1.3.1 from github.com/madler/zlib")
  include(FetchContent)
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  )
  FetchContent_MakeAvailable(zlib)
  set(GSPLY_ZLIB_TARGET zlibstatic)
  # FetchContent zlib doesn't export an include dir on the target; add it.
  target_include_directories(zlibstatic PUBLIC
    "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
else()
  set(GSPLY_ZLIB_TARGET ZLIB::ZLIB)
endif()

nanobind_add_module(
  gsply_cpp
  STABLE_ABI      # cp312+ stable ABI wheels
  NB_STATIC       # bundle libnanobind statically
  src/bindings.cpp
  src/ply.cpp
  src/spz.cpp
)
target_include_directories(gsply_cpp PRIVATE src)
set_property(TARGET gsply_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gsply_cpp PRIVATE ${GSPLY_ZLIB_TARGET})

# OpenMP to parallelize the per-Gaussian decode/encode loops (parity with numba prange).
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gsply_cpp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(gsply_cpp PRIVATE GSPLY_OPENMP=1)
endif()

install(TARGETS gsply_cpp LIBRARY DESTINATION .)
