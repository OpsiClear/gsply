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

# libdeflate for SPZ gzip decompression + CRC32 — fastest whole-buffer inflate,
# exactly the SPZ read use case.
include(FetchContent)
set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_GZIP OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  libdeflate
  GIT_REPOSITORY https://github.com/ebiggers/libdeflate.git
  GIT_TAG v1.23
)
FetchContent_MakeAvailable(libdeflate)

# zlib-ng (zlib-compatible API) for the SPZ *write* path: libdeflate has no
# flush API, but pigz-style parallel single-member gzip needs Z_SYNC_FLUSH to
# byte-align independently-compressed blocks. zlib-ng is a fast, portable zlib.
set(ZLIB_COMPAT ON CACHE BOOL "" FORCE)       # expose the classic zlib API/zlib.h
set(ZLIB_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ZLIBNG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  zlibng
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng.git
  GIT_TAG 2.2.2
)
FetchContent_MakeAvailable(zlibng)

# zstd for the NGSP v4 container (per-attribute streams). zstd's CMake lives in
# build/cmake within the release tarball, so point SOURCE_SUBDIR at it.
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  zstd
  URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
  SOURCE_SUBDIR build/cmake
)
FetchContent_MakeAvailable(zstd)

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
target_link_libraries(gsply_cpp PRIVATE libdeflate::libdeflate_static zlibstatic libzstd_static)
target_include_directories(gsply_cpp PRIVATE ${zstd_SOURCE_DIR}/lib)

# OpenMP to parallelize the per-Gaussian decode/encode loops (parity with numba prange).
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gsply_cpp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(gsply_cpp PRIVATE GSPLY_OPENMP=1)
endif()

install(TARGETS gsply_cpp LIBRARY DESTINATION .)
