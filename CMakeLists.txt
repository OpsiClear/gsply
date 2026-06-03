cmake_minimum_required(VERSION 3.15)
project(gsply LANGUAGES NONE)

option(GSPLY_BUILD_CPP "Build the bundled gsply_cpp extension" OFF)

if(GSPLY_BUILD_CPP)
  set(GSPLY_CPP_BUILD_PYTHON ON CACHE BOOL "Build the Python nanobind extension" FORCE)
  add_subdirectory(cpp)
endif()
