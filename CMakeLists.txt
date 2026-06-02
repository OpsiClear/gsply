cmake_minimum_required(VERSION 3.15)
project(gsply LANGUAGES NONE)

option(GSPLY_BUILD_CPP "Build the bundled gsply_cpp extension" OFF)

if(GSPLY_BUILD_CPP)
  add_subdirectory(cpp)
endif()
