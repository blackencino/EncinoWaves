##-*****************************************************************************
## Copyright 2015 Christopher Jon Horvath
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##-*****************************************************************************

cmake_minimum_required(VERSION 3.11.0)

project(encino_waves LANGUAGES CXX)

option(ENCINO_WAVES_DO_TESTS "Do unit tests" ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
    include("cmake/conan.cmake")
    message(STATUS "about to do conan cmake run")
    conan_cmake_run(CONANFILE conanfile.txt
                    BASIC_SETUP
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
endif()

if (MSVC) 
    add_compile_options("/wd4304" "/wd4267" "/wd4244")
endif()

find_package(FFTW3f REQUIRED)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(Alembic REQUIRED)

include("cmake/encino_waves_utils.cmake")

add_subdirectory(src)
