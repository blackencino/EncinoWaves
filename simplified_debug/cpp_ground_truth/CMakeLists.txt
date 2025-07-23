# Copyright 2015-2025 Christopher Jon Horvath
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
project(EncinoWaves CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP for parallel sections
find_package(OpenMP REQUIRED)

# Get CONDA_PREFIX from environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using CONDA_PREFIX: ${CONDA_PREFIX}")
    
    # Add conda include and library directories
    include_directories(${CONDA_PREFIX}/include)
    link_directories(${CONDA_PREFIX}/lib)
else()
    message(WARNING "CONDA_PREFIX not set - FFTW3 may not be found correctly")
endif()

# Define the shared library target
add_library(encino_waves SHARED generator.cpp)

# Improve visibility for exported symbols on Linux/macOS
set_target_properties(encino_waves PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Link against the required libraries
# Directly specify FFTW3 libraries (float version with OpenMP support)
target_link_libraries(encino_waves PRIVATE
    fftw3f          # Single precision FFTW3
    fftw3f_omp      # Single precision FFTW3 with OpenMP
    OpenMP::OpenMP_CXX
    m               # Math library
)