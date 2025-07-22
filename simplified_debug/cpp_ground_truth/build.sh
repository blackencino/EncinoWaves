#!/bin/bash

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

# Native build script for Encino Waves

set -e # Exit immediately if a command exits with a non-zero status.

# Create a build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake to configure the project, then build it
# Explicitly use g++ for better OpenMP support
if command -v ninja &> /dev/null; then
    cmake -DCMAKE_CXX_COMPILER=g++ -G Ninja ..
    ninja
else
    cmake -DCMAKE_CXX_COMPILER=g++ ..
    make -j$(nproc)
fi

# Copy the final shared library to the parent project directory
# This makes it easy for the Python script to find.
echo "Copying libencino_waves.so to project root..."
cp libencino_waves.so ../../