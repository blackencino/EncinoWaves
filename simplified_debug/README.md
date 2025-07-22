<!--
Copyright 2015-2025 Christopher Jon Horvath

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Encino Waves - Simplified Debug

This is a simplified, self-contained version of the Encino Waves library with a Python visualization interface.

## Prerequisites

Install the required system dependencies:

```bash
sudo apt update
sudo apt install cmake libfftw3-dev build-essential
```

Install Python dependencies:

```bash
pip install numpy polyscope
```

## Building

Build the C++ shared library:

```bash
cd cpp_ground_truth
./build.sh
```

This will create `libencino_waves.so` in the `simplified_debug` directory.

## Running

### Run Tests

Test that the library loads and works correctly:

```bash
python3 test_library.py
```

### Run Visualizer

Launch the interactive wave visualization:

```bash
python3 visualize.py
```

## File Structure

- `cpp_ground_truth/` - C++ source code and build files
  - `generator.cpp` - Main wave generation implementation
  - `CMakeLists.txt` - CMake build configuration
  - `build.sh` - Native build script
- `visualize.py` - Interactive Polyscope-based visualizer
- `test_library.py` - Basic functionality tests
- `verify_setup.sh` - Setup verification script 