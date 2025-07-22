#!/usr/bin/env python3
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

"""Simple test to verify the encino_waves library can be loaded and used."""

import ctypes
import numpy as np
from visualize import find_library, LIB, InitialStateParams, PropagationParams, InitialState

def test_library_loading():
    """Test that the library can be loaded."""
    print("Testing library loading...")
    try:
        lib_path = find_library("encino_waves")
        print(f"[OK] Library found at: {lib_path}")
    except FileNotFoundError as e:
        print(f"[FAIL] Library not found: {e}")
        return False
    
    # Test that we can access the functions
    try:
        assert hasattr(LIB, 'encino_waves_create_initial_state')
        assert hasattr(LIB, 'encino_waves_destroy_initial_state')
        assert hasattr(LIB, 'encino_waves_propagate')
        assert hasattr(LIB, 'encino_waves_shutdown')
        print("[OK] All expected functions are present")
    except AssertionError:
        print("[FAIL] Some functions are missing from the library")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of creating and using an initial state."""
    print("\nTesting basic functionality...")
    
    try:
        # Create initial state with minimal parameters
        params = InitialStateParams(
            resolution=64,  # Small for quick testing
            domain=50.0,
            random_seed=12345
        )
        
        state = InitialState(params)
        print(f"[OK] Created initial state with ID: {state.id.value}")
        
        # Test propagation
        prop_params = PropagationParams(time=0.0, pinch=0.75, amplitude_gain=1.0)
        height_field = np.zeros((64, 64), dtype=np.float32)
        
        state.propagate(prop_params, height_field)
        print(f"[OK] Propagation successful")
        print(f"  Height field range: [{height_field.min():.3f}, {height_field.max():.3f}]")
        
        # The destructor will be called automatically
        del state
        print("[OK] State cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("=== Encino Waves Library Test ===\n")
    
    if test_library_loading():
        test_basic_functionality()
    
    print("\n=== Test Complete ===") 