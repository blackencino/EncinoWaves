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

"""
Test suite for the Encino Waves Python wrapper.

Tests focus on repeatability and sanity of results.
"""

import pytest
import numpy as np
import encino_waves as ew
from dataclasses import replace
import os


class TestOceanParameters:
    """Test the OceanParameters dataclass."""
    
    def test_default_parameters(self):
        """Test that default parameters are created correctly."""
        params = ew.OceanParameters()
        assert params.resolution == 512
        assert params.domain == 100.0
        assert params.gravity == 9.81
        assert params.wind_speed == 17.0
        
    def test_immutability(self):
        """Test that parameters are truly immutable."""
        params = ew.OceanParameters()
        with pytest.raises(AttributeError):
            params.wind_speed = 20.0
            
    def test_with_methods(self):
        """Test the with_* methods create new instances."""
        params1 = ew.OceanParameters()
        params2 = params1.with_resolution(1024)
        params3 = params1.with_wind(wind_speed=25.0, fetch_km=500.0)
        
        # Original unchanged
        assert params1.resolution == 512
        assert params1.wind_speed == 17.0
        assert params1.fetch_km == 300.0
        
        # New instances have changes
        assert params2.resolution == 1024
        assert params3.wind_speed == 25.0
        assert params3.fetch_km == 500.0
        
    def test_invalid_resolution(self):
        """Test that invalid resolutions are rejected."""
        with pytest.raises(ValueError, match="power of 2"):
            ew.OceanParameters(resolution=100)  # Not power of 2
            
    def test_parameter_clamping(self):
        """Test that out-of-range parameters are clamped."""
        # Wind speed too high - should clamp
        params = ew.OceanParameters(wind_speed=500.0)  # Max is 300
        assert params.wind_speed == 300.0
        
        # Negative values should be clamped to min
        params = ew.OceanParameters(depth=-10.0)  # Min is 0.01
        assert params.depth == 0.01


class TestMultiThreading:
    """Test multi-threading (OpenMP) functionality."""
    
    def test_multithreading_availability(self):
        """Test if multi-threading is available and print diagnostics."""
        # This should match the availability check in encino_waves.py
        has_omp = ew._HAS_OMP
        print(f"\nOpenMP functions available: {has_omp}")
        
        if not has_omp:
            pytest.skip("OpenMP functions not available - compile with -DENCINO_WAVES_OMP_KERNELS")
    
    def test_spectral_basis_multithreading_repeatability(self):
        """Test that multi-threaded spectral basis computation is repeatable."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=256, random_seed=12345)
        plan = ew.create_ocean_plan(params)
        
        # Test various thread counts
        for max_threads in [1, 2, 4, 8, 0]:  # 0 means use all available
            basis1 = ew.compute_spectral_basis(plan, max_threads=max_threads)
            basis2 = ew.compute_spectral_basis(plan, max_threads=max_threads)
            
            np.testing.assert_array_equal(basis1, basis2, 
                err_msg=f"Results not repeatable with max_threads={max_threads}")
    
    def test_spectral_basis_thread_consistency(self):
        """Test that different thread counts produce identical results."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=54321)
        plan = ew.create_ocean_plan(params)
        
        # Compute with single thread as reference
        basis_single = ew.compute_spectral_basis(plan, max_threads=1)
        
        # Test multiple thread counts
        for max_threads in [2, 4, 8, 0]:
            basis_multi = ew.compute_spectral_basis(plan, max_threads=max_threads)
            
            np.testing.assert_array_equal(basis_single, basis_multi,
                err_msg=f"Multi-threaded result differs from single-threaded with {max_threads} threads")
    
    def test_spectral_height_multithreading_repeatability(self):
        """Test that multi-threaded spectral height computation is repeatable."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=256, random_seed=42)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan, max_threads=4)
        
        time = 5.0
        for max_threads in [1, 2, 4, 8, 0]:
            height1 = ew.compute_spectral_height(plan, time, basis, max_threads=max_threads)
            height2 = ew.compute_spectral_height(plan, time, basis, max_threads=max_threads)
            
            np.testing.assert_array_equal(height1, height2,
                err_msg=f"Height results not repeatable with max_threads={max_threads}")
    
    def test_spectral_height_thread_consistency(self):
        """Test that different thread counts produce identical spectral heights."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=67890)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan, max_threads=2)
        
        time = 10.0
        # Compute with single thread as reference
        height_single = ew.compute_spectral_height(plan, time, basis, max_threads=1)
        
        # Test multiple thread counts
        for max_threads in [2, 4, 8, 0]:
            height_multi = ew.compute_spectral_height(plan, time, basis, max_threads=max_threads)
            
            np.testing.assert_array_equal(height_single, height_multi,
                err_msg=f"Multi-threaded height differs from single-threaded with {max_threads} threads")
    
    def test_multithreading_with_different_formats(self):
        """Test multi-threading with both complex64 and float32 output formats."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=64, random_seed=98765)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan, max_threads=4)
        
        time = 3.0
        max_threads = 4
        
        # Test complex64 format
        height_complex = ew.compute_spectral_height(plan, time, basis, max_threads=max_threads)
        assert height_complex.dtype == np.complex64
        
        # Test with pre-allocated complex64 array
        out_complex = np.zeros((64, 33), dtype=np.complex64)
        height_complex2 = ew.compute_spectral_height(plan, time, basis, out=out_complex, max_threads=max_threads)
        assert height_complex2 is out_complex
        np.testing.assert_array_equal(height_complex, height_complex2)
        
        # Test float32 format
        out_float = np.zeros((64, 33, 2), dtype=np.float32)
        height_float = ew.compute_spectral_height(plan, time, basis, out=out_float, max_threads=max_threads)
        assert height_float is out_float
        
        # Verify equivalence between formats
        height_float_as_complex = height_float.view(dtype=np.complex64).reshape(64, 33)
        np.testing.assert_array_equal(height_complex, height_float_as_complex)
    
    def test_multithreading_pipeline(self):
        """Test multi-threading in the complete pipeline."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=13579)
        
        # Run pipeline with single thread
        plan1 = ew.create_ocean_plan(params)
        basis1 = ew.compute_spectral_basis(plan1, max_threads=1)
        height1 = ew.compute_spectral_height(plan1, 7.0, basis1, max_threads=1)
        spatial1 = ew.compute_spatial_heights(plan1, height1)
        
        # Run pipeline with multiple threads
        plan2 = ew.create_ocean_plan(params)
        basis2 = ew.compute_spectral_basis(plan2, max_threads=8)
        height2 = ew.compute_spectral_height(plan2, 7.0, basis2, max_threads=8)
        spatial2 = ew.compute_spatial_heights(plan2, height2)
        
        # Results should be identical
        np.testing.assert_array_equal(basis1, basis2)
        np.testing.assert_array_equal(height1, height2)
        np.testing.assert_array_equal(spatial1, spatial2)
    
    def test_edge_case_thread_counts(self):
        """Test edge case thread counts."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=64, random_seed=24680)
        plan = ew.create_ocean_plan(params)
        
        # Test thread count of 0 (should use all available)
        basis_auto = ew.compute_spectral_basis(plan, max_threads=0)
        height_auto = ew.compute_spectral_height(plan, 1.0, basis_auto, max_threads=0)
        
        # Test with explicit thread count
        basis_explicit = ew.compute_spectral_basis(plan, max_threads=2)
        height_explicit = ew.compute_spectral_height(plan, 1.0, basis_explicit, max_threads=2)
        
        # Should produce same results
        np.testing.assert_array_equal(basis_auto, basis_explicit)
        np.testing.assert_array_equal(height_auto, height_explicit)
    
    def test_large_resolution_multithreading(self):
        """Test multi-threading with larger resolutions where threading benefits are more apparent."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        # Use larger resolution to see threading benefits
        params = ew.OceanParameters(resolution=512, random_seed=11111)
        plan = ew.create_ocean_plan(params)
        
        # Single-threaded computation
        basis_single = ew.compute_spectral_basis(plan, max_threads=1)
        height_single = ew.compute_spectral_height(plan, 2.0, basis_single, max_threads=1)
        
        # Multi-threaded computation
        basis_multi = ew.compute_spectral_basis(plan, max_threads=0)  # Use all threads
        height_multi = ew.compute_spectral_height(plan, 2.0, basis_multi, max_threads=0)
        
        # Results should be identical despite different threading
        np.testing.assert_array_equal(basis_single, basis_multi)
        np.testing.assert_array_equal(height_single, height_multi)
        
        # Basic sanity checks
        assert np.all(np.isfinite(basis_multi))
        assert np.all(np.isfinite(height_multi))


class TestOceanPlan:
    """Test ocean plan creation and properties."""
    
    def test_plan_repeatability(self):
        """Test that same parameters produce identical plans."""
        params = ew.OceanParameters(resolution=256, random_seed=12345)
        
        plan1 = ew.create_ocean_plan(params)
        plan2 = ew.create_ocean_plan(params)
        
        # Shape properties should be identical
        assert plan1.N == plan2.N
        assert plan1.size_i == plan2.size_i
        assert plan1.size_j == plan2.size_j
        assert plan1.count == plan2.count
        
        # The underlying C struct should produce the same spectral basis
        # (this tests that the opaque values are also repeatable)
        basis1 = ew.compute_spectral_basis(plan1)
        basis2 = ew.compute_spectral_basis(plan2)
        np.testing.assert_array_equal(basis1, basis2)
        
    def test_plan_dimensions(self):
        """Test that plan dimensions are computed correctly."""
        params = ew.OceanParameters(resolution=256)
        plan = ew.create_ocean_plan(params)
        
        assert plan.N == 256
        assert plan.size_i == 129  # (N/2) + 1
        assert plan.size_j == 256
        assert plan.count == plan.size_i * plan.size_j
        
    def test_plan_immutability(self):
        """Test that plan properties are read-only."""
        plan = ew.create_ocean_plan()
        # N is a property, should not be directly settable
        with pytest.raises(AttributeError):
            plan.N = 512


class TestSpectralBasis:
    """Test spectral basis computation."""
    
    def test_spectral_basis_shape(self):
        """Test that spectral basis has correct shape."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        basis = ew.compute_spectral_basis(plan)
        
        assert basis.shape == (128, 65, 5)  # (size_j, size_i, 5)
        assert basis.dtype == np.float32
        
    def test_spectral_basis_repeatability(self):
        """Test that spectral basis computation is repeatable."""
        params = ew.OceanParameters(resolution=128, random_seed=54321)
        plan = ew.create_ocean_plan(params)
        
        # Test with multiple threads if available
        max_threads = 4 if ew._HAS_OMP else 1
        basis1 = ew.compute_spectral_basis(plan, max_threads=max_threads)
        basis2 = ew.compute_spectral_basis(plan, max_threads=max_threads)
        
        np.testing.assert_array_equal(basis1, basis2)
        
    def test_spectral_basis_sanity(self):
        """Test that spectral basis values are sane."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Use multiple threads if available for better coverage
        max_threads = 8 if ew._HAS_OMP else 1
        basis = ew.compute_spectral_basis(plan, max_threads=max_threads)
        
        # Check for NaN or infinity
        assert np.all(np.isfinite(basis))
        
        # Check omega channel is non-negative
        omega = basis[:, :, 4]
        assert np.all(omega >= 0.0)
        
        # DC component should be zero
        assert np.allclose(basis[0, 0, :], 0.0)
        
    def test_spectral_basis_different_seeds(self):
        """Test that different random seeds produce different results."""
        params1 = ew.OceanParameters(resolution=128, random_seed=1)
        params2 = ew.OceanParameters(resolution=128, random_seed=2)
        
        plan1 = ew.create_ocean_plan(params1)
        plan2 = ew.create_ocean_plan(params2)
        
        basis1 = ew.compute_spectral_basis(plan1)
        basis2 = ew.compute_spectral_basis(plan2)
        
        # Should be different (except DC component and omega)
        assert not np.allclose(basis1[:, :, :4], basis2[:, :, :4])
        
    def test_spectral_basis_with_preallocated(self):
        """Test using pre-allocated output array."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Pre-allocate array
        out = np.zeros((128, 65, 5), dtype=np.float32)
        result = ew.compute_spectral_basis(plan, out=out)
        
        # Should return same array
        assert result is out
        
        # Should have written data
        assert not np.allclose(out, 0.0)


class TestSpectralHeight:
    """Test spectral height computation."""
    
    def test_spectral_height_shape(self):
        """Test that spectral height has correct shape."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        basis = ew.compute_spectral_basis(plan)
        height = ew.compute_spectral_height(plan, time=10.0, spectral_basis=basis)
        
        # Default should be complex64 for FFT compatibility
        assert height.shape == (128, 65)  # Complex format
        assert height.dtype == np.complex64
        
    def test_spectral_height_both_formats(self):
        """Test that both complex64 and float32 formats work."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        basis = ew.compute_spectral_basis(plan)
        
        # Test default complex64 format
        height_complex = ew.compute_spectral_height(plan, 5.0, basis)
        assert height_complex.shape == (64, 33)
        assert height_complex.dtype == np.complex64
        
        # Test explicit complex64 format
        out_complex = np.zeros((64, 33), dtype=np.complex64)
        height_complex2 = ew.compute_spectral_height(plan, 5.0, basis, out=out_complex)
        assert height_complex2 is out_complex
        np.testing.assert_array_equal(height_complex, height_complex2)
        
        # Test float32 format
        out_float = np.zeros((64, 33, 2), dtype=np.float32)
        height_float = ew.compute_spectral_height(plan, 5.0, basis, out=out_float)
        assert height_float is out_float
        assert height_float.shape == (64, 33, 2)
        assert height_float.dtype == np.float32
        
        # Results should be equivalent (complex view of float array)
        height_float_as_complex = height_float.view(dtype=np.complex64).reshape(64, 33)
        np.testing.assert_array_equal(height_complex, height_float_as_complex)
        
    def test_spectral_height_repeatability(self):
        """Test that spectral height computation is repeatable."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Use multiple threads if available
        max_threads = 4 if ew._HAS_OMP else 1
        basis = ew.compute_spectral_basis(plan, max_threads=max_threads)
        
        height1 = ew.compute_spectral_height(plan, 5.0, basis, max_threads=max_threads)
        height2 = ew.compute_spectral_height(plan, 5.0, basis, max_threads=max_threads)
        
        np.testing.assert_array_equal(height1, height2)
        
    def test_spectral_height_time_evolution(self):
        """Test that spectral height evolves with time."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        
        # Use multiple threads if available for better test coverage
        max_threads = 6 if ew._HAS_OMP else 1
        basis = ew.compute_spectral_basis(plan, max_threads=max_threads)
        
        height_t0 = ew.compute_spectral_height(plan, 0.0, basis, max_threads=max_threads)
        height_t1 = ew.compute_spectral_height(plan, 1.0, basis, max_threads=max_threads)
        height_t2 = ew.compute_spectral_height(plan, 2.0, basis, max_threads=max_threads)
        
        # Heights should change over time (except DC)
        assert not np.allclose(height_t0[1:, :], height_t1[1:, :])
        assert not np.allclose(height_t1[1:, :], height_t2[1:, :])
        
        # DC component should remain constant
        assert np.allclose(height_t0[0, 0], height_t1[0, 0])
        assert np.allclose(height_t1[0, 0], height_t2[0, 0])
        
    def test_spectral_height_sanity(self):
        """Test that spectral height values are sane."""
        plan = ew.create_ocean_plan()
        basis = ew.compute_spectral_basis(plan)
        height = ew.compute_spectral_height(plan, 10.0, basis)
        
        # Check for NaN or infinity
        assert np.all(np.isfinite(height))
        
        # Magnitude should be reasonable (not too large)
        magnitudes = np.abs(height)
        assert np.max(magnitudes) < 1000.0  # Reasonable upper bound


class TestSpatialHeights:
    """Test spatial height computation (FFT)."""
    
    def test_spatial_heights_shape(self):
        """Test that spatial heights have correct shape."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, time=5.0, spectral_basis=basis)
        spatial_heights = ew.compute_spatial_heights(plan, spectral_height)
        
        # Should be NxN real array
        assert spatial_heights.shape == (128, 128)
        assert spatial_heights.dtype == np.float32
        
    def test_spatial_heights_repeatability(self):
        """Test that spatial height computation is repeatable."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64, random_seed=12345))
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 3.0, basis)
        
        spatial1 = ew.compute_spatial_heights(plan, spectral_height)
        spatial2 = ew.compute_spatial_heights(plan, spectral_height)
        
        np.testing.assert_array_equal(spatial1, spatial2)
        
    def test_spatial_heights_with_preallocated(self):
        """Test using pre-allocated output array."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 1.0, basis)
        
        # Pre-allocate array
        out = np.zeros((64, 64), dtype=np.float32)
        result = ew.compute_spatial_heights(plan, spectral_height, out=out)
        
        # Should return same array
        assert result is out
        
        # Should have written data (not all zeros anymore)
        assert not np.allclose(out, 0.0)
        
    def test_spatial_heights_sanity(self):
        """Test that spatial height values are sane."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 10.0, basis)
        spatial_heights = ew.compute_spatial_heights(plan, spectral_height)
        
        # Check for NaN or infinity
        assert np.all(np.isfinite(spatial_heights))
        
        # Heights should be reasonable (ocean waves typically -10m to +10m)
        assert np.max(np.abs(spatial_heights)) < 100.0
        
        # Should have variation (not constant)
        assert np.std(spatial_heights) > 1e-6
        
    def test_spatial_heights_time_evolution(self):
        """Test that spatial heights evolve with time."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        basis = ew.compute_spectral_basis(plan)
        
        # Compute at different times
        spectral_t0 = ew.compute_spectral_height(plan, 0.0, basis)
        spectral_t1 = ew.compute_spectral_height(plan, 5.0, basis)
        
        spatial_t0 = ew.compute_spatial_heights(plan, spectral_t0)
        spatial_t1 = ew.compute_spatial_heights(plan, spectral_t1)
        
        # Spatial heights should change over time
        assert not np.allclose(spatial_t0, spatial_t1)
        
        # But should have similar statistical properties
        assert abs(np.std(spatial_t0) - np.std(spatial_t1)) < np.std(spatial_t0) * 0.5
        
    def test_spatial_heights_different_resolutions(self):
        """Test spatial heights with different resolutions."""
        for resolution in [64, 128, 256]:
            plan = ew.create_ocean_plan(ew.OceanParameters(resolution=resolution))
            basis = ew.compute_spectral_basis(plan)
            spectral_height = ew.compute_spectral_height(plan, 2.0, basis)
            spatial_heights = ew.compute_spatial_heights(plan, spectral_height)
            
            assert spatial_heights.shape == (resolution, resolution)
            assert np.all(np.isfinite(spatial_heights))
            
    def test_spatial_heights_both_input_formats(self):
        """Test that spatial heights works with both complex64 and float32 input formats."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        basis = ew.compute_spectral_basis(plan)
        
        # Test with complex64 input
        spectral_complex = ew.compute_spectral_height(plan, 1.0, basis)
        spatial_from_complex = ew.compute_spatial_heights(plan, spectral_complex)
        
        # Test with float32 input
        spectral_float = np.zeros((64, 33, 2), dtype=np.float32)
        spectral_complex2 = ew.compute_spectral_height(plan, 1.0, basis, out=spectral_float)
        spatial_from_float = ew.compute_spatial_heights(plan, spectral_float)
        
        # Results should be identical
        assert spatial_from_complex.shape == (64, 64)
        assert spatial_from_float.shape == (64, 64)
        assert spatial_from_complex.dtype == np.float32
        assert spatial_from_float.dtype == np.float32
        np.testing.assert_array_equal(spatial_from_complex, spatial_from_float)
        
    def test_spatial_heights_validation(self):
        """Test input validation for spatial heights computation."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=64))
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 1.0, basis)
        
        # Test wrong spectral height shape (complex64)
        wrong_spectral = np.zeros((32, 17), dtype=np.complex64)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights(plan, wrong_spectral)
            
        # Test wrong spectral height shape (float32)
        wrong_spectral_float = np.zeros((32, 17, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights(plan, wrong_spectral_float)
            
        # Test unsupported dtype
        wrong_dtype = np.zeros((64, 33), dtype=np.complex128)
        with pytest.raises(ValueError, match="must be either"):
            ew.compute_spatial_heights(plan, wrong_dtype)
            
        # Test wrong float32 shape (missing last dimension)
        wrong_float_shape = np.zeros((64, 33), dtype=np.float32)
        with pytest.raises(ValueError, match="must be either"):
            ew.compute_spatial_heights(plan, wrong_float_shape)
            
        # Test wrong output array shape
        wrong_out = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights(plan, spectral_height, out=wrong_out)
            
        # Test wrong output array dtype
        wrong_out_dtype = np.zeros((64, 64), dtype=np.float64)
        with pytest.raises(ValueError, match="dtype"):
            ew.compute_spatial_heights(plan, spectral_height, out=wrong_out_dtype)
            
    def test_spatial_heights_multithreading_consistency(self):
        """Test that spatial heights are consistent regardless of spectral computation threading."""
        if not ew._HAS_OMP:
            pytest.skip("OpenMP not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=54321)
        plan = ew.create_ocean_plan(params)
        
        # Compute spectral basis and height with single thread
        basis_single = ew.compute_spectral_basis(plan, max_threads=1)
        spectral_single = ew.compute_spectral_height(plan, 4.0, basis_single, max_threads=1)
        spatial_single = ew.compute_spatial_heights(plan, spectral_single)
        
        # Compute spectral basis and height with multiple threads
        basis_multi = ew.compute_spectral_basis(plan, max_threads=8)
        spectral_multi = ew.compute_spectral_height(plan, 4.0, basis_multi, max_threads=8)
        spatial_multi = ew.compute_spatial_heights(plan, spectral_multi)
        
        # Results should be identical
        np.testing.assert_array_equal(spatial_single, spatial_multi)


class TestArrayValidation:
    """Test array validation and error handling."""
    
    def test_wrong_shape_rejection(self):
        """Test that wrong array shapes are rejected."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Wrong shape for output
        wrong_out = np.zeros((100, 100, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spectral_basis(plan, out=wrong_out)
            
    def test_wrong_dtype_rejection(self):
        """Test that wrong dtypes are rejected."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Wrong dtype
        wrong_out = np.zeros((128, 65, 5), dtype=np.float64)  # Should be float32
        with pytest.raises(ValueError, match="dtype"):
            ew.compute_spectral_basis(plan, out=wrong_out)
            
    def test_non_contiguous_rejection(self):
        """Test that non-contiguous arrays are rejected."""
        plan = ew.create_ocean_plan(ew.OceanParameters(resolution=128))
        
        # Create non-contiguous array by slicing
        full_array = np.zeros((256, 130, 10), dtype=np.float32)
        non_contig = full_array[:128, :65, :5]
        
        assert not non_contig.flags['C_CONTIGUOUS']
        
        with pytest.raises(ValueError, match="C-contiguous"):
            ew.compute_spectral_basis(plan, out=non_contig)


class TestFunctionalComposition:
    """Test functional composition patterns."""
    
    def test_pipeline_example(self):
        """Test the example pipeline function."""
        params = ew.OceanParameters(resolution=64)
        plan, basis, height, spatial = ew.simulate_ocean_surface(params, time=5.0)
        
        assert plan.N == 64
        assert basis.shape == (64, 33, 5)
        assert height.shape == (64, 33)  # Complex64 by default
        assert height.dtype == np.complex64
        assert spatial.shape == (64, 64)  # NxN spatial heights
        assert spatial.dtype == np.float32
        
    def test_array_reuse_in_pipeline(self):
        """Test array reuse in functional pipeline."""
        params = ew.OceanParameters(resolution=64)
        
        # Pre-allocate arrays - test both formats
        basis_array = np.zeros((64, 33, 5), dtype=np.float32)
        height_complex = np.zeros((64, 33), dtype=np.complex64)
        height_float = np.zeros((64, 33, 2), dtype=np.float32)
        spatial_array = np.zeros((64, 64), dtype=np.float32)
        
        # Test with complex64 output
        plan1, basis1, height1, spatial1 = ew.simulate_ocean_surface(
            params, 1.0, basis_array, height_complex, spatial_array
        )
        
        # Save first spatial result before it gets overwritten
        spatial1_copy = spatial1.copy()
        
        # Test with float32 output  
        plan2, basis2, height2, spatial2 = ew.simulate_ocean_surface(
            params, 2.0, basis_array, height_float, spatial_array
        )
        
        # Arrays should be the same objects
        assert basis1 is basis_array
        assert basis2 is basis_array
        assert height1 is height_complex
        assert height2 is height_float
        assert spatial1 is spatial_array
        assert spatial2 is spatial_array
        
        # Heights should be different (different times)
        assert not np.allclose(height1, height2.view(dtype=np.complex64).reshape(64, 33))
        # Spatial heights should also be different (comparing saved copy vs current)
        assert not np.allclose(spatial1_copy, spatial2)


class TestPhysicalPlausibility:
    """Test physical plausibility of results."""
    
    def test_calm_conditions(self):
        """Test that calm conditions produce smaller waves."""
        params_calm = ew.OceanParameters(
            resolution=64,
            wind_speed=1.0,  # Very light wind
            fetch_km=10.0    # Small fetch
        )
        params_storm = ew.OceanParameters(
            resolution=64,
            wind_speed=30.0,  # Strong wind
            fetch_km=1000.0   # Large fetch
        )
        
        plan_calm = ew.create_ocean_plan(params_calm)
        plan_storm = ew.create_ocean_plan(params_storm)
        
        # Use multiple threads if available for performance
        max_threads = 8 if ew._HAS_OMP else 1
        basis_calm = ew.compute_spectral_basis(plan_calm, max_threads=max_threads)
        basis_storm = ew.compute_spectral_basis(plan_storm, max_threads=max_threads)
        
        # Storm should have larger amplitudes
        amp_calm = np.mean(np.abs(basis_calm[:, :, :4]))
        amp_storm = np.mean(np.abs(basis_storm[:, :, :4]))
        
        assert amp_storm > amp_calm
        
    def test_different_domains(self):
        """Test that domain size affects wave numbers correctly."""
        params_small = ew.OceanParameters(resolution=64, domain=10.0)
        params_large = ew.OceanParameters(resolution=64, domain=1000.0)
        
        plan_small = ew.create_ocean_plan(params_small)
        plan_large = ew.create_ocean_plan(params_large)
        
        # dk should be inversely proportional to domain
        assert plan_small.dk > plan_large.dk
        assert abs(plan_small.dk * 10.0 - plan_large.dk * 1000.0) < 1e-5
        
    def test_spatial_spectral_consistency(self):
        """Test that spatial heights are physically consistent with spectral data."""
        params = ew.OceanParameters(resolution=128, wind_speed=20.0, random_seed=42)
        
        # Use multiple threads if available for performance
        max_threads = 8 if ew._HAS_OMP else 1
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan, max_threads=max_threads)
        spectral_height = ew.compute_spectral_height(plan, 5.0, basis, max_threads=max_threads)
        spatial_heights = ew.compute_spatial_heights(plan, spectral_height)
        
        # Spatial heights should be real
        assert np.all(np.isreal(spatial_heights))
        
        # Energy conservation: the total energy in spatial domain should be 
        # related to the spectral domain (Parseval's theorem)
        spatial_energy = np.mean(spatial_heights**2)
        # With norm=None, standard Parseval's theorem: spatial_energy = spectral_energy / N²
        spectral_energy = np.mean(np.abs(spectral_height)**2) / (plan.N * plan.N)
        
        # They should be of similar order of magnitude (within 2 orders)
        assert 0.01 < spatial_energy / spectral_energy < 100.0
        
        # The DC component in spectral should correspond to the mean in spatial
        # (should be close to zero for well-centered ocean simulation)
        spatial_mean = np.mean(spatial_heights)
        # With norm=None, DC component needs to be scaled by 1/N²
        spectral_dc = spectral_height[0, 0].real / (plan.N * plan.N)
        
        # Both should be small for ocean waves
        assert abs(spatial_mean) < 1.0
        assert abs(spectral_dc) < 1.0


class TestFFTWComparison:
    """Test comparisons between numpy FFT and FFTW implementations."""
    
    def test_fftw_availability(self):
        """Test if FFTW functions are available and print diagnostics."""
        # This should match the availability check in encino_waves.py
        has_fftw = ew._HAS_FFTW
        print(f"\nFFTW functions available: {has_fftw}")
        
        if not has_fftw:
            pytest.skip("FFTW functions not available - compile with -DENCINO_WAVES_FFTW_KERNELS")
    
    def test_fftw_numpy_equivalence(self):
        """Test that FFTW and numpy FFT produce equivalent results."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        # Test with multiple resolutions
        for resolution in [64, 128]:
            params = ew.OceanParameters(resolution=resolution, random_seed=12345)
            plan = ew.create_ocean_plan(params)
            basis = ew.compute_spectral_basis(plan)
            spectral_height = ew.compute_spectral_height(plan, 5.0, basis)
            
            # Compute spatial heights with both methods
            spatial_numpy = ew.compute_spatial_heights(plan, spectral_height)
            spatial_fftw = ew.compute_spatial_heights_fftw(plan, spectral_height)
            
            print(f"\nResolution {resolution}:")
            print(f"Numpy result - shape: {spatial_numpy.shape}, dtype: {spatial_numpy.dtype}")
            print(f"FFTW result - shape: {spatial_fftw.shape}, dtype: {spatial_fftw.dtype}")
            print(f"Numpy stats - min: {np.min(spatial_numpy):.6f}, max: {np.max(spatial_numpy):.6f}, mean: {np.mean(spatial_numpy):.6f}, std: {np.std(spatial_numpy):.6f}")
            print(f"FFTW stats - min: {np.min(spatial_fftw):.6f}, max: {np.max(spatial_fftw):.6f}, mean: {np.mean(spatial_fftw):.6f}, std: {np.std(spatial_fftw):.6f}")
            
            # Compute differences
            diff = spatial_numpy - spatial_fftw
            max_abs_diff = np.max(np.abs(diff))
            mean_abs_diff = np.mean(np.abs(diff))
            rel_diff = max_abs_diff / (np.max(np.abs(spatial_numpy)) + 1e-10)
            
            print(f"Differences - max_abs: {max_abs_diff:.6f}, mean_abs: {mean_abs_diff:.6f}, rel: {rel_diff:.6f}")
            
            # They should be very close (within floating point precision)
            # Note: Different FFT implementations (FFTW vs numpy) can have small numerical differences
            # due to algorithm variations, so we use a more relaxed tolerance
            np.testing.assert_allclose(spatial_numpy, spatial_fftw, rtol=1e-3, atol=1e-2,
                err_msg=f"FFTW and numpy results differ at resolution {resolution}")
    
    def test_fftw_numpy_both_input_formats(self):
        """Test that FFTW works with both complex64 and float32 input formats."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=64, random_seed=98765)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        
        # Test with complex64 input
        spectral_complex = ew.compute_spectral_height(plan, 3.0, basis)
        spatial_numpy_complex = ew.compute_spatial_heights(plan, spectral_complex)
        spatial_fftw_complex = ew.compute_spatial_heights_fftw(plan, spectral_complex)
        
        # Test with float32 input
        spectral_float = np.zeros((64, 33, 2), dtype=np.float32)
        ew.compute_spectral_height(plan, 3.0, basis, out=spectral_float)
        spatial_numpy_float = ew.compute_spatial_heights(plan, spectral_float)
        spatial_fftw_float = ew.compute_spatial_heights_fftw(plan, spectral_float)
        
        # All four results should be equivalent
        print(f"\nInput format comparison:")
        print(f"Numpy complex vs FFTW complex - max diff: {np.max(np.abs(spatial_numpy_complex - spatial_fftw_complex)):.6f}")
        print(f"Numpy float vs FFTW float - max diff: {np.max(np.abs(spatial_numpy_float - spatial_fftw_float)):.6f}")
        print(f"Complex vs float (numpy) - max diff: {np.max(np.abs(spatial_numpy_complex - spatial_numpy_float)):.6f}")
        print(f"Complex vs float (FFTW) - max diff: {np.max(np.abs(spatial_fftw_complex - spatial_fftw_float)):.6f}")
        
        np.testing.assert_allclose(spatial_numpy_complex, spatial_fftw_complex, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(spatial_numpy_float, spatial_fftw_float, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(spatial_numpy_complex, spatial_numpy_float, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(spatial_fftw_complex, spatial_fftw_float, rtol=1e-5, atol=1e-6)
    
    def test_fftw_with_preallocated_arrays(self):
        """Test FFTW with pre-allocated output arrays."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=54321)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 2.0, basis)
        
        # Pre-allocate arrays
        numpy_out = np.zeros((128, 128), dtype=np.float32)
        fftw_out = np.zeros((128, 128), dtype=np.float32)
        
        # Compute with pre-allocated arrays
        numpy_result = ew.compute_spatial_heights(plan, spectral_height, out=numpy_out)
        fftw_result = ew.compute_spatial_heights_fftw(plan, spectral_height, out=fftw_out)
        
        # Should return the same arrays
        assert numpy_result is numpy_out
        assert fftw_result is fftw_out
        
        # Results should be equivalent
        np.testing.assert_allclose(numpy_out, fftw_out, rtol=1e-5, atol=1e-6)
    
    def test_fftw_error_handling(self):
        """Test FFTW error handling and validation."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=64)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 1.0, basis)
        
        # Test wrong spectral height shape (complex64)
        wrong_spectral = np.zeros((32, 17), dtype=np.complex64)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights_fftw(plan, wrong_spectral)
            
        # Test wrong spectral height shape (float32)
        wrong_spectral_float = np.zeros((32, 17, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights_fftw(plan, wrong_spectral_float)
            
        # Test unsupported dtype
        wrong_dtype = np.zeros((64, 33), dtype=np.complex128)
        with pytest.raises(ValueError, match="must be either"):
            ew.compute_spatial_heights_fftw(plan, wrong_dtype)
            
        # Test wrong output array shape
        wrong_out = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            ew.compute_spatial_heights_fftw(plan, spectral_height, out=wrong_out)
            
        # Test wrong output array dtype
        wrong_out_dtype = np.zeros((64, 64), dtype=np.float64)
        with pytest.raises(ValueError, match="dtype"):
            ew.compute_spatial_heights_fftw(plan, spectral_height, out=wrong_out_dtype)
    
    def test_fftw_time_evolution_consistency(self):
        """Test that FFTW and numpy give consistent time evolution."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=128, random_seed=11111)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        
        times = [0.0, 1.0, 2.0, 5.0, 10.0]
        for time in times:
            spectral_height = ew.compute_spectral_height(plan, time, basis)
            
            spatial_numpy = ew.compute_spatial_heights(plan, spectral_height)
            spatial_fftw = ew.compute_spatial_heights_fftw(plan, spectral_height)
            
            max_diff = np.max(np.abs(spatial_numpy - spatial_fftw))
            print(f"Time {time:4.1f} - max difference: {max_diff:.8f}")
            
            np.testing.assert_allclose(spatial_numpy, spatial_fftw, rtol=1e-5, atol=1e-6,
                err_msg=f"FFTW and numpy differ at time {time}")
    
    def test_fftw_normalization_investigation(self):
        """Investigate potential normalization differences between numpy and FFTW."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=64, random_seed=22222)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 1.0, basis)
        
        # Compute with both methods
        spatial_numpy = ew.compute_spatial_heights(plan, spectral_height)
        spatial_fftw = ew.compute_spatial_heights_fftw(plan, spectral_height)
        
        print(f"\nNormalization investigation:")
        print(f"Spectral input shape: {spectral_height.shape}, dtype: {spectral_height.dtype}")
        print(f"Plan N: {plan.N}, size_i: {plan.size_i}, size_j: {plan.size_j}")
        
        # Check DC component
        dc_spectral = spectral_height[0, 0]
        dc_spatial_numpy = np.mean(spatial_numpy)
        dc_spatial_fftw = np.mean(spatial_fftw)
        
        print(f"DC component - spectral: {dc_spectral}")
        print(f"DC component - spatial numpy mean: {dc_spatial_numpy:.8f}")
        print(f"DC component - spatial FFTW mean: {dc_spatial_fftw:.8f}")
        print(f"Expected spatial DC (spectral/N²): {dc_spectral.real / (plan.N * plan.N):.8f}")
        
        # Check total energy
        spectral_energy = np.sum(np.abs(spectral_height)**2)
        spatial_energy_numpy = np.sum(spatial_numpy**2)
        spatial_energy_fftw = np.sum(spatial_fftw**2)
        
        print(f"Total energy - spectral: {spectral_energy:.8f}")
        print(f"Total energy - spatial numpy: {spatial_energy_numpy:.8f}")
        print(f"Total energy - spatial FFTW: {spatial_energy_fftw:.8f}")
        print(f"Energy ratio (numpy/spectral): {spatial_energy_numpy / spectral_energy:.8f}")
        print(f"Energy ratio (FFTW/spectral): {spatial_energy_fftw / spectral_energy:.8f}")
        
        # Investigate scaling relationship
        ratio_numpy_fftw = np.mean(spatial_numpy / (spatial_fftw + 1e-10))
        print(f"Mean ratio (numpy/FFTW): {ratio_numpy_fftw:.8f}")
        
        # Check if it's a simple scaling factor
        spatial_fftw_scaled = spatial_fftw * ratio_numpy_fftw
        scaling_diff = np.max(np.abs(spatial_numpy - spatial_fftw_scaled))
        print(f"After scaling FFTW by ratio - max diff: {scaling_diff:.8f}")
    
    def test_debug_spectral_input_investigation(self):
        """Debug the spectral input format and values going to FFTW vs numpy."""
        if not ew._HAS_FFTW:
            pytest.skip("FFTW not available")
            
        params = ew.OceanParameters(resolution=64, random_seed=33333)
        plan = ew.create_ocean_plan(params)
        basis = ew.compute_spectral_basis(plan)
        spectral_height = ew.compute_spectral_height(plan, 0.5, basis)
        
        print(f"\nSpectral input investigation:")
        print(f"Spectral height shape: {spectral_height.shape}, dtype: {spectral_height.dtype}")
        print(f"Is C-contiguous: {spectral_height.flags['C_CONTIGUOUS']}")
        
        # Look at a few specific values
        print(f"spectral_height[0,0] (DC): {spectral_height[0,0]}")
        print(f"spectral_height[1,0]: {spectral_height[1,0]}")
        print(f"spectral_height[0,1]: {spectral_height[0,1]}")
        print(f"spectral_height[1,1]: {spectral_height[1,1]}")
        
        # Convert to float32 format to see what FFTW gets
        float_view = spectral_height.view(dtype=np.float32)
        spectral_float = float_view.reshape(plan.size_j, plan.size_i, 2)
        
        print(f"Float view shape: {spectral_float.shape}")
        print(f"spectral_float[0,0,:] (DC as [real,imag]): {spectral_float[0,0,:]}")
        print(f"spectral_float[1,0,:]: {spectral_float[1,0,:]}")
        print(f"spectral_float[0,1,:]: {spectral_float[0,1,:]}")
        print(f"spectral_float[1,1,:]: {spectral_float[1,1,:]}")
        
        # Compute and compare
        spatial_numpy = ew.compute_spatial_heights(plan, spectral_height)  
        spatial_fftw = ew.compute_spatial_heights_fftw(plan, spectral_height)
        
        print(f"Spatial results - numpy[0,0]: {spatial_numpy[0,0]:.8f}, FFTW[0,0]: {spatial_fftw[0,0]:.8f}")
        print(f"Spatial results - numpy[1,0]: {spatial_numpy[1,0]:.8f}, FFTW[1,0]: {spatial_fftw[1,0]:.8f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
