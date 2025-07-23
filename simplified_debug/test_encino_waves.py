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
        
        # Run pipeline with multiple threads
        plan2 = ew.create_ocean_plan(params)
        basis2 = ew.compute_spectral_basis(plan2, max_threads=8)
        height2 = ew.compute_spectral_height(plan2, 7.0, basis2, max_threads=8)
        
        # Results should be identical
        np.testing.assert_array_equal(basis1, basis2)
        np.testing.assert_array_equal(height1, height2)
    
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
        plan, basis, height = ew.simulate_ocean_surface(params, time=5.0)
        
        assert plan.N == 64
        assert basis.shape == (64, 33, 5)
        assert height.shape == (64, 33)  # Complex64 by default
        assert height.dtype == np.complex64
        
    def test_array_reuse_in_pipeline(self):
        """Test array reuse in functional pipeline."""
        params = ew.OceanParameters(resolution=64)
        
        # Pre-allocate arrays - test both formats
        basis_array = np.zeros((64, 33, 5), dtype=np.float32)
        height_complex = np.zeros((64, 33), dtype=np.complex64)
        height_float = np.zeros((64, 33, 2), dtype=np.float32)
        
        # Test with complex64 output
        plan1, basis1, height1 = ew.simulate_ocean_surface(
            params, 1.0, basis_array, height_complex
        )
        
        # Test with float32 output  
        plan2, basis2, height2 = ew.simulate_ocean_surface(
            params, 2.0, basis_array, height_float
        )
        
        # Arrays should be the same objects
        assert basis1 is basis_array
        assert basis2 is basis_array
        assert height1 is height_complex
        assert height2 is height_float
        
        # Heights should be different (different times)
        assert not np.allclose(height1, height2.view(dtype=np.complex64).reshape(64, 33))


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
