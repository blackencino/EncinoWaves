#!/usr/bin/env python3

import numpy as np
from encino_waves import (
    OceanParameters, 
    create_ocean_plan, 
    compute_spectral_basis,
    compute_spectral_height, 
    compute_spatial_heights
)

def test_improved_parameters():
    """Test the improved parameters used in the visualizer."""
    print("=== Testing Improved Visualizer Parameters ===")
    
    # Use the same parameters as the updated visualizer
    ocean_params = OceanParameters(
        resolution=256,
        domain=1000.0,  # Larger domain for longer wavelengths
        gravity=9.81,
        surface_tension=0.074,
        density=1000.0,
        depth=200.0,
        wind_speed=25.0,  # Stronger wind for bigger waves
        fetch_km=1000.0,  # Longer fetch for more developed waves
        swell=0.0,
        random_seed=12345
    )
    
    print(f"Parameters: domain={ocean_params.domain}m, wind_speed={ocean_params.wind_speed}m/s, fetch_km={ocean_params.fetch_km}km")
    
    # Create plan and compute waves
    plan = create_ocean_plan(ocean_params)
    spectral_basis = compute_spectral_basis(plan)
    
    # Test at different times
    times = [0.0, 5.0, 10.0]
    wave_scale_factor = 100.0  # Same as visualizer
    
    for time in times:
        spectral_height = compute_spectral_height(plan, time, spectral_basis)
        spatial_heights = compute_spatial_heights(plan, spectral_height)
        
        # Show both raw and scaled amplitudes
        peak_to_peak_raw = spatial_heights.max() - spatial_heights.min()
        peak_to_peak_scaled = peak_to_peak_raw * wave_scale_factor
        
        print(f"\nTime {time}s:")
        print(f"  Raw wave height: {peak_to_peak_raw:.4f}m ({peak_to_peak_raw*100:.1f}cm)")
        print(f"  Scaled for visualization: {peak_to_peak_scaled:.1f}m")
        
        if peak_to_peak_raw > 0.02:
            print(f"  ✓ Good raw amplitude: {peak_to_peak_raw:.3f}m")
        elif peak_to_peak_raw > 0.005:
            print(f"  ⚠ Small but reasonable: {peak_to_peak_raw:.3f}m")
        else:
            print(f"  ✗ Still very small: {peak_to_peak_raw:.3f}m")

if __name__ == "__main__":
    test_improved_parameters() 