#!/usr/bin/env python3

import numpy as np
from encino_waves import (
    OceanParameters, 
    create_ocean_plan, 
    compute_spectral_basis,
    compute_spectral_height, 
    compute_spatial_heights
)

def test_wave_amplitudes():
    """Test different parameter combinations for better wave amplitudes."""
    print("=== Testing Wave Amplitudes ===")
    
    # Test different parameter sets
    param_sets = [
        ("Default small", OceanParameters(resolution=64, domain=100.0, wind_speed=15.0, fetch_km=300.0)),
        ("Stronger wind", OceanParameters(resolution=64, domain=100.0, wind_speed=30.0, fetch_km=500.0)),
        ("Bigger domain", OceanParameters(resolution=64, domain=500.0, wind_speed=20.0, fetch_km=1000.0)),
        ("Storm conditions", OceanParameters(resolution=64, domain=200.0, wind_speed=40.0, fetch_km=2000.0)),
    ]
    
    for name, ocean_params in param_sets:
        print(f"\n=== {name} ===")
        print(f"Parameters: wind_speed={ocean_params.wind_speed}, domain={ocean_params.domain}, fetch_km={ocean_params.fetch_km}")
        
        # Create plan and compute waves
        plan = create_ocean_plan(ocean_params)
        spectral_basis = compute_spectral_basis(plan)
        
        # Test at time=5.0 when waves should be well developed
        spectral_height = compute_spectral_height(plan, 5.0, spectral_basis)
        spatial_heights = compute_spatial_heights(plan, spectral_height)
        
        print(f"Wave height range: {spatial_heights.min():.4f} to {spatial_heights.max():.4f} meters")
        print(f"Wave height std: {spatial_heights.std():.4f} meters")
        print(f"Peak-to-peak: {spatial_heights.max() - spatial_heights.min():.4f} meters")
        
        # Check if waves are reasonable size (at least a few cm)
        peak_to_peak = spatial_heights.max() - spatial_heights.min()
        if peak_to_peak > 0.1:
            print(f"✓ Good wave amplitude: {peak_to_peak:.3f}m")
        elif peak_to_peak > 0.01:
            print(f"⚠ Small but visible waves: {peak_to_peak:.3f}m")
        else:
            print(f"✗ Waves too small: {peak_to_peak:.3f}m")

if __name__ == "__main__":
    test_wave_amplitudes() 