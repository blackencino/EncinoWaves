#!/usr/bin/env python3

import numpy as np
from encino_waves import (
    OceanParameters, 
    create_ocean_plan, 
    compute_spectral_basis,
    compute_spectral_height, 
    compute_spatial_heights
)

def debug_wave_computation():
    """Debug the wave computation pipeline step by step."""
    print("=== Debugging Wave Computation Pipeline ===")
    
    # Create parameters
    ocean_params = OceanParameters(
        resolution=64,  # Smaller for easier debugging
        domain=100.0,
        wind_speed=15.0,
        fetch_km=300.0,
        random_seed=12345
    )
    print(f"Ocean parameters: {ocean_params}")
    
    # Step 1: Create ocean plan
    print("\n1. Creating ocean plan...")
    plan = create_ocean_plan(ocean_params)
    print(f"Plan N: {plan.N}")
    print(f"Plan size_i: {plan.size_i}, size_j: {plan.size_j}")
    print(f"Plan count: {plan.count}")
    print(f"Plan dk: {plan.dk}")
    print(f"Plan max_k_mag: {plan.max_k_mag}")
    
    # Step 2: Compute spectral basis
    print("\n2. Computing spectral basis...")
    spectral_basis = compute_spectral_basis(plan)
    print(f"Spectral basis shape: {spectral_basis.shape}")
    print(f"Spectral basis dtype: {spectral_basis.dtype}")
    print(f"Spectral basis min: {spectral_basis.min()}")
    print(f"Spectral basis max: {spectral_basis.max()}")
    print(f"Spectral basis mean: {spectral_basis.mean()}")
    print(f"Spectral basis std: {spectral_basis.std()}")
    
    # Check each channel
    for i in range(5):
        channel = spectral_basis[:, :, i]
        print(f"  Channel {i}: min={channel.min():.6f}, max={channel.max():.6f}, mean={channel.mean():.6f}, std={channel.std():.6f}")
    
    # Step 3: Compute spectral height at different times
    times = [0.0, 1.0, 5.0]
    for time in times:
        print(f"\n3. Computing spectral height at time {time}...")
        spectral_height = compute_spectral_height(plan, time, spectral_basis)
        print(f"Spectral height shape: {spectral_height.shape}")
        print(f"Spectral height dtype: {spectral_height.dtype}")
        
        if spectral_height.dtype == np.complex64:
            print(f"Spectral height real min: {spectral_height.real.min()}")
            print(f"Spectral height real max: {spectral_height.real.max()}")
            print(f"Spectral height imag min: {spectral_height.imag.min()}")
            print(f"Spectral height imag max: {spectral_height.imag.max()}")
            print(f"Spectral height magnitude min: {np.abs(spectral_height).min()}")
            print(f"Spectral height magnitude max: {np.abs(spectral_height).max()}")
        else:
            print(f"Spectral height min: {spectral_height.min()}")
            print(f"Spectral height max: {spectral_height.max()}")
        
        # Step 4: Compute spatial heights
        print(f"4. Computing spatial heights at time {time}...")
        spatial_heights = compute_spatial_heights(plan, spectral_height)
        print(f"Spatial heights shape: {spatial_heights.shape}")
        print(f"Spatial heights dtype: {spatial_heights.dtype}")
        print(f"Spatial heights min: {spatial_heights.min()}")
        print(f"Spatial heights max: {spatial_heights.max()}")
        print(f"Spatial heights mean: {spatial_heights.mean()}")
        print(f"Spatial heights std: {spatial_heights.std()}")
        
        # Check a few sample values
        print(f"Sample spatial heights [0,0]: {spatial_heights[0,0]}")
        print(f"Sample spatial heights [N//2,N//2]: {spatial_heights[plan.N//2,plan.N//2]}")

if __name__ == "__main__":
    debug_wave_computation() 