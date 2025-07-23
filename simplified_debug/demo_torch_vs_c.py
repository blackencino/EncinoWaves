#!/usr/bin/env python3
"""
Demonstration script comparing PyTorch and C implementations of spectral basis computation.
"""

import numpy as np
import time
import encino_waves as ew

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


def main():
    print("=== Encino Waves PyTorch vs C Implementation Comparison ===\n")
    
    if not _HAS_TORCH:
        print("PyTorch not available - please install PyTorch to run this demo")
        return
    
    # Test configuration
    resolution = 256
    params = ew.OceanParameters(
        resolution=resolution,
        wind_speed=20.0,
        fetch_km=500.0,
        depth=100.0,
        random_seed=12345
    )
    
    print(f"Configuration:")
    print(f"  Resolution: {params.resolution}")
    print(f"  Wind Speed: {params.wind_speed} m/s")
    print(f"  Fetch: {params.fetch_km} km")
    print(f"  Depth: {params.depth} m")
    print(f"  Random Seed: {params.random_seed}")
    print()
    
    # Create plan
    plan = ew.create_ocean_plan(params)
    print(f"Ocean Plan:")
    print(f"  Grid size: {plan.N} x {plan.N}")
    print(f"  Spectral size: {plan.size_j} x {plan.size_i}")
    print(f"  Wave number spacing (dk): {plan.dk:.6f}")
    print()
    
    # C Implementation timing
    print("=== C Implementation ===")
    start_time = time.time()
    c_result = ew.compute_spectral_basis(plan, max_threads=0)  # Use all threads
    c_time = time.time() - start_time
    
    print(f"Time: {c_time:.4f} seconds")
    print(f"Shape: {c_result.shape}")
    print(f"Dtype: {c_result.dtype}")
    print(f"Memory: {c_result.nbytes / 1024 / 1024:.2f} MB")
    
    # Statistics for each channel
    channel_names = ['amp_pos_real', 'amp_pos_imag', 'amp_neg_real', 'amp_neg_imag', 'omega']
    for i, name in enumerate(channel_names):
        channel = c_result[:, :, i]
        print(f"  {name}: min={np.min(channel):.6f}, max={np.max(channel):.6f}, std={np.std(channel):.6f}")
    print()
    
    # PyTorch CPU Implementation timing
    print("=== PyTorch CPU Implementation ===")
    start_time = time.time()
    torch_cpu_result = ew.compute_spectral_basis_torch(plan, device=torch.device('cpu'))
    torch_cpu_time = time.time() - start_time
    
    print(f"Time: {torch_cpu_time:.4f} seconds")
    print(f"Shape: {torch_cpu_result.shape}")
    print(f"Dtype: {torch_cpu_result.dtype}")
    print(f"Memory: {torch_cpu_result.nbytes / 1024 / 1024:.2f} MB")
    
    for i, name in enumerate(channel_names):
        channel = torch_cpu_result[:, :, i]
        print(f"  {name}: min={np.min(channel):.6f}, max={np.max(channel):.6f}, std={np.std(channel):.6f}")
    print()
    
    print(f"PyTorch CPU vs C speed ratio: {torch_cpu_time / c_time:.2f}x")
    print()
    
    # PyTorch GPU Implementation if available
    if torch.cuda.is_available():
        print("=== PyTorch GPU Implementation ===")
        start_time = time.time()
        torch_gpu_result = ew.compute_spectral_basis_torch(plan, device=torch.device('cuda:0'))
        torch_gpu_time = time.time() - start_time
        
        print(f"Time: {torch_gpu_time:.4f} seconds")
        print(f"Shape: {torch_gpu_result.shape}")
        print(f"Dtype: {torch_gpu_result.dtype}")
        print(f"Memory: {torch_gpu_result.nbytes / 1024 / 1024:.2f} MB")
        
        for i, name in enumerate(channel_names):
            channel = torch_gpu_result[:, :, i]
            print(f"  {name}: min={np.min(channel):.6f}, max={np.max(channel):.6f}, std={np.std(channel):.6f}")
        print()
        
        print(f"PyTorch GPU vs C speed ratio: {torch_gpu_time / c_time:.2f}x")
        print(f"PyTorch GPU vs CPU speed ratio: {torch_gpu_time / torch_cpu_time:.2f}x")
        print()
        
        # Verify GPU and CPU results match
        max_diff = np.max(np.abs(torch_gpu_result - torch_cpu_result))
        print(f"Max difference between GPU and CPU: {max_diff:.8f}")
        print()
    
    # Compare C and PyTorch results
    print("=== Comparison Analysis ===")
    print("Differences between C and PyTorch CPU implementations:")
    
    for i, name in enumerate(channel_names):
        c_channel = c_result[:, :, i]
        torch_channel = torch_cpu_result[:, :, i]
        
        diff = c_channel - torch_channel
        max_abs_diff = np.max(np.abs(diff))
        mean_abs_diff = np.mean(np.abs(diff))
        
        c_magnitude = np.max(np.abs(c_channel))
        rel_diff = max_abs_diff / (c_magnitude + 1e-10) if c_magnitude > 1e-10 else 0.0
        
        print(f"  {name}: max_abs={max_abs_diff:.6f}, mean_abs={mean_abs_diff:.6f}, rel={rel_diff:.6f}")
    
    print()
    print("Note: Some differences are expected due to different implementations")
    print("      of wave number discretization and numerical precision.")
    print()
    
    # Sanity checks
    print("=== Sanity Checks ===")
    
    # DC components should be zero
    c_dc = c_result[0, 0, :]
    torch_dc = torch_cpu_result[0, 0, :]
    print(f"DC component (should be ~0): C={np.max(np.abs(c_dc)):.8f}, PyTorch={np.max(np.abs(torch_dc)):.8f}")
    
    # Omega should be non-negative
    c_omega = c_result[:, :, 4]
    torch_omega = torch_cpu_result[:, :, 4]
    print(f"Omega non-negative: C={np.all(c_omega >= 0)}, PyTorch={np.all(torch_omega >= 0)}")
    
    # Results should be finite
    print(f"All finite: C={np.all(np.isfinite(c_result))}, PyTorch={np.all(np.isfinite(torch_cpu_result))}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main() 