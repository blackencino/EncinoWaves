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
Encino Waves Visualization using Polyscope

Interactive visualization of ocean wave simulation with real-time parameter adjustment.
Features:
- Real-time ocean parameter tweaking 
- Wave animation with time control
- Visualization scaling for better wave visibility
- Z-up coordinate system for proper wave display
- FFT method comparison (FFTW vs numpy) with difference visualization
"""

import numpy as np
import polyscope as ps
from dataclasses import replace

# Import the new encino_waves interface 
from encino_waves import (
    OceanParameters, 
    create_ocean_plan, 
    compute_spectral_basis,
    compute_spectral_height, 
    compute_spatial_heights_fftw,
    compute_spatial_heights,
    _HAS_FFTW
)

# --- Global State Management ---

ps_mesh = None
current_ocean_plan = None
current_spectral_basis = None
height_field_data = None
Z_OFFSET = 50.0  # Offset to prevent intersection with ground plane

# Initial parameters using the new OceanParameters interface
# Use larger domain and stronger wind for more visible waves
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

# Propagation parameters (time is the main one we'll animate)
current_time = 0.0

# Visualization scaling factor to make waves more visible
wave_scale_factor = 100.0

# Comparison mode settings
use_fftw = True  # True = FFTW, False = numpy
show_difference = False  # Show difference between FFTW and numpy
height_field_fftw = None  # Cache FFTW results for comparison
height_field_numpy = None  # Cache numpy results for comparison
comparison_stats = {"max_diff": 0.0, "mean_diff": 0.0, "rmse_diff": 0.0}

def create_mesh_geometry():
    """Create the mesh geometry (faces and initial vertices). Only called when resolution changes."""
    global ps_mesh
    print("Creating new mesh geometry...")
    N = current_ocean_plan.N
    xx, yy = np.meshgrid(
        np.linspace(-ocean_params.domain/2, ocean_params.domain/2, N), 
        np.linspace(-ocean_params.domain/2, ocean_params.domain/2, N)
    )
    # Use z=Z_OFFSET for initial vertex positions (will be updated with height data)
    vertices = np.stack([xx.flatten(), yy.flatten(), np.full(N*N, Z_OFFSET)], axis=1)
    faces = [
        [i * N + j, i * N + (j + 1), (i + 1) * N + (j + 1), (i + 1) * N + j]
        for i in range(N - 1) for j in range(N - 1)
    ]
    ps_mesh = ps.register_surface_mesh("Encino Waves", vertices, np.array(faces), smooth_shade=True)

def update_vertices():
    """Update mesh vertices with current height data. Called for any parameter change."""
    global ps_mesh, height_field_data, height_field_fftw, height_field_numpy, comparison_stats
    
    # Compute spectral height once for both methods
    spectral_height = compute_spectral_height(current_ocean_plan, current_time, current_spectral_basis)
    
    # Compute heights with both methods if in comparison mode or if we need the specific method
    if show_difference or use_fftw:
        if _HAS_FFTW:
            height_field_fftw = compute_spatial_heights_fftw(current_ocean_plan, spectral_height, out=height_field_fftw)
        else:
            print("Warning: FFTW not available, using numpy instead")
            height_field_fftw = compute_spatial_heights(current_ocean_plan, spectral_height, out=height_field_fftw)
    
    if show_difference or not use_fftw:
        height_field_numpy = compute_spatial_heights(current_ocean_plan, spectral_height, out=height_field_numpy)
    
    # Choose which height field to display
    if show_difference and _HAS_FFTW:
        # Show the difference between FFTW and numpy
        difference = height_field_fftw - height_field_numpy
        height_field_data = difference
        
        # Update comparison statistics
        comparison_stats["max_diff"] = float(np.max(np.abs(difference)))
        comparison_stats["mean_diff"] = float(np.mean(np.abs(difference))) 
        comparison_stats["rmse_diff"] = float(np.sqrt(np.mean(difference**2)))
        
        scalar_name = "difference (FFTW - numpy)"
        colormap = 'coolwarm'  # Good for showing positive/negative differences
    else:
        # Show the selected method
        if use_fftw and _HAS_FFTW:
            height_field_data = height_field_fftw
            scalar_name = "height (FFTW)"
        else:
            height_field_data = height_field_numpy
            scalar_name = "height (numpy)"
        colormap = 'viridis'
    
    N = current_ocean_plan.N
    xx, yy = np.meshgrid(
        np.linspace(-ocean_params.domain/2, ocean_params.domain/2, N), 
        np.linspace(-ocean_params.domain/2, ocean_params.domain/2, N)
    )
    
    # Scale the height field for better visualization and add Z_OFFSET
    if show_difference:
        # For differences, use a smaller scale factor and center around Z_OFFSET
        scaled_heights = height_field_data * wave_scale_factor * 10.0 + Z_OFFSET
    else:
        scaled_heights = height_field_data * wave_scale_factor + Z_OFFSET
    
    vertices = np.stack([xx.flatten(), yy.flatten(), scaled_heights.flatten()], axis=1)
    ps_mesh.update_vertex_positions(vertices)
    ps_mesh.add_scalar_quantity(scalar_name, height_field_data.flatten(), 
                               defined_on='vertices', enabled=True, cmap=colormap)

def recreate_ocean_state():
    """Recreate the ocean plan and spectral basis with current parameters."""
    global current_ocean_plan, current_spectral_basis, height_field_data, height_field_fftw, height_field_numpy
    print("Recreating ocean state with new parameters...")
    
    # Create new ocean plan and spectral basis using functional interface
    current_ocean_plan = create_ocean_plan(ocean_params)
    current_spectral_basis = compute_spectral_basis(current_ocean_plan, use_cupy=True)
    
    # Prepare output arrays for spatial heights (all methods)
    array_shape = (current_ocean_plan.N, current_ocean_plan.N)
    height_field_data = np.zeros(array_shape, dtype=np.float32)
    height_field_fftw = np.zeros(array_shape, dtype=np.float32)
    height_field_numpy = np.zeros(array_shape, dtype=np.float32)

def initialize_all():
    """Full initialization - called at startup and when resolution changes."""
    global ps_mesh
    recreate_ocean_state()
    ps_mesh = None  # Force mesh recreation
    create_mesh_geometry()
    update_vertices()

def callback():
    global ocean_params, current_time, wave_scale_factor, use_fftw, show_difference
    resolution_changed = False
    ocean_params_changed = False
    time_changed = False
    scale_changed = False
    comparison_changed = False

    ps.imgui.PushItemWidth(160)
    
    if ps.imgui.CollapsingHeader("Ocean Parameters", open=True):
        resolutions = [2**i for i in range(5, 12)]
        res_options = [str(r) for r in resolutions]
        try:
            curr_res_index = resolutions.index(ocean_params.resolution)
        except ValueError:
            curr_res_index = 0
        
        changed, new_res_index = ps.imgui.Combo("Resolution", curr_res_index, res_options)
        if changed:
            ocean_params = replace(ocean_params, resolution=resolutions[new_res_index])
            resolution_changed = True
        
        changed, new_val = ps.imgui.InputFloat("Domain Size (m)", ocean_params.domain, step=10.0)
        if changed: 
            ocean_params = replace(ocean_params, domain=new_val)
            ocean_params_changed = True

        changed, new_val = ps.imgui.InputFloat("Wind Speed (m/s)", ocean_params.wind_speed, step=0.5)
        if changed: 
            ocean_params = replace(ocean_params, wind_speed=new_val)
            ocean_params_changed = True

        changed, new_val = ps.imgui.InputFloat("Fetch (km)", ocean_params.fetch_km, step=10.0)
        if changed: 
            ocean_params = replace(ocean_params, fetch_km=new_val)
            ocean_params_changed = True

        changed, new_val = ps.imgui.SliderFloat("Swell", ocean_params.swell, v_min=-1.0, v_max=1.0)
        if changed: 
            ocean_params = replace(ocean_params, swell=new_val)
            ocean_params_changed = True
        
        changed, new_val = ps.imgui.InputInt("Random Seed", ocean_params.random_seed)
        if changed: 
            ocean_params = replace(ocean_params, random_seed=new_val)
            ocean_params_changed = True

        changed, new_val = ps.imgui.InputFloat("Gravity (m/s²)", ocean_params.gravity, step=0.1)
        if changed: 
            ocean_params = replace(ocean_params, gravity=new_val)
            ocean_params_changed = True

        changed, new_val = ps.imgui.InputFloat("Depth (m)", ocean_params.depth, step=10.0)
        if changed: 
            ocean_params = replace(ocean_params, depth=new_val)
            ocean_params_changed = True

    if ps.imgui.CollapsingHeader("Animation Parameters", open=True):
        changed, new_val = ps.imgui.InputFloat("Time (s)", current_time, step=0.05)
        if changed: 
            current_time = new_val
            time_changed = True

    if ps.imgui.CollapsingHeader("FFT Comparison", open=True):
        if not _HAS_FFTW:
            ps.imgui.TextColored((1.0, 0.5, 0.0, 1.0), "FFTW not available - using numpy only")
        else:
            changed, new_val = ps.imgui.Checkbox("Use FFTW", use_fftw)
            if changed:
                use_fftw = new_val
                comparison_changed = True
            
            ps.imgui.SameLine()
            changed, new_val = ps.imgui.Checkbox("Show Difference", show_difference)
            if changed:
                show_difference = new_val
                comparison_changed = True
            
            if show_difference:
                ps.imgui.Separator()
                ps.imgui.Text("Difference Statistics:")
                ps.imgui.Text(f"Max difference: {comparison_stats['max_diff']:.2e}")
                ps.imgui.Text(f"Mean abs diff:  {comparison_stats['mean_diff']:.2e}")
                ps.imgui.Text(f"RMS difference: {comparison_stats['rmse_diff']:.2e}")
                ps.imgui.Text("(Difference scaled 10x for visibility)")
                if comparison_stats['max_diff'] < 1e-5:
                    ps.imgui.TextColored((0.0, 1.0, 0.0, 1.0), "Excellent agreement!")

    if ps.imgui.CollapsingHeader("Visualization", open=True):
        changed, new_val = ps.imgui.SliderFloat("Wave Scale", wave_scale_factor, v_min=1.0, v_max=1000.0)
        if changed:
            wave_scale_factor = new_val
            scale_changed = True

    ps.imgui.PopItemWidth()

    # Handle changes with appropriate optimization
    if resolution_changed:
        # Resolution change requires full rebuild
        initialize_all()
    elif ocean_params_changed:
        # Other ocean param changes need new plan and spectral basis + vertex update
        recreate_ocean_state()
        update_vertices() 
    elif time_changed or scale_changed or comparison_changed:
        # Time, scale, or comparison changes only need vertex update (reuses existing plan and spectral basis)
        update_vertices()

def main():
    ps.init()
    ps.set_up_dir("z_up")  # Make polyscope use z-up orientation
    initialize_all()
    ps.set_user_callback(callback)
    ps.show()

if __name__ == "__main__":
    main()
