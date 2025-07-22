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

import ctypes
import numpy as np
import polyscope as ps
import platform
import os
import atexit

# --- Configuration & Library Loading ---

def find_library(name):
    if platform.system() == "Windows":
        lib_name = f"{name}.dll"
    elif platform.system() == "Darwin":
        lib_name = f"lib{name}.dylib"
    else: # Linux
        lib_name = f"lib{name}.so"
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_name))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shared library not found at {path}. "
            "Please run the build script in 'cpp_library/'.") # Corrected directory name
    return path

LIB = ctypes.CDLL(find_library("encino_waves"))

# --- Error Handling ---
ERROR_CODES = {0: "OK", 1: "Unknown Error", 2: "Invalid Argument", 3: "Out of Memory"}

def check_err(result_code, context=""):
    if result_code != 0:
        message = ERROR_CODES.get(result_code, f"Unknown error code: {result_code}")
        raise RuntimeError(f"Error in C++ library: {message} ({context})")

# --- ctypes Structure Definitions ---
class InitialStateParams(ctypes.Structure):
    _fields_ = [
        ("resolution", ctypes.c_int), 
        ("domain", ctypes.c_float),
        ("gravity", ctypes.c_float), 
        ("surface_tension", ctypes.c_float),
        ("density", ctypes.c_float), 
        ("depth", ctypes.c_float),
        ("wind_speed", ctypes.c_float), 
        ("fetch", ctypes.c_float),
        ("swell", ctypes.c_float), 
        ("filter_soft_width", ctypes.c_float),
        ("filter_small_wavelength", ctypes.c_float), 
        ("filter_big_wavelength", ctypes.c_float),
        ("filter_min", ctypes.c_float), 
        ("filter_invert", ctypes.c_bool),
        ("filter", ctypes.c_bool), 
        ("random_seed", ctypes.c_int),
    ]

class PropagationParams(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_float), 
        ("pinch", ctypes.c_float),
        ("amplitude_gain", ctypes.c_float),
    ]

# --- ctypes Function Prototypes ---
LIB.encino_waves_create_initial_state.argtypes = [ctypes.POINTER(InitialStateParams), ctypes.POINTER(ctypes.c_int)]
LIB.encino_waves_create_initial_state.restype = ctypes.c_int
LIB.encino_waves_destroy_initial_state.argtypes = [ctypes.c_int]
LIB.encino_waves_destroy_initial_state.restype = ctypes.c_int
LIB.encino_waves_propagate.argtypes = [
    ctypes.c_int, ctypes.POINTER(PropagationParams), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
]
LIB.encino_waves_propagate.restype = ctypes.c_int
LIB.encino_waves_shutdown.argtypes = []
LIB.encino_waves_shutdown.restype = None
atexit.register(LIB.encino_waves_shutdown)

# --- Pythonic Wrapper Class ---
class InitialState:
    def __init__(self, params: InitialStateParams):
        print("Creating new C++ initial state...")
        self.params = params
        self.id = ctypes.c_int(0)
        check_err(LIB.encino_waves_create_initial_state(ctypes.byref(self.params), ctypes.byref(self.id)), "create_initial_state")
        self.N = params.resolution
        print(f"  --> Success! Created state with ID: {self.id.value}")

    def __del__(self):
        if hasattr(self, 'id') and self.id.value != 0:
            print(f"Destroying C++ initial state with ID: {self.id.value}")
            try:
                check_err(LIB.encino_waves_destroy_initial_state(self.id), "destroy_initial_state")
            except Exception as e:
                print(f"  --> Warning: Could not destroy state {self.id.value}. Reason: {e}")

    def propagate(self, prop_params: PropagationParams, out_array: np.ndarray):
        if out_array.shape != (self.N, self.N):
            raise ValueError(f"Output array shape must be ({self.N}, {self.N}) but got {out_array.shape}")
        rank = out_array.ndim
        shape = (ctypes.c_int * rank)(*out_array.shape)
        data_ptr = out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        check_err(LIB.encino_waves_propagate(self.id, ctypes.byref(prop_params), rank, shape, data_ptr), "propagate")

# --- Polyscope UI and Main Logic ---
ps_mesh = None
current_initial_state = None
height_field_data = None
Z_OFFSET = 50.0  # Offset to prevent intersection with ground plane

init_params = InitialStateParams(
    resolution=256, domain=500.0, gravity=9.81, surface_tension=0.074,
    density=1000.0, depth=100.0, wind_speed=15.0, fetch=300.0,
    swell=0.0, filter_soft_width=0.01, filter_small_wavelength=0.01,
    filter_big_wavelength=1000000.0, filter_min=0.0, filter_invert=False,
    filter=False, random_seed=12345
)
prop_params = PropagationParams(time=0.0, pinch=0.75, amplitude_gain=1.0)

def create_mesh_geometry():
    """Create the mesh geometry (faces and initial vertices). Only called when resolution changes."""
    global ps_mesh
    print("Creating new mesh geometry...")
    N = current_initial_state.N
    xx, yy = np.meshgrid(np.linspace(-init_params.domain/2, init_params.domain/2, N), 
                         np.linspace(-init_params.domain/2, init_params.domain/2, N))
    # Use z=Z_OFFSET for initial vertex positions (will be updated with height data)
    vertices = np.stack([xx.flatten(), yy.flatten(), np.full(N*N, Z_OFFSET)], axis=1)
    faces = [
        [i * N + j, i * N + (j + 1), (i + 1) * N + (j + 1), (i + 1) * N + j]
        for i in range(N - 1) for j in range(N - 1)
    ]
    ps_mesh = ps.register_surface_mesh("Encino Waves", vertices, np.array(faces), smooth_shade=True)

def update_vertices():
    """Update mesh vertices with current height data. Called for any parameter change."""
    global ps_mesh
    current_initial_state.propagate(prop_params, height_field_data)
    
    N = current_initial_state.N
    xx, yy = np.meshgrid(np.linspace(-init_params.domain/2, init_params.domain/2, N), 
                         np.linspace(-init_params.domain/2, init_params.domain/2, N))
    # Add Z_OFFSET to height field to prevent ground plane intersection
    vertices = np.stack([xx.flatten(), yy.flatten(), height_field_data.flatten() + Z_OFFSET], axis=1)
    ps_mesh.update_vertex_positions(vertices)
    ps_mesh.add_scalar_quantity("height", height_field_data.flatten(), defined_on='vertices', enabled=True, cmap='viridis')

def recreate_initial_state():
    """Recreate the C++ initial state with current parameters."""
    global current_initial_state, height_field_data
    print("Recreating initial state with new parameters...")
    current_initial_state = InitialState(init_params)
    height_field_data = np.zeros((current_initial_state.N, current_initial_state.N), dtype=np.float32)

def initialize_all():
    """Full initialization - called at startup and when resolution changes."""
    global ps_mesh
    recreate_initial_state()
    ps_mesh = None  # Force mesh recreation
    create_mesh_geometry()
    update_vertices()

def callback():
    global init_params, prop_params
    resolution_changed = False
    init_changed = False
    prop_changed = False

    ps.imgui.PushItemWidth(160)
    
    if ps.imgui.CollapsingHeader("Initial State Parameters", open=True):
        resolutions = [2**i for i in range(5, 12)]
        res_options = [str(r) for r in resolutions]
        try:
            curr_res_index = resolutions.index(init_params.resolution)
        except ValueError:
            curr_res_index = 0
        
        changed, new_res_index = ps.imgui.Combo("Resolution", curr_res_index, res_options)
        if changed:
            init_params.resolution = resolutions[new_res_index]
            resolution_changed = True
        
        changed, new_val = ps.imgui.InputFloat("Domain Size (m)", init_params.domain, step=10.0)
        if changed: init_params.domain = new_val; init_changed = True

        changed, new_val = ps.imgui.InputFloat("Wind Speed (m/s)", init_params.wind_speed, step=0.5)
        if changed: init_params.wind_speed = new_val; init_changed = True

        changed, new_val = ps.imgui.InputFloat("Fetch (km)", init_params.fetch, step=10.0)
        if changed: init_params.fetch = new_val; init_changed = True

        changed, new_val = ps.imgui.SliderFloat("Swell", init_params.swell, v_min=-1.0, v_max=1.0)
        if changed: init_params.swell = new_val; init_changed = True
        
        changed, new_val = ps.imgui.InputInt("Random Seed", init_params.random_seed)
        if changed: init_params.random_seed = new_val; init_changed = True


    if ps.imgui.CollapsingHeader("Propagation Parameters", open=True):
        changed, new_val = ps.imgui.InputFloat("Time (s)", prop_params.time, step=0.05)
        if changed: prop_params.time = new_val; prop_changed = True
        
        changed, new_val = ps.imgui.SliderFloat("Amplitude Gain", prop_params.amplitude_gain, v_min=0.0, v_max=5.0)
        if changed: prop_params.amplitude_gain = new_val; prop_changed = True
        

    ps.imgui.PopItemWidth()

    # Handle changes with appropriate optimization
    if resolution_changed:
        # Resolution change requires full rebuild
        initialize_all()
    elif init_changed:
        # Other init param changes only need new initial state + vertex update
        recreate_initial_state()
        update_vertices()
    elif prop_changed:
        # Propagation param changes only need vertex update
        update_vertices()

def main():
    ps.init()
    ps.set_up_dir("z_up")  # Make polyscope use z-up orientation
    initialize_all()
    ps.set_user_callback(callback)
    ps.show()

if __name__ == "__main__":
    main()
