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
Python wrapper for the Encino Waves C++ library using ctypes.

This module provides a functional, immutable interface to the high-performance 
ocean wave simulation library. All operations are pure functions that take
their dependencies as parameters.
"""

import ctypes
import numpy as np
import platform
import os
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, NamedTuple
from enum import IntEnum


# --- Library Loading ---

def find_library(name: str) -> str:
    """Find the platform-specific shared library."""
    if platform.system() == "Windows":
        lib_name = f"{name}.dll"
    elif platform.system() == "Darwin":
        lib_name = f"lib{name}.dylib"
    else:  # Linux
        lib_name = f"lib{name}.so"
    
    # Look in the same directory as this module
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_name))
    if not os.path.exists(path):
        # Try parent directory
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", lib_name))
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shared library '{lib_name}' not found. "
            "Please compile the C++ library first."
        )
    return path


# Load the library
LIB = ctypes.CDLL(find_library("encino_waves"))

# Check what functions are available
def _check_function_availability():
    """Check what functions are available in the library."""
    functions = [
        'encino_waves_create_ocean_plan',
        'encino_waves_spectral_basis_omp', 
        'encino_waves_spectral_height_omp',
        'encino_waves_classic_spectral_basis_at_kidx',
        'encino_waves_spectral_height_at'
    ]
    
    available = {}
    for func_name in functions:
        try:
            getattr(LIB, func_name)
            available[func_name] = True
        except AttributeError:
            available[func_name] = False
    
    return available

_AVAILABLE_FUNCTIONS = _check_function_availability()

# Print diagnostics
print("Library function availability:")
for func, avail in _AVAILABLE_FUNCTIONS.items():
    print(f"  {func}: {'✓' if avail else '✗'}")

if not _AVAILABLE_FUNCTIONS['encino_waves_create_ocean_plan']:
    raise RuntimeError("Core library functions not found. Library may not be properly compiled.")

_HAS_OMP = _AVAILABLE_FUNCTIONS['encino_waves_spectral_basis_omp']

if not _HAS_OMP:
    print("\nWarning: OpenMP functions not available.")
    print("Compile the C++ library with -DENCINO_WAVES_OMP_KERNELS for better performance.")
    print("Falling back to single-threaded computation.\n")


# --- Error Handling ---

class ErrorCode(IntEnum):
    """Encino Waves error codes."""
    OK = 0
    NULL_INPUT = 1
    INVALID_RESOLUTION = 2
    INVALID_RANK = 3
    INVALID_SHAPE = 4


def check_error(result: int, context: str = "") -> None:
    """Check the result code and raise an exception if it indicates an error."""
    if result != ErrorCode.OK:
        error_messages = {
            ErrorCode.NULL_INPUT: "Null input provided",
            ErrorCode.INVALID_RESOLUTION: "Invalid resolution (must be power of 2 between 4 and 8192)",
            ErrorCode.INVALID_RANK: "Invalid array rank",
            ErrorCode.INVALID_SHAPE: "Invalid array shape"
        }
        msg = error_messages.get(result, f"Unknown error code: {result}")
        if context:
            msg = f"{msg} (context: {context})"
        raise RuntimeError(f"Encino Waves error: {msg}")


# --- Parameter Ranges ---

@dataclass(frozen=True)
class ParameterRange:
    """Defines the valid range and default value for a parameter."""
    default: float
    min: float
    max: float
    description: str = ""

    def clamp(self, value: float) -> float:
        """Clamp a value to the valid range."""
        return np.clip(value, self.min, self.max)


# Parameter definitions with defaults and ranges (immutable)
PARAM_RANGES = {
    'resolution': ParameterRange(512, 4, 8192, "Grid resolution (must be power of 2)"),
    'domain': ParameterRange(100.0, 0.01, 100000.0, "Domain size in meters"),
    'gravity': ParameterRange(9.81, 0.1, 100.0, "Gravitational acceleration (m/s²)"),
    'surface_tension': ParameterRange(0.074, 0.001, 1.0, "Surface tension (N/m)"),
    'density': ParameterRange(1000.0, 0.001, 10000.0, "Fluid density (kg/m³)"),
    'depth': ParameterRange(100.0, 0.01, 20000.0, "Ocean depth (m)"),
    'wind_speed': ParameterRange(17.0, 0.01, 300.0, "Wind speed (m/s)"),
    'fetch_km': ParameterRange(300.0, 0.01, 20000.0, "Fetch distance (km)"),
    'swell': ParameterRange(0.0, -1.0, 1.0, "Swell directionality (-1 to 1)"),
    'random_seed': ParameterRange(54321, 0, 2**31-1, "Random seed for wave generation")
}


# --- ctypes Structure Definitions ---

class _OceanParamsStruct(ctypes.Structure):
    """Ocean simulation parameters (matches Encino_waves_ocean_params)."""
    _fields_ = [
        ("resolution", ctypes.c_int),
        ("domain", ctypes.c_float),
        ("gravity", ctypes.c_float),
        ("surface_tension", ctypes.c_float),
        ("density", ctypes.c_float),
        ("depth", ctypes.c_float),
        ("wind_speed", ctypes.c_float),
        ("fetch_km", ctypes.c_float),
        ("swell", ctypes.c_float),
        ("random_seed", ctypes.c_int),
    ]


class _OceanPlanStruct(ctypes.Structure):
    """Ocean simulation plan (matches Encino_waves_ocean_plan)."""
    _fields_ = [
        ("N", ctypes.c_int),
        ("size_i", ctypes.c_int),
        ("size_j", ctypes.c_int),
        ("count", ctypes.c_int),
        ("dk", ctypes.c_float),
        ("max_k_mag", ctypes.c_float),
        ("gravity", ctypes.c_float),
        ("sigma_over_rho", ctypes.c_float),
        ("depth", ctypes.c_float),
        ("wind_speed", ctypes.c_float),
        ("fetch_m", ctypes.c_float),
        ("tma_gamma", ctypes.c_float),
        ("tma_alpha", ctypes.c_float),
        ("tma_kd_gain", ctypes.c_float),
        ("peak_omega", ctypes.c_float),
        ("wind_speed_over_celerity", ctypes.c_float),
        ("swell", ctypes.c_float),
        ("random_seed", ctypes.c_int),
    ]


# --- Function Prototypes ---

# encino_waves_create_ocean_plan
LIB.encino_waves_create_ocean_plan.argtypes = [
    ctypes.POINTER(_OceanParamsStruct),
    ctypes.POINTER(_OceanPlanStruct)
]
LIB.encino_waves_create_ocean_plan.restype = ctypes.c_int

# encino_waves_spectral_basis_omp
LIB.encino_waves_spectral_basis_omp.argtypes = [
    ctypes.POINTER(_OceanPlanStruct),
    ctypes.c_int,  # max_threads
    ctypes.c_int,  # out_rank
    ctypes.POINTER(ctypes.c_int),  # out_shape
    ctypes.POINTER(ctypes.c_float)  # out_spectral_basis
]
LIB.encino_waves_spectral_basis_omp.restype = ctypes.c_int

# encino_waves_spectral_height_omp
LIB.encino_waves_spectral_height_omp.argtypes = [
    ctypes.POINTER(_OceanPlanStruct),
    ctypes.c_float,  # time
    ctypes.c_int,  # max_threads
    ctypes.c_int,  # in_rank
    ctypes.POINTER(ctypes.c_int),  # in_shape
    ctypes.POINTER(ctypes.c_float),  # spectral_basis
    ctypes.c_int,  # out_rank
    ctypes.POINTER(ctypes.c_int),  # out_shape
    ctypes.POINTER(ctypes.c_float)  # out_spectral_height
]
LIB.encino_waves_spectral_height_omp.restype = ctypes.c_int

# encino_waves_classic_spectral_basis_at_kidx
LIB.encino_waves_classic_spectral_basis_at_kidx.argtypes = [
    ctypes.POINTER(_OceanPlanStruct),
    ctypes.c_int,  # wave_number_index
    ctypes.POINTER(ctypes.c_float)  # out
]
LIB.encino_waves_classic_spectral_basis_at_kidx.restype = None  # void

# encino_waves_spectral_height_at
LIB.encino_waves_spectral_height_at.argtypes = [
    ctypes.c_float,  # time
    ctypes.POINTER(ctypes.c_float),  # spectral_basis
    ctypes.POINTER(ctypes.c_float)  # out
]
LIB.encino_waves_spectral_height_at.restype = None  # void


# --- Immutable Data Classes ---

@dataclass(frozen=True)
class OceanParameters:
    """
    Immutable ocean simulation parameters.
    
    All parameters are validated and clamped to valid ranges upon creation.
    To modify parameters, use the `replace()` function or the `with_*` methods.
    """
    resolution: int = 512
    domain: float = 100.0
    gravity: float = 9.81
    surface_tension: float = 0.074
    density: float = 1000.0
    depth: float = 100.0
    wind_speed: float = 17.0
    fetch_km: float = 300.0
    swell: float = 0.0
    random_seed: int = 54321

    def __post_init__(self):
        """Validate parameters upon creation."""
        # Validate resolution is power of 2
        if not is_power_of_two(self.resolution):
            valid = get_valid_resolutions()
            raise ValueError(f"Resolution must be a power of 2. Valid values: {valid}")
        
        # Use object.__setattr__ to set values on frozen dataclass
        for name, param_range in PARAM_RANGES.items():
            if name == 'resolution':
                continue  # Already validated
            value = getattr(self, name)
            clamped = param_range.clamp(value)
            if clamped != value:
                print(f"Warning: {name} value {value} clamped to {clamped}")
                object.__setattr__(self, name, clamped)

    def with_resolution(self, resolution: int) -> 'OceanParameters':
        """Return new parameters with updated resolution."""
        return replace(self, resolution=resolution)

    def with_domain(self, domain: float) -> 'OceanParameters':
        """Return new parameters with updated domain."""
        return replace(self, domain=domain)

    def with_wind(self, wind_speed: float, fetch_km: Optional[float] = None) -> 'OceanParameters':
        """Return new parameters with updated wind conditions."""
        kwargs = {'wind_speed': wind_speed}
        if fetch_km is not None:
            kwargs['fetch_km'] = fetch_km
        return replace(self, **kwargs)

    def _to_c_struct(self) -> _OceanParamsStruct:
        """Convert to C structure (internal use)."""
        return _OceanParamsStruct(
            resolution=self.resolution,
            domain=self.domain,
            gravity=self.gravity,
            surface_tension=self.surface_tension,
            density=self.density,
            depth=self.depth,
            wind_speed=self.wind_speed,
            fetch_km=self.fetch_km,
            swell=self.swell,
            random_seed=self.random_seed
        )


class OceanPlan:
    """
    Ocean simulation plan wrapper.
    
    Holds the C struct and exposes the shape properties needed for arrays.
    All other fields are kept opaque in the C struct.
    """
    
    def __init__(self, c_struct: _OceanPlanStruct):
        """Initialize with C struct. Not intended for direct use."""
        self._c_struct = c_struct
    
    @property
    def N(self) -> int:
        """Grid resolution."""
        return self._c_struct.N
    
    @property 
    def size_i(self) -> int:
        """Size in i direction (for spectral arrays)."""
        return self._c_struct.size_i
        
    @property
    def size_j(self) -> int:
        """Size in j direction (for spectral arrays)."""
        return self._c_struct.size_j
        
    @property
    def count(self) -> int:
        """Total number of spectral elements."""
        return self._c_struct.count
    
    @property
    def dk(self) -> float:
        """Wave number spacing."""
        return self._c_struct.dk
    
    @property
    def max_k_mag(self) -> float:
        """Maximum wave number magnitude."""
        return self._c_struct.max_k_mag
    
    def _get_c_struct_pointer(self) -> ctypes.POINTER(_OceanPlanStruct):
        """Get pointer to the C struct for library calls."""
        return ctypes.byref(self._c_struct)


# --- Pure Functions ---

def create_ocean_plan(params: Optional[OceanParameters] = None) -> OceanPlan:
    """
    Create an ocean simulation plan from parameters.
    
    This is a pure function that computes all derived values needed for
    wave simulation from the input parameters.
    
    Args:
        params: Ocean parameters. If None, uses defaults.
        
    Returns:
        OceanPlan containing the computed C struct.
    """
    if params is None:
        params = OceanParameters()
    
    c_params = params._to_c_struct()
    c_plan = _OceanPlanStruct()
    
    result = LIB.encino_waves_create_ocean_plan(
        ctypes.byref(c_params),
        ctypes.byref(c_plan)
    )
    check_error(result, "create_ocean_plan")
    
    return OceanPlan(c_plan)


def compute_spectral_basis(plan: OceanPlan,
                          out: Optional[np.ndarray] = None,
                          max_threads: int = 1) -> np.ndarray:
    """
    Compute the spectral basis for wave generation.
    
    This is a pure function that computes the spectral basis from the ocean plan.
    
         Args:
         plan: Ocean simulation plan from create_ocean_plan().
         out: Optional output array of shape (size_j, size_i, 5). 
              If provided, writes directly into this array for efficiency.
              If None, allocates a new array.
         max_threads: Maximum OpenMP threads (0 = use all available, default: 0).
        
    Returns:
        Spectral basis array with shape (size_j, size_i, 5).
        The 5 channels are: amp_pos_real, amp_pos_imag, amp_neg_real, amp_neg_imag, omega
    """
    expected_shape = (plan.size_j, plan.size_i, 5)
    
    if out is None:
        out = np.zeros(expected_shape, dtype=np.float32)
    else:
        _validate_array(out, expected_shape, np.float32, "output")
    
    # Prepare for C call
    shape_array = (ctypes.c_int * 3)(*expected_shape)
    
    result = LIB.encino_waves_spectral_basis_omp(
        plan._get_c_struct_pointer(),
        max_threads,
        3,  # rank
        shape_array,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    check_error(result, "spectral_basis_omp")
    
    return out


def compute_spectral_height(plan: OceanPlan,
                           time: float,
                           spectral_basis: np.ndarray,
                           out: Optional[np.ndarray] = None,
                           max_threads: int = 1) -> np.ndarray:
    """
    Compute wave heights from spectral basis at a given time.
    
    This is a pure function that propagates the spectral basis to a specific time.
    
    Args:
        plan: Ocean simulation plan from create_ocean_plan().
        time: Simulation time in seconds.
        spectral_basis: Input spectral basis array from compute_spectral_basis().
        out: Optional output array. Supports two formats:
             - (size_j, size_i) complex64: Natural for FFT input
             - (size_j, size_i, 2) float32: [real, imag] components
             If None, creates complex64 array (default for FFT compatibility).
        max_threads: Maximum OpenMP threads (0 = use all available, default: 0).
        
    Returns:
        Spectral height array in the same format as provided (or complex64 if none).
        This is typically passed to an inverse FFT to get spatial heights.
    """
    # Validate input
    expected_in_shape = (plan.size_j, plan.size_i, 5)
    _validate_array(spectral_basis, expected_in_shape, np.float32, "spectral_basis")
    
    # Determine output format and prepare array
    if out is None:
        # Default to complex64 for FFT compatibility
        out = np.zeros((plan.size_j, plan.size_i), dtype=np.complex64)
        out_for_c, return_complex = _prepare_complex_array_for_c(out)
    else:
        out_for_c, return_complex = _prepare_complex_array_for_c(out)
    
    # Prepare for C call
    in_shape = (ctypes.c_int * 3)(*expected_in_shape)
    out_shape = (ctypes.c_int * 3)(plan.size_j, plan.size_i, 2)
    
    result = LIB.encino_waves_spectral_height_omp(
        plan._get_c_struct_pointer(),
        ctypes.c_float(time),
        max_threads,  # max_threads parameter
        3,  # in_rank
        in_shape,
        spectral_basis.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        3,  # out_rank
        out_shape,
        out_for_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    check_error(result, "spectral_height_omp")
    
    return out


def compute_spatial_heights(plan: OceanPlan,
                           spectral_height: np.ndarray,
                           out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert spectral height data to spatial height field using inverse FFT.
    
    This is a pure function that transforms frequency-domain wave heights
    to spatial-domain heights using numpy.fft.irfft2.
    
    Args:
        plan: Ocean simulation plan from create_ocean_plan().
        spectral_height: Spectral height data from compute_spectral_height().
                        Supports two formats:
                        - (size_j, size_i) complex64: Default complex format
                        - (size_j, size_i, 2) float32: [real, imag] components
        out: Optional output array of shape (N, N) float32.
             If provided, writes directly into this array for efficiency.
             If None, allocates a new array.
        
    Returns:
        Real spatial height field array with shape (N, N) float32.
        This represents the wave heights at each grid point.
    """
    # Convert input to complex64 format for FFT
    if spectral_height.dtype == np.complex64 and len(spectral_height.shape) == 2:
        # Already in complex64 format
        expected_shape = (plan.size_j, plan.size_i)
        if spectral_height.shape != expected_shape:
            raise ValueError(f"spectral_height must have shape {expected_shape}, got {spectral_height.shape}")
        complex_data = spectral_height
        
    elif spectral_height.dtype == np.float32 and len(spectral_height.shape) == 3 and spectral_height.shape[2] == 2:
        # Float32 format with [real, imag] components
        expected_shape = (plan.size_j, plan.size_i, 2)
        if spectral_height.shape != expected_shape:
            raise ValueError(f"spectral_height must have shape {expected_shape}, got {spectral_height.shape}")
        # Convert to complex64 view
        complex_data = spectral_height.view(dtype=np.complex64).reshape(plan.size_j, plan.size_i)
        
    else:
        raise ValueError(
            f"spectral_height must be either (size_j, size_i) complex64 or (size_j, size_i, 2) float32, "
            f"got shape {spectral_height.shape} with dtype {spectral_height.dtype}"
        )
    
    if not spectral_height.flags['C_CONTIGUOUS']:
        raise ValueError("spectral_height must be C-contiguous")
    
    # Prepare output array
    spatial_shape = (plan.N, plan.N)
    if out is None:
        out = np.zeros(spatial_shape, dtype=np.float32)
    else:
        _validate_array(out, spatial_shape, np.float32, "output")
    
    # Perform inverse FFT to get spatial heights
    # irfft2 expects the input to be the result of rfft2, which has shape (N, N//2+1)
    # The spectral_height should already be in this format from the C library
    # Use norm=None to ensure no normalization - user handles normalization internally
    spatial_heights = np.fft.irfft2(complex_data, s=(plan.N, plan.N), norm=None)
    
    # Copy result to output array (with type conversion if needed)
    out[:] = spatial_heights.astype(np.float32)
    
    return out


# --- Helper Functions ---

def _validate_array(arr: np.ndarray, 
                   expected_shape: Tuple[int, ...], 
                   expected_dtype: np.dtype,
                   name: str) -> None:
    """Validate numpy array properties."""
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    if arr.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {arr.dtype}")
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError(f"{name} must be C-contiguous")


def _prepare_complex_array_for_c(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Prepare array for C library call, handling both complex64 and float32 formats.
    
    Args:
        arr: Either (size_j, size_i) complex64 or (size_j, size_i, 2) float32
        
    Returns:
        Tuple of (array_for_c, is_complex_format)
        - array_for_c: float32 array with shape (size_j, size_i, 2) 
        - is_complex_format: True if input was complex64
    """
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError("Array must be C-contiguous")
    
    if arr.dtype == np.complex64 and len(arr.shape) == 2:
        # Convert complex64 to float32 view with extra dimension
        # complex64 is stored as 2 consecutive float32s in memory
        float_view = arr.view(dtype=np.float32)
        # Reshape from (size_j, size_i*2) to (size_j, size_i, 2)
        size_j, size_i = arr.shape
        array_for_c = float_view.reshape(size_j, size_i, 2)
        return array_for_c, True
        
    elif arr.dtype == np.float32 and len(arr.shape) == 3 and arr.shape[2] == 2:
        # Already in the right format
        return arr, False
        
    else:
        raise ValueError(
            f"Array must be either (size_j, size_i) complex64 or (size_j, size_i, 2) float32, "
            f"got shape {arr.shape} with dtype {arr.dtype}"
        )


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def get_valid_resolutions() -> list[int]:
    """Get list of valid resolutions within the allowed range."""
    min_res = int(PARAM_RANGES['resolution'].min)
    max_res = int(PARAM_RANGES['resolution'].max)
    return [2**i for i in range(2, 14) if min_res <= 2**i <= max_res]


def get_default_parameters() -> OceanParameters:
    """Get ocean parameters with all defaults."""
    return OceanParameters()


def create_complex_spectral_array(plan: OceanPlan) -> np.ndarray:
    """Create a complex64 array suitable for spectral height output."""
    return np.zeros((plan.size_j, plan.size_i), dtype=np.complex64)


def create_float_spectral_array(plan: OceanPlan) -> np.ndarray:
    """Create a float32 array suitable for spectral height output."""
    return np.zeros((plan.size_j, plan.size_i, 2), dtype=np.float32)


def create_spatial_heights_array(plan: OceanPlan) -> np.ndarray:
    """Create a float32 array suitable for spatial height output."""
    return np.zeros((plan.N, plan.N), dtype=np.float32)


# --- Example Usage Functions ---

def create_default_ocean_plan() -> OceanPlan:
    """Create an ocean plan with default parameters."""
    return create_ocean_plan(OceanParameters())


def simulate_ocean_surface(params: OceanParameters,
                          time: float,
                          spectral_basis_out: Optional[np.ndarray] = None,
                          spectral_height_out: Optional[np.ndarray] = None,
                          spatial_heights_out: Optional[np.ndarray] = None) -> Tuple[OceanPlan, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete ocean simulation pipeline including spatial height computation.
    
    This demonstrates the functional composition of the library functions.
    
    Args:
        params: Ocean parameters
        time: Simulation time
        spectral_basis_out: Optional array to reuse for spectral basis (shape: size_j, size_i, 5)
        spectral_height_out: Optional array to reuse for spectral height. Supports:
                            - (size_j, size_i) complex64: Default, FFT-ready
                            - (size_j, size_i, 2) float32: [real, imag] components
        spatial_heights_out: Optional array to reuse for spatial heights (shape: N, N)
        
    Returns:
        Tuple of (plan, spectral_basis, spectral_height, spatial_heights)
        - spectral_height: complex64 by default, or matches provided format
        - spatial_heights: float32 array with real wave heights at each grid point
    """
    # Each step is a pure function
    plan = create_ocean_plan(params)
    spectral_basis = compute_spectral_basis(plan, out=spectral_basis_out)
    spectral_height = compute_spectral_height(plan, time, spectral_basis, out=spectral_height_out)
    spatial_heights = compute_spatial_heights(plan, spectral_height, out=spatial_heights_out)
    
    return plan, spectral_basis, spectral_height, spatial_heights
