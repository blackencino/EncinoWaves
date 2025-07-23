# encino_waves_torch.py
#
# ---------------------------------------------------------------------------------
# Torch re-implementation of the Encino Waves "initial_state'' kernel.
# This version is refactored to avoid torch.complex in JIT-scripted passes.
# ---------------------------------------------------------------------------------
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch

# 32-bit unsigned integer mask for proper wrapping
U32_MASK: int = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# HASHGRID HELPERS IN PYTHON
# ---------------------------------------------------------------------------


def _py_hash_state(state: int) -> int:
    """
    Python equivalent of the HASH_STATE macro.

    C++ Macro:
    #define HASH_STATE(STATE) ((STATE)*747796405U + 2891336453U)
    """
    # Perform the multiplication and addition, then apply the mask
    # to simulate the 32-bit unsigned integer wrap-around.
    result = (state * 747796405) + 2891336453
    return result & U32_MASK


def _py_u32_from_state(state: int) -> int:
    """
    Python equivalent of the __device__ function word_from_state.

    C++ Function:
    __device__ __forceinline__ uint32_t word_from_state(uint32_t const state) {
        auto const word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
        return (word >> 22U) ^ word;
    }
    """
    # Ensure the input state is a valid 32-bit unsigned integer
    state &= U32_MASK

    # Calculate the dynamic shift amount
    shift_amount = (state >> 28) + 4

    # Perform the first set of operations, masking after each step
    xored = ((state >> shift_amount) ^ state) & U32_MASK
    word = (xored * 277803737) & U32_MASK

    # Perform the final operations
    final_result = ((word >> 22) ^ word) & U32_MASK
    return final_result


def _py_bit_cast_f32_from_u32(u32_val: int) -> float:
    """
    Python equivalent of the BIT_CAST_F32_FROM_U32 macro.
    This reinterprets the bits of a 32-bit integer as a 32-bit float.

    C++ Macro:
    #define BIT_CAST_F32_FROM_U32(U32) __int_as_float((U32))
    """
    # 'I' format specifier is for a 4-byte unsigned integer (uint32_t)
    # 'f' format specifier is for a 4-byte single-precision float
    packed_bytes = struct.pack("I", u32_val & U32_MASK)
    # The result of unpack is a tuple, so we take the first element.
    return struct.unpack("f", packed_bytes)[0]


def _py_norm_f32_from_u32(u32_val: int) -> float:
    """
    Python equivalent of the NORM_F32_FROM_U32 macro.
    This generates a pseudo-random float in the range [0.0, 1.0].

    C++ Macro:
    #define NORM_F32_FROM_U32(U32) \
        (BIT_CAST_F32_FROM_U32((U32) >> 9 | 0x3f800000) - 1.0f)
    """
    # Perform the bitwise operations on the integer
    intermediate_u32 = (u32_val >> 9) | 0x3F800000

    # Bit-cast the result to a float and subtract 1.0
    return _py_bit_cast_f32_from_u32(intermediate_u32) - 1.0


# ---------------------------------------------------------------------------
# HASHGRID HELPERS IN TORCHSCRIPT
# ---------------------------------------------------------------------------


@torch.jit.script
def _torch_hash_state(state: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to: ((STATE)*747796405U + 2891336453U)
    We use int64 for intermediate math to prevent signed 32-bit overflow.
    """
    # Promote to int64, perform math, then mask back to 32-bit.
    state_64 = state.to(torch.int64)
    result_64 = state_64 * 747796405 + 2891336453
    # The final result fits in a 32-bit signed int, so we can cast back.
    return result_64.bitwise_and(0xFFFFFFFF).to(torch.int32)


@torch.jit.script
def _torch_u32_from_state(state: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to the PCG XSH-RS step.
    Handles dynamic shifts and uses int64 for intermediate multiplication.
    """
    # PyTorch's >> on a signed int is an arithmetic shift, but since the
    # value of `state >> 28` is small, it works out correctly here.
    shift = state.bitwise_right_shift(28).add(4)
    xored = state.bitwise_right_shift(shift).bitwise_xor(state)

    # Use int64 for the multiplication to avoid signed overflow
    word_64 = xored.to(torch.int64) * 277803737
    word = word_64.bitwise_and(0xFFFFFFFF).to(torch.int32)

    # Final step
    return word.bitwise_right_shift(22).bitwise_xor(word)


@torch.jit.script
def _torch_norm_f32_from_u32(u32: torch.Tensor) -> torch.Tensor:
    """
    Generates a float in [0.0, 1.0) from a 32-bit unsigned integer.
    Uses the upper 23 bits as a uniform distribution.
    """
    # Extract the upper 23 bits by shifting right by 9
    # This gives us values in [0, 2^23 - 1]
    upper_bits = u32.bitwise_right_shift(9) & 0x007FFFFF

    # Convert to float and normalize to [0.0, 1.0)
    # 2^23 = 8388608.0
    return upper_bits.to(torch.float32) / 8388608.0

@torch.jit.script
def _torch_encino_waves_spectral_wave_numbers_kernel(
    dk: float, N: int, device: torch.device
) -> torch.Tensor:
    ki = N * dk * torch.fft.rfftfreq(N, dtype=torch.float32, device=device)
    kj = N * dk * torch.fft.fftfreq(N, dtype=torch.float32, device=device)
    ki, kj = torch.meshgrid(ki, kj, indexing="xy")
    ki = ki.flatten()
    kj = kj.flatten()
    k_mag = torch.maximum(
        torch.tensor(dk, dtype=torch.float32, device=device),
        torch.sqrt(ki**2 + kj**2),
    )

    return torch.stack((ki, kj, k_mag), dim=1)


@torch.jit.script
def _torch_encino_waves_spectral_basis_kernel(
    dk: float,
    N: int,
    gravity: float,
    depth: float,
    surface_tens_over_rho: float,
    wind_speed: float,
    swell: float,
    peak_omega: float,
    TMA_alpha: float,
    TMA_gamma: float,
    random_seed: int,
    device: torch.device,
) -> torch.Tensor:
    wave_numbers = _torch_encino_waves_spectral_wave_numbers_kernel(dk, N, device)

    ki = wave_numbers[:, 0]
    kj = wave_numbers[:, 1]
    k_mag = wave_numbers[:, 2]

    dk2 = dk * dk

    # -----------------------------------------------------------------------
    # Dispersion (includes capillary term)
    # -----------------------------------------------------------------------
    hk = max(0.01, depth) * k_mag
    tanh_hk = torch.tanh(hk)
    cosh_hk = torch.cosh(hk)

    k2 = k_mag * k_mag
    k2s = k2 * surface_tens_over_rho
    gpk2s = gravity + k2s  # g + sigma*k^2/rho

    omega = torch.sqrt(torch.abs(k_mag * gpk2s * tanh_hk))

    numer = ((gpk2s + 2.0 * k2s) * tanh_hk) + ((hk * gpk2s) / (cosh_hk * cosh_hk))
    domega_dk = torch.abs(numer) / (2.0 * omega)

    # -----------------------------------------------------------------------
    # Directional spreading (stack pos/neg into dim -1)
    # -----------------------------------------------------------------------
    # theta+ and theta-
    theta = torch.stack((torch.atan2(-kj, ki), torch.atan2(kj, -ki)), dim=1)

    delta_s_multiplier = (dk2 * domega_dk / k_mag).unsqueeze(1)

    inv_omega_peak_ratio = peak_omega / omega

    # Handle positive vs negative swell (swell is scalar)
    swell_step: float = 1.0 if swell > 0.0 else 0.0
    shape_bias = swell_step * 16.1 * torch.tanh(inv_omega_peak_ratio) * swell * swell

    omega_peak_ratio = omega / peak_omega
    is_fast = omega > peak_omega

    shape_gain = torch.where(
        is_fast, torch.tensor(9.77, dtype=torch.float32), torch.tensor(6.97, dtype=torch.float32)
    )

    wind_speed_over_cel = wind_speed * peak_omega / gravity
    shape_exp = torch.where(
        is_fast,
        torch.tensor(
            -2.33 - 1.45 * (wind_speed_over_cel - 1.17), dtype=torch.float32, device=device
        ),
        torch.tensor(4.06, dtype=torch.float32, device=device),
    )

    shape = shape_bias + shape_gain * torch.pow(omega_peak_ratio, shape_exp)

    factor_a = torch.pow(2.0, (2.0 * shape) - 1.0) / math.pi
    factor_b = torch.exp(2.0 * torch.lgamma(shape + 1.0) - torch.lgamma((2.0 * shape) + 1.0))

    factor_c = torch.abs(torch.cos(theta / 2.0))
    factor_c *= factor_c

    factor = (factor_a * factor_b).unsqueeze(1) * factor_c

    # Swell interpolation factor (swell is scalar)
    swell_interp = max(0.0, -swell)
    dspread = delta_s_multiplier * (
        (1.0 - swell_interp) * factor + (swell_interp * (1.0 / math.tau))
    )



    # -----------------------------------------------------------------------
    # JONSWAP / TMA spectrum
    # -----------------------------------------------------------------------
    TMA_sigma = torch.where(
        is_fast,
        torch.tensor(0.09, dtype=torch.float32, device=device),
        torch.tensor(0.07, dtype=torch.float32, device=device),
    )
    p = (omega_peak_ratio - 1.0) / TMA_sigma
    peak_sharpen = torch.pow(TMA_gamma, torch.exp(-(p * p) / 2.0))

    wh = omega * math.sqrt(depth / gravity)
    kitaigorodskii_depth = torch.sigmoid(3.6 * (wh - 1.125))

    gain = (TMA_alpha * gravity * gravity) / torch.pow(omega, 5.0)
    alpha_beta_spec = gain * torch.exp(-1.25 * torch.pow(inv_omega_peak_ratio, 4.0))

    spect_eng = kitaigorodskii_depth * peak_sharpen * alpha_beta_spec

    delta_s = dspread * spect_eng.unsqueeze(1)

    # -----------------------------------------------------------------------
    #   P C G   R A N D O M   (bit-exact replication)
    # -----------------------------------------------------------------------
    ki_int = (ki * 10000.0).to(torch.int32)
    kj_int = (kj * 10000.0).to(torch.int32)

    state = torch.full_like(ki_int, random_seed, dtype=torch.int32)
    state = _torch_hash_state(state)
    state = _torch_hash_state(state + kj_int)
    state = _torch_hash_state(state + ki_int)

    u0 = _torch_u32_from_state(state)
    state = _torch_hash_state(state)
    u1 = _torch_u32_from_state(state)
    state = _torch_hash_state(state)
    u2 = _torch_u32_from_state(state)
    state = _torch_hash_state(state)
    u3 = _torch_u32_from_state(state)

    # -----------------------------------------------------------------------
    #   Amplitude (Box-Muller) and Phase
    # -----------------------------------------------------------------------
    r = torch.sqrt(-2.0 * torch.log(torch.clamp(_torch_norm_f32_from_u32(u0), 1.1920929e-07, 1.0)))
    theta_rand = math.tau * _torch_norm_f32_from_u32(u1)

    amp = torch.stack((r * torch.cos(theta_rand), r * torch.sin(theta_rand)), dim=1)
    amp = amp * torch.sqrt(torch.abs(delta_s * 2.0))

    phase = math.tau * torch.stack((_torch_norm_f32_from_u32(u2), _torch_norm_f32_from_u32(u3)), dim=1)

    # h_spec = amp * exp(i*phase) = amp * (cos(phase) + i*sin(phase))
    # We store this as a (..., 2) float tensor instead of a complex one.
    h_spec_real = amp * torch.cos(phase)
    h_spec_imag = amp * torch.sin(phase)


    # h_spec: (count, 2, 2) float32
    # Rearrange to (count, 4): [hpos_real, hpos_imag, hneg_real, hneg_imag]
    hpos_real = h_spec_real[:, 0]
    hpos_imag = h_spec_imag[:, 0]
    hneg_real = h_spec_real[:, 1]
    hneg_imag = h_spec_imag[:, 1]

    out_tensor = torch.stack([hpos_real, hpos_imag, hneg_real, hneg_imag, omega], dim=1)  # (count, 5)

    # Zero-wave special-case
    mask_zero = (ki < dk) & (kj < dk)
    out_tensor = out_tensor.masked_fill(mask_zero.unsqueeze(1), 0.0)

    # Reshape to (size_j, size_i, 5) expected output shape
    size_j = N
    size_i = (N // 2) + 1
    out_tensor = out_tensor.reshape(size_j, size_i, 5)

    return out_tensor

def encino_waves_compute_spectral_basis_torch(
    plan,
    out: Optional[np.ndarray] = None,
    max_threads: int = 1,
) -> np.ndarray:
    """
    Pure torch implementation of the Encino Waves spectral basis computation.

    Args:
        plan: OceanPlan object with simulation parameters.
        out: Optional output array of shape (size_j, size_i, 5), float32.
        max_threads: Ignored (for API compatibility).

    Returns:
        np.ndarray of shape (size_j, size_i, 5), float32.
        Channels: amp_pos_real, amp_pos_imag, amp_neg_real, amp_neg_imag, omega
    """
    import torch
    import numpy as np

    N = plan.N
    size_j = plan.size_j
    size_i = plan.size_i
    dk = plan.dk
    random_seed = plan.random_seed

    # Create k-space grid (match C++/numpy convention)
    kj = torch.fft.fftfreq(N, d=plan.domain / N, device='cpu', dtype=torch.float32) * 2 * np.pi
    ki = torch.fft.rfftfreq(N, d=plan.domain / N, device='cpu', dtype=torch.float32) * 2 * np.pi
    kj_grid, ki_grid = torch.meshgrid(kj, ki, indexing='ij')  # (size_j, size_i)

    # Flatten for vectorized computation
    ki_flat = ki_grid.reshape(-1)
    kj_flat = kj_grid.reshape(-1)

    # Call the kernel (inlined from above, but as a function)
    h_spec, omega = _encino_waves_spectral_basis_kernel(
        ki_flat, kj_flat, plan, random_seed
    )  # h_spec: (count, 2, 2), omega: (count,)

    # Reshape to (size_j, size_i, ...)
    h_spec = h_spec.reshape(size_j, size_i, 2, 2)
    omega = omega.reshape(size_j, size_i)

    # Compose output: [amp_pos_real, amp_pos_imag, amp_neg_real, amp_neg_imag, omega]
    out_arr = np.empty((size_j, size_i, 5), dtype=np.float32) if out is None else out
    out_arr[..., 0] = h_spec[..., 0, 0].numpy()
    out_arr[..., 1] = h_spec[..., 0, 1].numpy()
    out_arr[..., 2] = h_spec[..., 1, 0].numpy()
    out_arr[..., 3] = h_spec[..., 1, 1].numpy()
    out_arr[..., 4] = omega.numpy()
    return out_arr

@torch.jit.script
def _encino_waves_spectral_basis_kernel(
    ki: torch.Tensor,  # (count,)
    kj: torch.Tensor,  # (count,)
    plan,  # OceanPlan, but only fields accessed
    random_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    # --- Parameters from plan ---
    dk = plan.dk
    gravity = plan.gravity
    depth = plan.depth
    wind_speed = plan.wind_speed
    fetch_km = plan.fetch_km
    surface_tension = plan.surface_tension
    density = plan.density
    swell = plan.swell

    # --- Compute k magnitude ---
    k_mag = torch.sqrt(ki ** 2 + kj ** 2)
    hk = depth * k_mag
    sigma_over_rho = surface_tension / density

    # --- Dispersion relation (capillary) ---
    gpk2s = gravity + sigma_over_rho * k_mag ** 2
    omega = torch.sqrt(torch.abs(k_mag * gpk2s * torch.tanh(hk)))
    # DC component: omega=0
    mask_zero = (ki.abs() < dk) & (kj.abs() < dk)
    omega = omega.masked_fill(mask_zero, 0.0)

    # --- JONSWAP spectrum (simplified, see C++) ---
    # For brevity, use fixed values for tma_alpha, tma_gamma, etc.
    tma_alpha = 0.0081
    tma_gamma = 3.3
    peak_omega = 2 * math.pi * 3.5 * (gravity / wind_speed) * (gravity * fetch_km * 1000.0 / wind_speed ** 2) ** -0.33
    sigma = torch.where(omega <= peak_omega, 0.07, 0.09)
    peak_sharpen = torch.pow(
        tma_gamma,
        torch.exp(-((omega - peak_omega) / (sigma * peak_omega)) ** 2 / 2.0)
    )
    alpha_beta_spec = (tma_alpha * gravity ** 2 / omega.clamp_min(1e-6) ** 5) * torch.exp(
        -1.25 * (peak_omega / omega.clamp_min(1e-6)) ** 4
    )
    kitaigorodskii_depth = 0.5 + 0.5 * torch.tanh(1.8 * (omega * 1.0 - 1.125))
    spectrum = peak_sharpen * alpha_beta_spec * kitaigorodskii_depth

    # --- Directional spreading (simplified) ---
    dspread = torch.ones_like(spectrum)

    # --- Spectral energy ---
    delta_s = dspread * spectrum.unsqueeze(0)

    # --- PCG random (bit-exact) ---
    ki_int = (ki * 10000.0).to(torch.int32)
    kj_int = (kj * 10000.0).to(torch.int32)
    state = torch.full_like(ki_int, random_seed, dtype=torch.int32)
    state = _hash_state(state)
    state = _hash_state(state + kj_int)
    state = _hash_state(state + ki_int)
    u0 = _u32_from_state(state)
    state = _hash_state(state)
    u1 = _u32_from_state(state)
    state = _hash_state(state)
    u2 = _u32_from_state(state)
    state = _hash_state(state)
    u3 = _u32_from_state(state)

    # --- Amplitude (Box-Muller) and Phase ---
    r = torch.sqrt(-2.0 * torch.log(torch.clamp(_norm_f32_from_u32(u0), 1.1920929e-07, 1.0)))
    theta_rand = math.tau * _norm_f32_from_u32(u1)
    amp = torch.stack((r * torch.cos(theta_rand), r * torch.sin(theta_rand)), dim=1)
    amp = amp * torch.sqrt(torch.abs(delta_s.squeeze(0) * 2.0))
    phase = math.tau * torch.stack((_norm_f32_from_u32(u2), _norm_f32_from_u32(u3)), dim=1)
    h_spec_real = amp * torch.cos(phase)
    h_spec_imag = amp * torch.sin(phase)
    h_spec = torch.stack((h_spec_real, h_spec_imag), dim=-1)  # (count, 2, 2)

    # Zero-wave special-case
    h_spec = h_spec.masked_fill(mask_zero.unsqueeze(1).unsqueeze(2), 0.0)

    return h_spec, omega


def compute_spectral_basis_torch(plan: OceanPlan,
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

# ---------------------------------------------------------
@torch.jit.script
def _encino_waves_pre_propagate_kernel(
    wave_numbers: torch.Tensor,
    h_spec: torch.Tensor,  # (count, 2, 2) float32 (pos/neg, real/imag)
    omegas: torch.Tensor,  # (count,) float32
    N: int,
    time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # h_spec is (count, pos/neg, real/imag)
    h0_pos_real, h0_pos_imag = h_spec[:, 0, 0], h_spec[:, 0, 1]
    h0_neg_real, h0_neg_imag = h_spec[:, 1, 0], h_spec[:, 1, 1]

    omega_t = omegas * time
    c_omega_t = torch.cos(omega_t)
    s_omega_t = torch.sin(omega_t)

    # phase_pos = exp(-i*omega*t) -> (c, -s)
    phase_pos_real, phase_pos_imag = c_omega_t, -s_omega_t
    # phase_neg = exp(+i*omega*t) -> (c, s)
    phase_neg_real, phase_neg_imag = c_omega_t, s_omega_t

    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    # h_hat_pos = h0_pos * phase_pos
    h_hat_pos_real = h0_pos_real * phase_pos_real - h0_pos_imag * phase_pos_imag
    h_hat_pos_imag = h0_pos_real * phase_pos_imag + h0_pos_imag * phase_pos_real
    # h_hat_neg = h0_neg * phase_neg
    h_hat_neg_real = h0_neg_real * phase_neg_real - h0_neg_imag * phase_neg_imag
    h_hat_neg_imag = h0_neg_real * phase_neg_imag + h0_neg_imag * phase_neg_real

    # h_hat_flat(k,t) = h_hat_pos + h_hat_neg
    h_hat_flat_real = h_hat_pos_real + h_hat_neg_real
    h_hat_flat_imag = h_hat_pos_imag + h_hat_neg_imag

    # Stack real/imag parts into the last dimension
    h_hat_flat = torch.stack((h_hat_flat_real, h_hat_flat_imag), dim=-1)

    width = (N // 2) + 1
    expected_count = N * width
    assert (
        h_hat_flat.shape[0] == expected_count
    ), f"h_hat_flat must have shape ({expected_count}, 2), got {h_hat_flat.shape}"

    # Reshape to (N, width, 2) so irfft2 can receive an (N, width) complex view
    h_hat = h_hat_flat.view(N, width, 2)
    h_hat_real = h_hat[..., 0]
    h_hat_imag = h_hat[..., 1]

    # ------------ spectral differentiation factors ----------------------------
    ki = wave_numbers[:, 0]
    kj = wave_numbers[:, 1]
    k_mag = wave_numbers[:, 2]

    # Multiplication factors (all real)
    dii_fac = (ki * ki) / k_mag
    djj_fac = (kj * kj) / k_mag
    dij_fac = (ki * kj) / k_mag
    di_fac = -ki / k_mag
    dj_fac = -kj / k_mag

    # Reshape factors to match h_hat's shape (N, width)
    di_fac = di_fac.view(N, width)
    dj_fac = dj_fac.view(N, width)
    dii_fac = dii_fac.view(N, width)
    djj_fac = djj_fac.view(N, width)
    dij_fac = dij_fac.view(N, width)

    # --- Spectral derivatives as floatx2 tensors ---
    # Multiplication by i: (a+bi)*i = -b + ai
    h_hat_times_i_real = -h_hat_imag
    h_hat_times_i_imag = h_hat_real

    # di_spec = h_hat * i * di_fac
    di_spec_real = h_hat_times_i_real * di_fac
    di_spec_imag = h_hat_times_i_imag * di_fac
    di_spec = torch.stack((di_spec_real, di_spec_imag), dim=-1)

    # dj_spec = h_hat * i * dj_fac
    dj_spec_real = h_hat_times_i_real * dj_fac
    dj_spec_imag = h_hat_times_i_imag * dj_fac
    dj_spec = torch.stack((dj_spec_real, dj_spec_imag), dim=-1)

    # Multiplication by real factors
    dii_spec_real = h_hat_real * dii_fac
    dii_spec_imag = h_hat_imag * dii_fac
    dii_spec = torch.stack((dii_spec_real, dii_spec_imag), dim=-1)

    djj_spec_real = h_hat_real * djj_fac
    djj_spec_imag = h_hat_imag * djj_fac
    djj_spec = torch.stack((djj_spec_real, djj_spec_imag), dim=-1)

    dij_spec_real = h_hat_real * dij_fac
    dij_spec_imag = h_hat_imag * dij_fac
    dij_spec = torch.stack((dij_spec_real, dij_spec_imag), dim=-1)

    return h_hat, di_spec, dj_spec, dii_spec, djj_spec, dij_spec


# ---------------------------------------------------------
def _encino_waves_ifft_propagate_kernel(
    h_hat: torch.Tensor,
    di_spec: torch.Tensor,
    dj_spec: torch.Tensor,
    dii_spec: torch.Tensor,
    djj_spec: torch.Tensor,
    dij_spec: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # ------------ inverse FFTs  (cuFFT under the hood) ------------------------
    # Convert floatx2 tensors back to complex for the FFT library.
    # torch.view_as_complex combines the last dimension of size 2 into a complex number.
    h_hat_complex = torch.view_as_complex(h_hat)
    di_spec_complex = torch.view_as_complex(di_spec)
    dj_spec_complex = torch.view_as_complex(dj_spec)
    dii_spec_complex = torch.view_as_complex(dii_spec)
    djj_spec_complex = torch.view_as_complex(djj_spec)
    dij_spec_complex = torch.view_as_complex(dij_spec)

    # torch.fft.irfft2 expects shape (N,width) and produces (N,N)
    norm = "ortho"
    # gain = 1.0 / math.sqrt(float(N))
    gain = 1.0
    height = torch.fft.irfft2(h_hat_complex, s=(N, N), norm=norm) * gain
    di = torch.fft.irfft2(di_spec_complex, s=(N, N), norm=norm) * gain
    dj = torch.fft.irfft2(dj_spec_complex, s=(N, N), norm=norm) * gain
    dii = torch.fft.irfft2(dii_spec_complex, s=(N, N), norm=norm) * gain
    djj = torch.fft.irfft2(djj_spec_complex, s=(N, N), norm=norm) * gain
    dij = torch.fft.irfft2(dij_spec_complex, s=(N, N), norm=norm) * gain

    return height, di, dj, dii, djj, dij


# ---------------------------------------------------------
@torch.jit.script
def _encino_waves_post_propagate_kernel(
    height: torch.Tensor,
    di: torch.Tensor,
    dj: torch.Tensor,
    dii: torch.Tensor,
    djj: torch.Tensor,
    dij: torch.Tensor,
    amplitude_gain: float,
    pinch: float,
) -> torch.Tensor:
    # ------------ minimum eigenvalue of Hessian ------------------------------
    # lambda_min = (1/2)*[(dxx+dyy) - sqrt((dxx-dyy)^2 + 4*dxy^2)]
    trace = dii + djj
    diff = dii - djj
    radicand = diff * diff + 4.0 * dij * dij
    lambda_min = 0.5 * (trace - torch.sqrt(radicand.clamp_min(0.0)))

    # ------------ pinch + amplitude gain ------------
    # the original encinowaves negates di,dj,height after gain; we follow that:
    di = -amplitude_gain * (pinch * di)
    dj = -amplitude_gain * (pinch * dj)
    height = -amplitude_gain * height

    # lambda_min is already signed correctly because we negated height
    return torch.stack((height, di, dj, lambda_min), dim=0)


@torch.jit.script
def _encino_waves_undisplaced_height_field_kernel(
    domain: float, N: int, up: str, device: torch.device
) -> torch.Tensor:
    # Create a NxNx3 tensor of float32 dtype, representing the
    # flat ocean surface without any waves. if up is "z", the z-axis points
    # up, otherwise if up is "y", the y-axis points up. The domain is the
    # length, in meters, of the domain. N is the number of grid points in
    # each dimension.
    # The lowest wave number is _tau/domain, and the highest wave number is
    # (N-1) * _tau/domain.
    # These are essentially the mesh vertices if there were no waves.

    dL = domain / N
    # Create coordinates so that the grid is centered at the origin.
    # The first point is at -domain/2 + dL/2, the last at +domain/2 - dL/2
    coords_1d = (torch.arange(N, dtype=torch.float32, device=device) + 0.5) * dL - (domain / 2)
    X, Y = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    if up == "z":
        Z = torch.zeros_like(X)
        result = torch.stack((X, Y, Z), dim=-1)
    elif up == "y":
        Y_up = torch.zeros_like(X)
        result = torch.stack((X, Y_up, Y), dim=-1)
    else:
        raise RuntimeError("Unknown up axis: " + up)
    return result


@torch.jit.script
def _encino_waves_displaced_height_field_kernel(
    undisplaced: torch.Tensor, propagated: torch.Tensor, up: str
) -> torch.Tensor:
    disp_up = propagated[0]
    disp_rh_axis_0 = propagated[1]
    disp_rh_axis_1 = propagated[2]

    if up == "z":
        return undisplaced + torch.stack((disp_rh_axis_0, disp_rh_axis_1, disp_up), dim=2)
    elif up == "y":
        # in right-handed coordinate system, z cross x is up, so z is axis 0, x is axis 1
        return undisplaced + torch.stack((disp_rh_axis_1, disp_up, disp_rh_axis_0), dim=2)
    else:
        raise RuntimeError("Unknown up axis: " + up)


# ---------------------------------------------------------------------------
#   F R O N T E N D  I N I T I A L   S T A T E
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncinoWavesParameters:
    resolution_power_of_two: int = 10
    domain: float = 500.0

    gravity: float = 9.81
    surface_tension: float = 0.074
    density: float = 1_000.0
    depth: float = 100.0

    wind_speed: float = 17.0
    fetch_km: float = 300.0
    swell: float = 0.0

    amplitude_gain: float = 1.0
    pinch: float = 1.0

    random_seed: int = 54321

    @property
    def safe_domain(self) -> float:
        return max(0.1, min(100_000.0, abs(self.domain)))

    @property
    def dk(self) -> float:
        return math.tau / self.safe_domain

    @property
    def N(self) -> int:
        return 1 << max(6, min(14, abs(self.resolution_power_of_two)))

    @property
    def spectral_size_i(self) -> int:
        return self.N

    @property
    def spectral_size_j(self) -> int:
        return (self.N // 2) + 1

    @property
    def spectral_size(self) -> int:
        return self.spectral_size_i * self.spectral_size_j

    @property
    def spectral_count(self) -> int:
        return self.spectral_size * 2


def encino_waves_spectral_wave_numbers(
    P: EncinoWavesParameters,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    device = device if isinstance(device, torch.device) else torch.device(device)

    return _encino_waves_spectral_wave_numbers_kernel(
        dk=P.dk,
        N=P.N,
        device=device,
    )


def encino_waves_initial_state(
    P: EncinoWavesParameters,
    wave_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-Torch replacement for `encino_waves_initial_state()`.

    Returns
    -------
    h_spec :  (count, 2, 2) float32 -- Spectral data in real/imag format.
              h_spec[:, 0, :] is h+, h_spec[:, 1, :] is h-
              h_spec[..., 0] is real, h_spec[..., 1] is imag
    omegas :  (count,)    float32
    """
    safe_gravity = max(abs(P.gravity), 0.1)
    safe_wind_speed = 0.1 + abs(P.wind_speed)
    safe_depth = 0.1 + abs(P.depth)
    safe_swell = max(-1.0, min(1.0, P.swell))

    safe_surface_tens_over_rho = (0.001 + abs(P.surface_tension)) / max(0.1, abs(P.density))

    safe_fetch = 1.0 + max(0.0, abs(P.fetch_km) * 1000.0)
    dimless_fetch = (safe_gravity * safe_fetch) / (safe_wind_speed * safe_wind_speed)

    safe_peak_omega = math.tau * 3.5 * (safe_gravity / safe_wind_speed) * pow(dimless_fetch, -0.33)



    # --- TMA gamma (random) ---
    state = int(P.random_seed)
    state = _py_hash_state(state)
    state = _py_hash_state(state + 191_819)

    u0 = _py_u32_from_state(state)
    state = _py_hash_state(state)
    u1 = _py_u32_from_state(state)

    r0 = math.sqrt(-2.0 * math.log(max(1.1920929e-07, float(_py_norm_f32_from_u32(u0)))))
    gamma_variation = 3.30 + math.sqrt(0.67) * r0 * math.cos(
        math.tau * float(_py_norm_f32_from_u32(u1))
    )
    TMA_gamma = max(1.0, min(6.0, gamma_variation))
    TMA_alpha = 0.076 * pow(dimless_fetch, -0.22)

    h_spec, omegas = _encino_waves_initial_state_kernel(
        wave_numbers=wave_numbers,
        dk=P.dk,
        gravity=safe_gravity,
        depth=safe_depth,
        surface_tens_over_rho=safe_surface_tens_over_rho,
        wind_speed=safe_wind_speed,
        swell=safe_swell,
        peak_omega=safe_peak_omega,
        TMA_alpha=TMA_alpha,
        TMA_gamma=TMA_gamma,
        random_seed=P.random_seed,
    )

    return h_spec, omegas


def encino_waves_propagate(
    P: EncinoWavesParameters,
    time: float,
    wave_numbers: torch.Tensor,
    h_spec: torch.Tensor,  # (count, 2, 2) float32
    omegas: torch.Tensor,  # (count,) float32
) -> torch.Tensor:
    """
    Forward-propagate the Encino-Waves spectrum to time `t`
    and return spatial displacement / curvature fields.

    Returns a (4, N, N) tensor of float32 dtype, with
    T[0] : height
    T[1] : daxis0
    T[2] : daxis1
    T[3] : min_eigen
    """

    # Validate input shapes and types
    N = P.N
    width = N // 2 + 1
    expected_count = N * width

    # Check wave_numbers shape
    if not isinstance(wave_numbers, torch.Tensor):
        raise TypeError("wave_numbers must be a torch.Tensor")
    if wave_numbers.shape[0] != expected_count or wave_numbers.shape[1] != 3:
        raise ValueError(
            f"wave_numbers must have shape ({expected_count}, 3), got {wave_numbers.shape}"
        )

    # Check h_spec shape and type
    if not isinstance(h_spec, torch.Tensor):
        raise TypeError("h_spec must be a torch.Tensor")
    if h_spec.dtype != torch.float32:
        raise TypeError("h_spec must be a float32 tensor")
    if h_spec.shape != (expected_count, 2, 2):
        raise ValueError(
            f"h_spec must have shape ({expected_count}, 2, 2) for "
            f"(count, pos/neg, real/imag), got {h_spec.shape}"
        )

    # Check omegas shape
    if not isinstance(omegas, torch.Tensor):
        raise TypeError("omegas must be a torch.Tensor")
    if omegas.shape[0] != expected_count:
        raise ValueError(f"omegas must have shape ({expected_count},), got {omegas.shape}")

    # ------------ build H_hat(k,t)  ----------------------------------------------

    h_hat, di_spec, dj_spec, dii_spec, djj_spec, dij_spec = _encino_waves_pre_propagate_kernel(
        wave_numbers=wave_numbers, h_spec=h_spec, omegas=omegas, N=P.N, time=time
    )

    height, di, dj, dii, djj, dij = _encino_waves_ifft_propagate_kernel(
        h_hat=h_hat,
        di_spec=di_spec,
        dj_spec=dj_spec,
        dii_spec=dii_spec,
        djj_spec=djj_spec,
        dij_spec=dij_spec,
        N=P.N,
    )

    return _encino_waves_post_propagate_kernel(
        height=height,
        di=di,
        dj=dj,
        dii=dii,
        djj=djj,
        dij=dij,
        amplitude_gain=P.amplitude_gain,
        pinch=P.pinch,
    )


def encino_waves_undisplaced_height_field(
    P: EncinoWavesParameters,
    up: str = "z",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    device = device if isinstance(device, torch.device) else torch.device(device)
    return _encino_waves_undisplaced_height_field_kernel(
        domain=P.domain,
        N=P.N,
        up=up,
        device=device,
    )


def encino_waves_displaced_height_field(
    undisplaced: torch.Tensor,
    propagated: torch.Tensor,
    up: str = "z",
) -> torch.Tensor:
    return _encino_waves_displaced_height_field_kernel(undisplaced, propagated, up)

