## Copyright 2015-2025 Christopher Jon Horvath
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import cupy as cp
import numpy as np

_spectral_basis_cuda_source_code = """

extern "C" {

#define uint32_t unsigned int
#define uint64_t unsigned long long

// Error codes
#define ENCINO_WAVES_ERROR_OK 0
#define ENCINO_WAVES_ERROR_NULL_INPUT 1
#define ENCINO_WAVES_ERROR_INVALID_RESOLUTION 2
#define ENCINO_WAVES_ERROR_INVALID_RANK 3
#define ENCINO_WAVES_ERROR_INVALID_SHAPE 4
#define ENCINO_WAVES_ERROR_FAILED_TO_CREATE_PLAN 5

#define ENCINO_WAVES_TAU 6.2831853071795f
#define ENCINO_WAVES_PI 3.1415926535897f

#define ENCINO_WAVES_BLOCK_SIZE 256

struct Encino_waves_ocean_plan {
    int N;
    int size_i;
    int size_j;
    int count;

    float dk;
    float max_k_mag;

    float gravity;
    float sigma_over_rho;
    float depth;
    float wind_speed;
    float fetch_m;

    float tma_gamma;
    float tma_alpha;
    float tma_kd_gain;
    float peak_omega;
    float wind_speed_over_celerity;
    float swell;

    int random_seed;
};

inline __device__ float sqr(float const x) {
    return x * x;
}

inline __device__ float lerpf(float const a, float const b, float const t) {
    return (1.0f - t) * a + t * b;
}

inline __device__ float clampf(float const x, float const min, float const max) {
    return fmaxf(min, fminf(x, max));
}

inline __device__ float sigmoidf(float const x) {
    // Implement sigmoid in terms of tanh:
    // sigmoid(x) = 0.5 * (1 + tanh(x / 2))
    return 0.5f * (1.0f + tanhf(0.5f * x));
}

inline __device__ uint64_t word_from_state(uint64_t const state) {
    uint32_t s = ((uint32_t)(state));
    uint32_t const word = ((s >> ((s >> 28U) + 4U)) ^ s) * 277803737U;
    return ((uint64_t)((word >> 22U) ^ word));
}

inline __device__ float float_01_from_word(uint64_t const word) {
    double const d = ((double)(word)) / 4294967296.0;
    return d < 0.0 ? 0.0f : d > 1.0 ? 1.0f : ((float)(d));
}

inline __device__ uint32_t wave_number_to_seed_offset(float k) {
    k = roundf(k * 10000.0f);
#ifdef __CUDA_ARCH__
    return __float_as_uint(k);
#else
    uint32_t k_uint;
    memcpy(&k_uint, &k, sizeof(uint32_t));
    return k_uint;
#endif
}

inline __device__ uint64_t hash_state(uint64_t const state) {
    uint32_t s = ((uint32_t)(state));
    s = (s * 747796405U) + 2891336453U;
    return ((uint64_t)(s));
}

__device__
void encino_waves_classic_spectral_basis_at_k(
  struct Encino_waves_ocean_plan const* plan, float ki, float kj, float k_mag, float* out) {
    if (k_mag > plan->max_k_mag) {
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
        out[3] = 0.0f;
        out[4] = 0.0f;
        return;
    }

    // get thetaPos and thetaNeg from k.
    float const theta_pos = atan2f(-kj, ki);
    float const theta_neg = atan2f(kj, -ki);

    // DISPERSION
    // https://en.wikipedia.org/wiki/Dispersion_(water_waves)
    // The dispersion relationship relates the angular frequency of waves
    // to their wavelengths, based on gravity, and ocean depth, and other physical
    // parameters. These are all statistical approximations, and there are
    // a ton of them. (Look at the bottom of the wikipedia page)
    //
    // For some simpler discussion, I referenced:
    // http://www.atm.ox.ac.uk/user/read/fluids/fluidsnotes5.pdf
    //
    // A simple deep water dispersion relationship, in which ocean depth is vastly
    // larger than considered wavelengths, is:
    // omega^2 = g * k  // g = gravity
    // dOmega_dk = g / 2 sqrt( g * k ) = g / ( 2 omega )
    //
    // A finite-depth water dispersion relationship is:
    // omega^2 = g * k * tanh( k * h )  // g = gravity, h = depth
    // dOmega_dk = g ( tanh( h k ) + h k sech^2( h k ) ) / 2 sqrt( g k tanh( h k ) )
    // dOmega_dk = g ( tanh( h k ) + h k sech^2( h k ) ) / ( 2 omega )
    //
    // A capillary-wave dispersion relationship is:
    // omega^2 = ( g*k + (sigma/rho)k^3 ) tanh( k * h )
    // // g = gravity, h = depth, sigma = surface tension, rho = fluid density
    // // default sigma = 0.074 N/m, rho = 1000 kg/m^3
    // dOmega_dk = ( g + 3 k^2 s )tanh( h k ) + h k ( g + k^2 s )sech^2( h k )
    //             / ( 2 omega )
    //
    // There are others, but they get kinda crazy. We can always come back to
    // these.
    //
    // Note that they're always written in terms of omega^2, which means there
    // are two solutions: omega & -omega.  We'll return the positive one.
    //
    // Note that the capillary formulation given here will turn into
    // the finite depth dispersion, for big waves, and the finite depth dispersion
    // will turn into the deep dispersion, for deep water. So you can just
    // use the capillary dispersion all the time.

    float omega;
    float domega_dk;
    {
        float const hk = plan->depth * k_mag;
        float const k2s = sqr(k_mag) * plan->sigma_over_rho;
        float const gpk2s = plan->gravity + k2s;
        omega = sqrtf(fabsf(k_mag * gpk2s * tanhf(hk)));
        float const numer = ((gpk2s + k2s + k2s) * tanhf(hk)) + (hk * gpk2s / sqr(coshf(hk)));
        domega_dk = fabsf(numer) / (2.0f * omega);
    }

    float spectrum;
    {
        // The JONSWAP spectrum introduces the fetch parameter. It is an
        // AlphaBeta spectrum times a peak sharpening function.  The peak sharpening
        // coefficient, gamma, is chosen as a random draw.
        // alpha = 0.076 * pow( xbar, -0.22 )
        // sigma = 0.07 for w < wm, 0.09 for w > wm
        // wm = ENCINO_WAVES_TAU * 3.5 * ( g / U ) * pow( xbar, -0.33 )
        // xbar = g F / U^2
        // F = fetch
        // U = wind speed
        // g = gravity
        // y = gaussian( mean=3.30, variance=0.62 ), clamped from 1 to 6
        // peakSharpening = pow( y, exp( -(w-wm)^2/2(sigma wm)^2 )
        // beta = 1.25
        float const sigma = omega <= plan->peak_omega ? 0.07f : 0.09f;
        float const peak_sharpening =
          powf(plan->tma_gamma, expf(-sqr((omega - plan->peak_omega) / (sigma * plan->peak_omega)) / 2.0f));
        float const jonswap_alpha_beta = (plan->tma_alpha * sqr(plan->gravity) / powf(omega, 5.0f)) *
                                         expf(-1.25f * powf(plan->peak_omega / omega, 4.0f));

        float const wh = omega * plan->tma_kd_gain;
        // float const kitaigorodskii_depth = 0.5f + 0.5f * tanhf(1.8f * (wh - 1.125f));
        float const kitaigorodskii_depth = sigmoidf(3.6f * (wh - 1.125f));

        spectrum = peak_sharpening * jonswap_alpha_beta * kitaigorodskii_depth;
    }

    // Attenuate by directional spreading
    float dir_spread_pos;
    float dir_spread_neg;
    {
        // Hasselmann Directional Spreading
        float shape_bias = 0.0;
        if (plan->swell >= 0.0f) { shape_bias = 16.1f * tanhf(plan->peak_omega / omega) * sqr(plan->swell); }
        float shape;
        if (omega > plan->peak_omega) {
            shape = 9.77f * powf(omega / plan->peak_omega, -2.33f - (1.45f * (plan->wind_speed_over_celerity - 1.17f)));
        } else {
            shape = 6.97f * powf(omega / plan->peak_omega, 4.06f);
        }
        shape += shape_bias;
        float const factor_a = powf(2.0f, (2.0f * shape) - 1.0f) / ENCINO_WAVES_PI;

        // factor_b = torch.exp(2.0 * torch.lgamma(shape + 1.0) - torch.lgamma((2.0 * shape) - 1.0))
        // float const factor_b = sqr(tgammaf(shape + 1.0f)) / tgammaf((2.0f * shape) + 1.0f);
        float const factor_b = expf(2.0f * lgammaf(shape + 1.0f) - lgammaf((2.0f * shape) + 1.0f));
        float const factor_c_pos = powf(fabsf(cosf(theta_pos / 2.0f)), 2.0f * shape);
        float const factor_c_neg = powf(fabsf(cosf(theta_neg / 2.0f)), 2.0f * shape);

        dir_spread_pos = factor_a * factor_b * factor_c_pos;
        dir_spread_neg = factor_a * factor_b * factor_c_neg;

        if (plan->swell < 0.0f) {
            dir_spread_pos = lerpf(dir_spread_pos, 1.0f / ENCINO_WAVES_TAU, -plan->swell);
            dir_spread_neg = lerpf(dir_spread_neg, 1.0f / ENCINO_WAVES_TAU, -plan->swell);
        }
    }

    // The area of each point being integrated is dki * dkj.  However,
    // We're evaluating the function in omega & theta space.  We need to
    // use the theory of change of variables to convert dki * dkj to
    // dtheta * domega.  We actually do two change of variables. The
    // first is to kMag and theta, where kMag is sqrt( ki^2 + kj^2 ), and
    // theta is atan( kj / ki ).  The second is from kMag and theta to
    // omega and theta.  Both changes involve multiplying the result by the
    // absolute value of the determinant of the Jacobian of the variable
    // matrix. Using the finite-depth form of the dispersion relationship
    // for: omega = sqrt( g kmag tanh( h kmag ) ), we get the following
    // relationship.
    // (1/2)amp^2 = Spectrum( omega, theta ) deltaOmega deltaTheta
    // (1/2)amp^2 = S(omega,theta) *
    //              abs( ( domega_dkmag / kMag ) deltaKi deltaKj )
    // Where domega_dkmag is computed by the Dispersion relationship.

    // Multiply DeltaSPos by domega_dk / kMag, which completes the
    // change of variables, and then multiply by dK^2.
    float const change_of_variables_factor = sqr(plan->dk) * fabsf(domega_dk / k_mag);
    float const delta_s_pos = spectrum * dir_spread_pos * change_of_variables_factor;
    float const delta_s_neg = spectrum * dir_spread_neg * change_of_variables_factor;

    // Get four variates
    // This is the 32-bit version of the PCG spatial hash for producing
    // structureless uniform noise on a grid, using ki & kj discretized.
    // Doing it in this way makes it so that increasing resolution doesn't
    // change the wave layout, it just adds detail. This is a critical
    // artistic requirement.
    uint64_t variate_0;
    uint64_t variate_1;
    uint64_t variate_2;
    uint64_t variate_3;
    {
        uint64_t state = hash_state(plan->random_seed);
        state = hash_state(((uint32_t)(state)) + wave_number_to_seed_offset(kj));
        state = hash_state(((uint32_t)(state)) + wave_number_to_seed_offset(ki));

        variate_0 = word_from_state(state);
        state = hash_state(state);

        variate_1 = word_from_state(state);
        state = hash_state(state);

        variate_2 = word_from_state(state);
        state = hash_state(state);

        variate_3 = word_from_state(state);
    }

    float amp_pos;
    float amp_neg;
    {
        // This is just the box-muller method for producing two normally
        // distributed random variates.
        float const r = sqrtf(-2.0f * logf(fmaxf(1.0e-6f, float_01_from_word(variate_0))));
        float const theta = ENCINO_WAVES_TAU * float_01_from_word(variate_1);
        amp_pos = r * cosf(theta);
        amp_neg = r * sinf(theta);
    }

    // Factor in the spectrum and directional spreading.
    amp_pos *= sqrtf(fabsf(delta_s_pos * 2.0f));
    amp_neg *= sqrtf(fabsf(delta_s_neg * 2.0f));

    if (!isfinite(amp_pos) || !isfinite(amp_neg)) {
        amp_pos = 0.0f;
        amp_neg = 0.0f;
    }

    // Final results!
    // Get random draws for phase.
    float const phase_pos = ENCINO_WAVES_TAU * float_01_from_word(variate_2);
    float const phase_neg = ENCINO_WAVES_TAU * float_01_from_word(variate_3);

    out[0] = amp_pos * cosf(phase_pos);
    out[1] = amp_pos * -sinf(phase_pos);
    out[2] = amp_neg * cosf(phase_neg);
    out[3] = amp_neg * -sinf(phase_neg);
    out[4] = omega;
}

__device__
void encino_waves_classic_spectral_basis_at_kidx(struct Encino_waves_ocean_plan const* plan,
                                                 int wave_number_index,
                                                 float* out) {
    int const j = wave_number_index / plan->size_i;
    int const real_j = j < (plan->N / 2) ? j : j - plan->N;
    int const i = wave_number_index % plan->size_i;

    // ki, kj are the wave numbers in the i and j directions.
    float const ki = i * plan->dk;
    float const kj = real_j * plan->dk;
    float const k_mag = sqrtf(ki * ki + kj * kj);

    // Get out for the DC component
    if (i == 0 && real_j == 0) {
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
        out[3] = 0.0f;
        out[4] = 0.0f;
    } else {
        encino_waves_classic_spectral_basis_at_k(plan, ki, kj, k_mag, out);
    }
}

__device__
int encino_waves_classic_spectral_basis_block(struct Encino_waves_ocean_plan const* plan,
                                              int wave_number_index_begin,
                                              int wave_number_index_end,
                                              int wave_number_index_step,
                                              float* spectral_basis_out) {
    if (!plan || !spectral_basis_out) { return ENCINO_WAVES_ERROR_NULL_INPUT; }

    if (wave_number_index_begin < 0 || wave_number_index_end > plan->count ||
        wave_number_index_begin >= wave_number_index_end) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    for (int wave_number_index = wave_number_index_begin; wave_number_index < wave_number_index_end;
         wave_number_index += wave_number_index_step) {
        encino_waves_classic_spectral_basis_at_kidx(
          plan, wave_number_index, spectral_basis_out + (wave_number_index * 5));
    }

    return ENCINO_WAVES_ERROR_OK;
}

__global__ __launch_bounds__(ENCINO_WAVES_BLOCK_SIZE) 
void encino_waves_spectral_basis_cuda_kernel(
    int N,
    int size_i,
    int size_j,
    int count,

    float dk,
    float max_k_mag,

    float gravity,
    float sigma_over_rho,
    float depth,
    float wind_speed,
    float fetch_m,

    float tma_gamma,
    float tma_alpha,
    float tma_kd_gain,
    float peak_omega,
    float wind_speed_over_celerity,
    float swell,

    int random_seed,
    float* out_spectral_basis) {

    struct Encino_waves_ocean_plan plan{
        N,
        size_i,
        size_j,
        count,
        dk,
        max_k_mag,
        gravity,
        sigma_over_rho,
        depth,
        wind_speed,
        fetch_m,
        tma_gamma,
        tma_alpha,
        tma_kd_gain,
        peak_omega,
        wind_speed_over_celerity,
        swell,
        random_seed
    };

    // Uses grid striding.
    int const index_inside_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int const grid_stride = gridDim.x * blockDim.x;
    encino_waves_classic_spectral_basis_block(&plan, index_inside_grid, plan.count, grid_stride, out_spectral_basis);
}

} // extern "C"
"""

_simple_debug_kernel_code = """
extern "C" __global__
void debug_kernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (float)idx + 100.0f;  // Simple test pattern
    }
}
"""

_diagnostic_kernel_code = """
extern "C" {

#define uint32_t unsigned int
#define uint64_t unsigned long long
#define ENCINO_WAVES_TAU 6.2831853071795f
#define ENCINO_WAVES_PI 3.1415926535897f
#define ENCINO_WAVES_BLOCK_SIZE 256

struct Encino_waves_ocean_plan {
    int N;
    int size_i;
    int size_j;
    int count;
    float dk;
    float max_k_mag;
    float gravity;
    float sigma_over_rho;
    float depth;
    float wind_speed;
    float fetch_m;
    float tma_gamma;
    float tma_alpha;
    float tma_kd_gain;
    float peak_omega;
    float wind_speed_over_celerity;
    float swell;
    int random_seed;
};

__global__ void diagnostic_kernel(
    int N, int size_i, int size_j, int count,
    float dk, float max_k_mag, float gravity, float sigma_over_rho, float depth,
    float wind_speed, float fetch_m, float tma_gamma, float tma_alpha, 
    float tma_kd_gain, float peak_omega, float wind_speed_over_celerity, float swell,
    int random_seed, float* diagnostics) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Calculate wave number indices
    int j = idx / size_i;
    int real_j = j < (N / 2) ? j : j - N;
    int i = idx % size_i;
    
    float ki = i * dk;
    float kj = real_j * dk;
    float k_mag = sqrtf(ki * ki + kj * kj);
    
    // Output diagnostic info for the first few points
    if (idx < 10) {
        diagnostics[idx * 8 + 0] = (float)i;      // i index
        diagnostics[idx * 8 + 1] = (float)real_j; // real_j index  
        diagnostics[idx * 8 + 2] = ki;            // ki wave number
        diagnostics[idx * 8 + 3] = kj;            // kj wave number
        diagnostics[idx * 8 + 4] = k_mag;         // magnitude
        diagnostics[idx * 8 + 5] = max_k_mag;     // max allowed magnitude
        diagnostics[idx * 8 + 6] = (k_mag > max_k_mag) ? 1.0f : 0.0f; // cutoff test
        diagnostics[idx * 8 + 7] = ((i == 0 && real_j == 0) ? 1.0f : 0.0f); // DC test
    }
}

} // extern "C"
"""

_parameter_test_kernel_code = """
extern "C" __global__
void parameter_test_kernel(
    int N, int size_i, int size_j, int count,
    float dk, float max_k_mag, float gravity, float sigma_over_rho, float depth,
    float wind_speed, float fetch_m, float tma_gamma, float tma_alpha, 
    float tma_kd_gain, float peak_omega, float wind_speed_over_celerity, float swell,
    int random_seed, float* output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 18) {  // Output first 18 parameter values
        if (idx == 0) output[idx] = (float)N;
        else if (idx == 1) output[idx] = (float)size_i;
        else if (idx == 2) output[idx] = (float)size_j;
        else if (idx == 3) output[idx] = (float)count;
        else if (idx == 4) output[idx] = dk;
        else if (idx == 5) output[idx] = max_k_mag;
        else if (idx == 6) output[idx] = gravity;
        else if (idx == 7) output[idx] = sigma_over_rho;
        else if (idx == 8) output[idx] = depth;
        else if (idx == 9) output[idx] = wind_speed;
        else if (idx == 10) output[idx] = fetch_m;
        else if (idx == 11) output[idx] = tma_gamma;
        else if (idx == 12) output[idx] = tma_alpha;
        else if (idx == 13) output[idx] = tma_kd_gain;
        else if (idx == 14) output[idx] = peak_omega;
        else if (idx == 15) output[idx] = wind_speed_over_celerity;
        else if (idx == 16) output[idx] = swell;
        else if (idx == 17) output[idx] = (float)random_seed;
    }
}
"""

ENCINO_WAVES_BLOCK_SIZE = 256

def encino_waves_spectral_basis_cupy(plan: "OceanPlan", max_blocks_per_grid=1024, out:np.ndarray|None=None, debug=False) -> np.ndarray:

    spectral_basis_kernel = cp.RawKernel(_spectral_basis_cuda_source_code, "encino_waves_spectral_basis_cuda_kernel")

    N = plan.N 
    size_i = plan.size_i 
    size_j = plan.size_j

    # Debug output
    if debug:
        print(f"Plan parameters: N={N}, size_i={size_i}, size_j={size_j}, count={plan.count}")
        print(f"Expected output shape: ({plan.count * 5},) -> reshape to ({size_j}, {size_i}, 5)")

    # Create 1D output tensor as expected by kernel
    out_tensor_flat_cp = cp.empty(plan.count * 5, dtype=cp.float32)
    
    # Initialize with a known pattern for debugging
    if debug:
        out_tensor_flat_cp.fill(-999.0)  # Fill with sentinel value
    
    # Launch the kernel
    needed_block_count = (plan.count + ENCINO_WAVES_BLOCK_SIZE - 1) // ENCINO_WAVES_BLOCK_SIZE;
    if max_blocks_per_grid > 0:
        blocks_per_grid = min(max_blocks_per_grid, needed_block_count)
    else:
        blocks_per_grid = needed_block_count

    if debug:
        print(f"Kernel launch: grid=({blocks_per_grid},), block=({ENCINO_WAVES_BLOCK_SIZE},)")

    try:
        spectral_basis_kernel(
            grid=(blocks_per_grid,),
            block=(ENCINO_WAVES_BLOCK_SIZE,), 
            args=(plan.N, 
            plan.size_i, 
            plan.size_j, 
            plan.count, 
            np.float32(plan.dk), 
            np.float32(plan.max_k_mag), 
            np.float32(plan.gravity), 
            np.float32(plan.sigma_over_rho), 
            np.float32(plan.depth), 
            np.float32(plan.wind_speed), 
            np.float32(plan.fetch_m), 
            np.float32(plan.tma_gamma), 
            np.float32(plan.tma_alpha), 
            np.float32(plan.tma_kd_gain), 
            np.float32(plan.peak_omega), 
            np.float32(plan.wind_speed_over_celerity), 
            np.float32(plan.swell), 
            plan.random_seed, 
            out_tensor_flat_cp))
        
        # Synchronize to ensure kernel completion
        cp.cuda.Device().synchronize()
        
        if debug:
            # Check if kernel actually wrote data
            flat_data = cp.asnumpy(out_tensor_flat_cp)
            non_sentinel_count = np.sum(flat_data != -999.0)
            print(f"Values changed from sentinel: {non_sentinel_count}/{len(flat_data)}")
            print(f"Output range: [{np.min(flat_data)}, {np.max(flat_data)}]")
            print(f"First 10 values: {flat_data[:10]}")
            
    except Exception as e:
        print(f"Kernel launch error: {e}")
        if debug:
            # Return the unmodified tensor for debugging
            return cp.asnumpy(out_tensor_flat_cp.reshape(size_j, size_i, 5))
        raise
    
    # Reshape to expected 3D format
    out_tensor_cp = out_tensor_flat_cp.reshape(size_j, size_i, 5)

    if out is not None and isinstance(out, np.ndarray) and out.shape == (size_j, size_i, 5):
        cp.asnumpy(out_tensor_cp, out=out)
        return out
    else:
        return cp.asnumpy(out_tensor_cp)
    
    


def test_simple_debug_kernel(plan: "OceanPlan") -> np.ndarray:
    """Test with a simple kernel to verify basic CuPy functionality."""
    debug_kernel = cp.RawKernel(_simple_debug_kernel_code, "debug_kernel")
    
    n = plan.count * 5
    output = cp.zeros(n, dtype=cp.float32)
    
    blocks_per_grid = (n + ENCINO_WAVES_BLOCK_SIZE - 1) // ENCINO_WAVES_BLOCK_SIZE
    debug_kernel((blocks_per_grid,), (ENCINO_WAVES_BLOCK_SIZE,), (output, n))
    cp.cuda.Device().synchronize()
    
    return cp.asnumpy(output.reshape(plan.size_j, plan.size_i, 5))
    
    


def run_diagnostic_kernel(plan: "OceanPlan") -> np.ndarray:
    """Run diagnostic kernel to see intermediate values."""
    diagnostic_kernel = cp.RawKernel(_diagnostic_kernel_code, "diagnostic_kernel")
    
    # Output 8 diagnostic values for first 10 points
    diagnostics = cp.zeros(10 * 8, dtype=cp.float32)
    
    blocks_per_grid = (plan.count + ENCINO_WAVES_BLOCK_SIZE - 1) // ENCINO_WAVES_BLOCK_SIZE
    diagnostic_kernel(
        (blocks_per_grid,), (ENCINO_WAVES_BLOCK_SIZE,),
        (plan.N, plan.size_i, plan.size_j, plan.count,
         np.float32(plan.dk), np.float32(plan.max_k_mag), np.float32(plan.gravity), np.float32(plan.sigma_over_rho), np.float32(plan.depth),
         np.float32(plan.wind_speed), np.float32(plan.fetch_m), np.float32(plan.tma_gamma), np.float32(plan.tma_alpha),
         np.float32(plan.tma_kd_gain), np.float32(plan.peak_omega), np.float32(plan.wind_speed_over_celerity), np.float32(plan.swell),
         plan.random_seed, diagnostics))
    
    cp.cuda.Device().synchronize()
    return cp.asnumpy(diagnostics).reshape(10, 8)
    
    


def test_parameter_passing(plan: "OceanPlan"):
    """Test if parameters are being passed correctly to the kernel."""
    param_test_kernel = cp.RawKernel(_parameter_test_kernel_code, "parameter_test_kernel")
    
    output = cp.zeros(18, dtype=cp.float32)
    
    param_test_kernel(
        (1,), (32,),
        (plan.N, plan.size_i, plan.size_j, plan.count,
         np.float32(plan.dk), np.float32(plan.max_k_mag), np.float32(plan.gravity), np.float32(plan.sigma_over_rho), np.float32(plan.depth),
         np.float32(plan.wind_speed), np.float32(plan.fetch_m), np.float32(plan.tma_gamma), np.float32(plan.tma_alpha),
         np.float32(plan.tma_kd_gain), np.float32(plan.peak_omega), np.float32(plan.wind_speed_over_celerity), np.float32(plan.swell),
         plan.random_seed, output))
    
    cp.cuda.Device().synchronize()
    result = cp.asnumpy(output)
    
    param_names = ["N", "size_i", "size_j", "count", "dk", "max_k_mag", "gravity", 
                   "sigma_over_rho", "depth", "wind_speed", "fetch_m", "tma_gamma", 
                   "tma_alpha", "tma_kd_gain", "peak_omega", "wind_speed_over_celerity", 
                   "swell", "random_seed"]
    
    print("Parameter passing test:")
    print("Parameter         | Expected        | Kernel received | Match")
    print("------------------|-----------------|-----------------|------")
    
    all_match = True
    for i, name in enumerate(param_names):
        # Handle case differences between parameter names and OceanPlan properties
        prop_name = name
        if name == "N":
            expected = plan.N
        elif name == "size_i":
            expected = plan.size_i
        elif name == "size_j":  
            expected = plan.size_j
        elif name == "count":
            expected = plan.count
        elif name == "dk":
            expected = plan.dk
        elif name == "max_k_mag":
            expected = plan.max_k_mag
        elif name == "gravity":
            expected = plan.gravity
        elif name == "sigma_over_rho":
            expected = plan.sigma_over_rho
        elif name == "depth":
            expected = plan.depth
        elif name == "wind_speed":
            expected = plan.wind_speed
        elif name == "fetch_m":
            expected = plan.fetch_m
        elif name == "tma_gamma":
            expected = plan.tma_gamma
        elif name == "tma_alpha":
            expected = plan.tma_alpha
        elif name == "tma_kd_gain":
            expected = plan.tma_kd_gain
        elif name == "peak_omega":
            expected = plan.peak_omega
        elif name == "wind_speed_over_celerity":
            expected = plan.wind_speed_over_celerity
        elif name == "swell":
            expected = plan.swell
        elif name == "random_seed":
            expected = plan.random_seed
        else:
            expected = 0.0
            
        received = result[i]
        match = abs(expected - received) < 1e-6 if isinstance(expected, float) else expected == received
        all_match &= match
        print(f"{name:17} | {expected:15.6g} | {received:15.6g} | {'✓' if match else '✗'}")
    
    return all_match
    
    

