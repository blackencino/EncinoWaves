// Copyright 2015-2025 Christopher Jon Horvath
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifndef ENCINO_WAVES_H_INCLUDED
#define ENCINO_WAVES_H_INCLUDED

#include <cstdint.h>
#include <math.h>

#if !defined(__CUDA_ARCH__)

// This is the non-CUDA version
#if defined(_WIN32)
#define ENCINO_WAVES_API __declspec(dllexport)
#else
#define ENCINO_WAVES_API __attribute__((visibility("default")))
#endif

#define ENCINO_WAVES_HOSTDEV

#else

// This is the CUDA version
#define ENCINO_WAVES_API
#define ENCINO_WAVES_HOSTDEV __host__ __device__

#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ENCINO_WAVES_SOURCE_CODE_BODY_ONLY

// Error codes
#define ENCINO_WAVES_ERROR_OK 0
#define ENCINO_WAVES_ERROR_NULL_INPUT 1
#define ENCINO_WAVES_ERROR_INVALID_RESOLUTION 2
#define ENCINO_WAVES_ERROR_INVALID_RANK 3
#define ENCINO_WAVES_ERROR_INVALID_SHAPE 4

// Parameter defaults and ranges

// Resolution, must be a power of two, between 4 and 8192 (inclusive)
#define ENCINO_WAVES_RESOLUTION_DEFAULT 512
#define ENCINO_WAVES_RESOLUTION_MIN 4
#define ENCINO_WAVES_RESOLUTION_MAX 8192

// Domain is the size of the world space that the waves occupy, in meters.
// Range is 1cm to 100km. This determines the size of the largest wavelength,
// and with the resolution, determines the size of the smallest wavelength.
#define ENCINO_WAVES_DOMAIN_DEFAULT 100.0f
#define ENCINO_WAVES_DOMAIN_MIN 0.01f
#define ENCINO_WAVES_DOMAIN_MAX 100000.0f

// Gravitational constant, in meters per second squared.
// Range is 0.1 to 100.
#define ENCINO_WAVES_GRAVITY_DEFAULT 9.81f
#define ENCINO_WAVES_GRAVITY_MIN 0.1f
#define ENCINO_WAVES_GRAVITY_MAX 100.0f

// Surface tension, in Newtons per meter.
// Range is 0.001 to 1.0.
// Plausible physical range of surface tensions is
// 0.008 (liquid nitrogen) to 0.485 (mercury).
#define ENCINO_WAVES_SURFACE_TENSION_DEFAULT 0.074f
#define ENCINO_WAVES_SURFACE_TENSION_MIN 0.001f
#define ENCINO_WAVES_SURFACE_TENSION_MAX 1.0f

// Density, in kilograms per meter cubed.
// Range is 0.001 to 10000.
// Plausible physical range of densities is
// 0.001 (liquid helium) to 10000 (osmium).
#define ENCINO_WAVES_DENSITY_DEFAULT 1000.0f
#define ENCINO_WAVES_DENSITY_MIN 0.001f
#define ENCINO_WAVES_DENSITY_MAX 10000.0f

// Depth, in meters.
// Range is 0.01 to 20000.
// Deepest ocean depth is 10,984 meters, at the Challenger Deep in the
// Mariana Trench.
#define ENCINO_WAVES_DEPTH_DEFAULT 100.0f
#define ENCINO_WAVES_DEPTH_MIN 0.01f
#define ENCINO_WAVES_DEPTH_MAX 20000.0f

// Wind speed, in meters per second.
// Range is 0.01 to 300.
// Highest windspeed ever recorded on earth (over an ocean) is 113.3
// m/s, on April 10, 1996, near Barrow Island, Australia, as part of
// Severe Tropical Cyclone Olivia.
#define ENCINO_WAVES_WIND_SPEED_DEFAULT 17.0f
#define ENCINO_WAVES_WIND_SPEED_MIN 0.01f
#define ENCINO_WAVES_WIND_SPEED_MAX 300.0f

// Fetch, in kilometers.
// Range is 0.01 to 20000.
// The longest fetch for a single storm is between 4000 and 5000 km, for
// a massive extratropical cyclone. The theoretical maximum fetch is
// 10,000 km, but that's for the persistent global wind belt of the
// Southern Ocean Westerlies.
#define ENCINO_WAVES_FETCH_KM_DEFAULT 300.0f
#define ENCINO_WAVES_FETCH_KM_MIN 0.01f
#define ENCINO_WAVES_FETCH_KM_MAX 20000.0f

// Swell, dimensionless.
// Range is -1 to 1.
// -1 = no directionality, 0 = hasselmann, 1 = full elongation
#define ENCINO_WAVES_SWELL_DEFAULT 0.0f
#define ENCINO_WAVES_SWELL_MIN -1.0f
#define ENCINO_WAVES_SWELL_MAX 1.0f

// Random seed.
#define ENCINO_WAVES_RANDOM_SEED_DEFAULT 54321

#define ENCINO_WAVES_TAU 6.2831853071795f
#define ENCINO_WAVES_PI 3.1415926535897f

#define ENCINO_WAVES_BLOCK_SIZE 256

struct Encino_waves_ocean_params {
    int resolution;         // must be a power of two, between 4 and 8192 (inclusive)
    float domain;           // in meters
    float gravity;          // in meters per second squared.
    float surface_tension;  // in Newtons per meter
    float density;          // in kilograms per meter cubed
    float depth;            // in meters.
    float wind_speed;       // in meters per second
    float fetch_km;         // in kilometers
    float swell;            // -1 = no directionality, 0 = hasselmann, 1 = full elongation
    int random_seed;
};

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

struct Encino_waves_propagation_params {
    float time = 0.0f;
    float pinch = 0.75f;          // lateral displacement
    float amplitude_gain = 1.0f;  // vertical displacement
};

ENCINO_WAVES_API
int encino_waves_create_ocean_plan(struct Encino_waves_ocean_params const* params, struct Encino_waves_ocean_plan* out);

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
void encino_waves_classic_spectral_basis_at_k(
  struct Encino_waves_ocean_plan const* plan, float ki, float kj, float k_mag, float* out);

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
void encino_waves_classic_spectral_basis_at_kidx(struct Encino_waves_ocean_plan const* plan,
                                                 int wave_number_index,
                                                 float* out);

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
int encino_waves_classic_spectral_basis_block(struct Encino_waves_ocean_plan const* plan,
                                              int wave_number_index_begin,
                                              int wave_number_index_end,
                                              int wave_number_index_step,
                                              float* out);

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
void encino_waves_spectral_height_at(float time, float const* spectral_basis, float* out);

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
int encino_waves_spectral_height_block(struct Encino_waves_ocean_plan const* plan,
                                       float time,
                                       float const* spectral_basis,
                                       int wave_number_index_begin,
                                       int wave_number_index_end,
                                       int wave_number_index_step,
                                       float* spectral_height_out);

/**********************************************************************************************
 *                                BEGIN CUDA HEADER SECTION                                   *
 **********************************************************************************************/

#ifdef ENCINO_WAVES_CUDA_KERNELS

#include <cuda_runtime.h>

ENCINO_WAVES_API
int encino_waves_spectral_basis_cuda(cudaStream_t stream,
                                     struct Encino_waves_ocean_plan const* plan,
                                     int max_blocks_per_grid,

                                     int out_rank,
                                     int const* out_shape,
                                     float* out_spectral_basis);

ENCINO_WAVES_API
int encino_waves_spectral_height_cuda(cudaStream_t stream,
                                      struct Encino_waves_ocean_plan const* plan,
                                      float time,
                                      int max_blocks_per_grid,
                                      int in_rank,
                                      int const* in_shape,
                                      float const* spectral_basis,
                                      int out_rank,
                                      int const* out_shape,
                                      float* out_spectral_height);

#endif  // ENCINO_WAVES_CUDA_KERNELS

/**********************************************************************************************
 *                                BEGIN OMP HEADER SECTION                                    *
 **********************************************************************************************/

#ifdef ENCINO_WAVES_OMP_KERNELS

#include <omp.h>

ENCINO_WAVES_API
int encino_waves_spectral_basis_omp(struct Encino_waves_ocean_plan const* plan,
                                    int max_threads,
                                    int out_rank,
                                    int const* out_shape,
                                    float* out_spectral_basis);

ENCINO_WAVES_API
int encino_waves_spectral_height_omp(struct Encino_waves_ocean_plan const* plan,
                                     float time,
                                     int in_rank,
                                     int const* in_shape,
                                     float const* spectral_basis,
                                     int out_rank,
                                     int const* out_shape,
                                     float* out_spectral_height);

#endif  // ENCINO_WAVES_OMP_KERNELS

/**********************************************************************************************
 *                                END OF HEADER SECTION                                       *
 **********************************************************************************************/

#endif  // ENCINO_WAVES_SOURCE_CODE_BODY_ONLY

#ifndef ENCINO_WAVES_SOURCE_CODE_HEADER_ONLY

/**********************************************************************************************
 *                                BEGIN OF BODY SECTION                                       *
 **********************************************************************************************/

inline ENCINO_WAVES_HOSTDEV float sqr(float const x) {
    return x * x;
}

inline ENCINO_WAVES_HOSTDEV float lerpf(float const a, float const b, float const t) {
    return (1.0f - t) * a + t * b;
}

inline ENCINO_WAVES_HOSTDEV float clampf(float const x, float const min, float const max) {
    return fmaxf(min, fminf(x, max));
}

inline ENCINO_WAVES_HOSTDEV float smoothstepf(float const edge0, float const edge1, float const x) {
    float const t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline ENCINO_WAVES_HOSTDEV uint64_t word_from_state(uint64_t const state) {
    uint32_t s = ((uint32_t)(state));
    uint32_t const word = ((s >> ((s >> 28U) + 4U)) ^ s) * 277803737U;
    return ((uint64_t)((word >> 22U) ^ word));
}

inline ENCINO_WAVES_HOSTDEV float float_01_from_word(uint64_t const word) {
    double const d = ((double)(word)) / 4294967296.0;
    return d < 0.0 ? 0.0f : d > 1.0 ? 1.0f : ((float)(d));
}

inline ENCINO_WAVES_HOSTDEV uint64_t hash_state(uint64_t const state) {
    uint32_t s = ((uint32_t)(state));
    s = (s * 747796405U) + 2891336453U;
    return ((uint64_t)(s));
}

ENCINO_WAVES_API
int encino_waves_create_ocean_plan(struct Encino_waves_ocean_params const* params,
                                   struct Encino_waves_ocean_plan* out) {
    if (!params || !out) { return ENCINO_WAVES_ERROR_NULL_INPUT; }

    // Check that params->resolution is a power of two between 4 and 8192 (inclusive)
    if (!(params->resolution >= 4 && params->resolution <= 8192 &&
          (params->resolution & (params->resolution - 1)) == 0)) {
        return ENCINO_WAVES_ERROR_INVALID_RESOLUTION;
    }

    out->N = params->resolution;
    out->size_i = (out->N / 2) + 1;
    out->size_j = out->N;
    out->count = out->size_i * out->size_j;

    float const L = clampf(fabsf(params->domain), ENCINO_WAVES_DOMAIN_MIN, ENCINO_WAVES_DOMAIN_MAX);

    // Derived constants
    out->dk = ENCINO_WAVES_TAU / L;
    out->max_k_mag = (out->N / 2) * out->dk;

    out->gravity = clampf(fabsf(params->gravity), ENCINO_WAVES_GRAVITY_MIN, ENCINO_WAVES_GRAVITY_MAX);
    float const surface_tension =
      clampf(fabsf(params->surface_tension), ENCINO_WAVES_SURFACE_TENSION_MIN, ENCINO_WAVES_SURFACE_TENSION_MAX);
    float const rho = clampf(fabsf(params->density), ENCINO_WAVES_DENSITY_MIN, ENCINO_WAVES_DENSITY_MAX);
    out->depth = clampf(fabsf(params->depth), ENCINO_WAVES_DEPTH_MIN, ENCINO_WAVES_DEPTH_MAX);
    out->wind_speed = clampf(fabsf(params->wind_speed), ENCINO_WAVES_WIND_SPEED_MIN, ENCINO_WAVES_WIND_SPEED_MAX);
    out->fetch_m = 1000.0f * clampf(fabsf(params->fetch), ENCINO_WAVES_FETCH_KM_MIN, ENCINO_WAVES_FETCH_KM_MAX);
    out->swell = clampf(params->swell, ENCINO_WAVES_SWELL_MIN, ENCINO_WAVES_SWELL_MAX);

    out->sigma_over_rho = surface_tension / rho;

    // TMA gamma random draw.
    {
        uint64_t state = hash_state(params->random_seed + 191819);
        uint64_t const variate_0 = word_from_state(state);
        state = hash_state(state);
        uint64_t const variate_1 = word_from_state(state);

        // This is just the box-muller method for producing two normally
        // distributed random variates, but only taking the first one.
        float const r = sqrtf(-2.0f * logf(fmaxf(1.0e-6f, float_01_from_word(variate_0))));
        float const theta = ENCINO_WAVES_TAU * float_01_from_word(variate_1);
        float const gamma = 3.30f + sqrtf(0.67f) * (r * cosf(theta));
        out->tma_gamma = clampf(gamma, 1.0f, 6.0f);
    }

    float const dimensionless_fetch = fabsf(out->gravity * out->fetch_m / sqr(out->wind_speed));
    out->tma_alpha = 0.076f * powf(dimensionless_fetch, -0.22f);
    out->peak_omega =
      ENCINO_WAVES_TAU * 3.5f * fabsf(out->gravity / out->wind_speed) * powf(dimensionless_fetch, -0.33f);
    out->tma_kd_gain = sqrtf(out->depth / out->gravity);

    // Hasselmann Directional Spreading
    float const modal_shape = 11.5f * powf(out->peak_omega * out->wind_speed / out->gravity, -2.5f);
    float const modal_celerity = out->gravity / out->peak_omega;
    out->wind_speed_over_celerity = out->wind_speed / modal_celerity;

    out->random_seed = params->random_seed;

    return ENCINO_WAVES_ERROR_OK;
}

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
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
        float const numer = ((gpk2s + k2s + k2s) * fabsf(hk)) + (hk * gpk2s / sqr(coshf(hk)));
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
        float const kitaigorodskii_depth = 0.5f + 0.5f * tanhf(1.8f * (wh - 1.125f));

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
        float const factor_b = sqr(tgammaf(shape + 1.0f)) / tgammaf((2.0f * shape) + 1.0f);
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
    float const change_of_variables_factor = (plan->dk * plan->dk) * fabsf(domega_dk / k_mag);
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
        state = hash_state(((uint32_t)(state)) + ((uint32_t)(kj * 10000.0f)));
        state = hash_state(((uint32_t)(state)) + ((uint32_t)(ki * 10000.0f)));

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

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
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

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
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

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
void encino_waves_spectral_height_at(float time, float const* spectral_basis, float* out) {
    float const h_pos_real = spectral_basis[0];
    float const h_pos_imag = spectral_basis[1];
    float const h_neg_real = spectral_basis[2];
    float const h_neg_imag = spectral_basis[3];
    float const omega = spectral_basis[4];

    float const cos_omega_t = cosf(omega * time);
    float const sin_omega_t = sinf(omega * time);

    // (h_pos_real + i h_pos_imag) * (cos_omega_t - i sin_omega_t)
    float const fwd_real = h_pos_real * cos_omega_t + h_pos_imag * sin_omega_t;
    float const fwd_imag = h_pos_imag * cos_omega_t - h_pos_real * sin_omega_t;

    // (h_neg_real + i h_neg_imag) * (cos_omega_t + i sin_omega_t)
    float const bkwd_real = h_neg_real * cos_omega_t - h_neg_imag * sin_omega_t;
    float const bkwd_imag = h_neg_imag * cos_omega_t + h_neg_real * sin_omega_t;

    // final complex result, just add the forward and backward components.
    out[0] = fwd_real + bkwd_real;
    out[1] = fwd_imag + bkwd_imag;
}

ENCINO_WAVES_API
ENCINO_WAVES_HOSTDEV
int encino_waves_spectral_height_block(struct Encino_waves_ocean_plan const* plan,
                                       float time,
                                       float const* spectral_basis,
                                       int wave_number_index_begin,
                                       int wave_number_index_end,
                                       int wave_number_index_step,
                                       float* spectral_height_out) {
    if (!plan || !spectral_basis || !spectral_height_out) { return ENCINO_WAVES_ERROR_NULL_INPUT; }

    if (wave_number_index_begin < 0 || wave_number_index_end > plan->count ||
        wave_number_index_begin >= wave_number_index_end) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    for (int wave_number_index = wave_number_index_begin; wave_number_index < wave_number_index_end;
         wave_number_index += wave_number_index_step) {
        encino_waves_spectral_height_at(
          time, spectral_basis + (wave_number_index * 5), spectral_height_out + (wave_number_index * 2));
    }

    return ENCINO_WAVES_ERROR_OK;
}

/**********************************************************************************************
 *                                BEGIN CUDA KERNELS SECTION                                  *
 **********************************************************************************************/

#ifdef ENCINO_WAVES_CUDA_KERNELS

__global__ void encino_waves_spectral_basis_cuda_kernel(struct Encino_waves_ocean_plan const plan,

                                                        float* out_spectral_basis) {
    // Uses grid striding.
    int const index_inside_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int const grid_stride = gridDim.x * blockDim.x;
    encino_waves_classic_spectral_basis_block(&plan, index_inside_grid, plan.count, grid_stride, out_spectral_basis);
}

ENCINO_WAVES_API
int encino_waves_spectral_basis_cuda(cudaStream_t stream,
                                     struct Encino_waves_ocean_plan const* plan,
                                     int max_blocks_per_grid,

                                     int out_rank,
                                     int const* out_shape,
                                     float* out_spectral_basis) {
    if (!plan || !out_shape || !out_spectral_basis) { return ENCINO_WAVES_ERROR_NULL_INPUT; }

    if (out_rank != 3 || out_shape[0] != plan->size_j || out_shape[1] != plan->size_i || out_shape[2] != 5) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    // Launch the kernel
    int32_t const needed_block_count = (plan->count + ENCINO_WAVES_BLOCK_SIZE - 1) / ENCINO_WAVES_BLOCK_SIZE;
    int32_t const blocks_per_grid =
      (max_blocks_per_grid > 0)
        ? ((max_blocks_per_grid < needed_block_count) ? max_blocks_per_grid : needed_block_count)
        : needed_block_count;

    encino_waves_spectral_basis_cuda_kernel<<<blocks_per_grid, ENCINO_WAVES_BLOCK_SIZE, 0, stream>>>(
      *plan, out_spectral_basis);

    return ENCINO_WAVES_ERROR_OK;
}

__global__ void encino_waves_spectral_height_cuda_kernel(struct Encino_waves_ocean_plan const plan,
                                                         float time,
                                                         float const* spectral_basis,
                                                         float* spectral_height_out) {
    // Uses grid striding.
    int const index_inside_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int const grid_stride = gridDim.x * blockDim.x;
    encino_waves_spectral_height_block(
      &plan, time, spectral_basis, index_inside_grid, plan.count, grid_stride, spectral_height_out);
}

ENCINO_WAVES_API
int encino_waves_spectral_height_cuda(cudaStream_t stream,
                                      struct Encino_waves_ocean_plan const* plan,
                                      float time,
                                      int max_blocks_per_grid,

                                      int in_rank,
                                      int const* in_shape,
                                      float const* spectral_basis,

                                      int out_rank,
                                      int const* out_shape,
                                      float* out_spectral_height) {
    if (!plan || !in_shape || !spectral_basis || !out_shape || !out_spectral_height) {
        return ENCINO_WAVES_ERROR_NULL_INPUT;
    }

    if (in_rank != 3 || in_shape[0] != plan->size_j || in_shape[1] != plan->size_i || in_shape[2] != 5) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    if (out_rank != 2 || out_shape[0] != plan->size_j || out_shape[1] != plan->size_i) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    // Launch the kernel
    int32_t const needed_block_count = (plan->count + ENCINO_WAVES_BLOCK_SIZE - 1) / ENCINO_WAVES_BLOCK_SIZE;
    int32_t const blocks_per_grid =
      (max_blocks_per_grid > 0)
        ? ((max_blocks_per_grid < needed_block_count) ? max_blocks_per_grid : needed_block_count)
        : needed_block_count;

    encino_waves_spectral_height_cuda_kernel<<<blocks_per_grid, ENCINO_WAVES_BLOCK_SIZE, 0, stream>>>(
      *plan, time, spectral_basis, out_spectral_height);

    return ENCINO_WAVES_ERROR_OK;
}

#endif  // ENCINO_WAVES_CUDA_KERNELS

/**********************************************************************************************
 *                                BEGIN OMP BODY SECTION                                      *
 **********************************************************************************************/

#ifdef ENCINO_WAVES_OMP_KERNELS

#include <omp.h>

ENCINO_WAVES_API
int encino_waves_spectral_basis_omp(struct Encino_waves_ocean_plan const* plan,
                                    int max_threads,
                                    int out_rank,
                                    int const* out_shape,
                                    float* out_spectral_basis) {
    if (!plan || !out_shape || !out_spectral_basis) { return ENCINO_WAVES_ERROR_NULL_INPUT; }

    if (out_rank != 3 || out_shape[0] != plan->size_j || out_shape[1] != plan->size_i || out_shape[2] != 5) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    int total = plan->count;
    int nthreads = (max_threads > 0) ? max_threads : omp_get_max_threads();

#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nprocs = omp_get_num_threads();
        int chunk = (total + nprocs - 1) / nprocs;
        int begin = tid * chunk;
        int end = (begin + chunk < total) ? (begin + chunk) : total;

        encino_waves_classic_spectral_basis_block(plan, begin, end, 1, out_spectral_basis + begin * 5);
    }

    return ENCINO_WAVES_ERROR_OK;
}

ENCINO_WAVES_API
int encino_waves_spectral_height_omp(struct Encino_waves_ocean_plan const* plan,
                                     float time,
                                     int in_rank,
                                     int const* in_shape,
                                     float const* spectral_basis,
                                     int out_rank,
                                     int const* out_shape,
                                     float* out_spectral_height) {
    if (!plan || !in_shape || !spectral_basis || !out_shape || !out_spectral_height) {
        return ENCINO_WAVES_ERROR_NULL_INPUT;
    }

    if (in_rank != 3 || in_shape[0] != plan->size_j || in_shape[1] != plan->size_i || in_shape[2] != 5) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    if (out_rank != 2 || out_shape[0] != plan->size_j || out_shape[1] != plan->size_i) {
        return ENCINO_WAVES_ERROR_INVALID_SHAPE;
    }

    int total = plan->count;
    int nthreads = omp_get_max_threads();

#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nprocs = omp_get_num_threads();
        int chunk = (total + nprocs - 1) / nprocs;
        int begin = tid * chunk;
        int end = (begin + chunk < total) ? (begin + chunk) : total;

        encino_waves_spectral_height_block(plan, time, spectral_basis, begin, end, 1, out_spectral_height + begin);
    }

    return ENCINO_WAVES_ERROR_OK;
}

#endif  // ENCINO_WAVES_OMP_KERNELS

#endif  // ENCINO_WAVES_SOURCE_CODE_HEADER_ONLY

#ifdef __cplusplus
}  // extern "C"
#endif

// //-----------------------------------------------------------------------------------------------
// // Though in general everything is built to be immutable, value-semantic, to
// // support pure functions - this is a backend helper that's not explicitly
// // visible to the user-facing API. FFTw is designed to work on memory allocated
// // in this way, so we use this as a kind of "working scratch space" for the FFT.
// class Ifft_wrapper {
// private:
//     int _N = 512;
//     Complex* _spectral_field = NULL;
//     float* _spatial_field = NULL;
//     fftwf_plan _plan = NULL;

//     void nullify() noexcept {
//         _spectral_field = NULL;
//         _spatial_field = NULL;
//         _plan = NULL;
//         _N = 0;
//     }

//     void clear() noexcept {
//         try {
//             if (_spectral_field) { fftwf_free(_spectral_field); }
//             if (_spatial_field) { fftwf_free(_spatial_field); }
//             if (_plan) { fftwf_destroy_plan(_plan); }
//         } catch (...) {}

//         nullify();
//     }

// public:
//     // Delete copy constructor and copy assignment operator
//     Ifft_wrapper(Ifft_wrapper const&) = delete;
//     Ifft_wrapper& operator=(Ifft_wrapper const&) = delete;

//     // Move constructor
//     Ifft_wrapper(Ifft_wrapper&& other) noexcept
//       : _N(other._N)
//       , _spectral_field(other._spectral_field)
//       , _spatial_field(other._spatial_field)
//       , _plan(other._plan) {
//         other.nullify();
//     }

//     // Move assignment operator
//     Ifft_wrapper& operator=(Ifft_wrapper&& other) noexcept {
//         if (this != &other) {
//             // Free existing resources
//             clear();
//             // Move resources
//             _N = other._N;
//             _spectral_field = other._spectral_field;
//             _spatial_field = other._spatial_field;
//             _plan = other._plan;

//             // Null out other's pointers
//             other.nullify();
//         }
//         return *this;
//     }

//     Ifft_wrapper(int const in_N) {
//         static bool did_init_fftw_threads = false;
//         if (!did_init_fftw_threads) {
//             fftwf_init_threads();
//             fftwf_plan_with_nthreads(omp_get_max_threads());
//             did_init_fftw_threads = true;
//         }

//         _N = in_N;

//         _spectral_field = reinterpret_cast<Complex*>(fftwf_malloc(sizeof(Complex) * _N * (_N / 2 + 1)));
//         _spatial_field = reinterpret_cast<float*>(fftwf_malloc(sizeof(float) * _N * _N));

//         _plan = fftwf_plan_dft_c2r_2d(
//           _N, _N, reinterpret_cast<fftwf_complex*>(_spectral_field), _spatial_field, FFTW_ESTIMATE);
//     }

//     ~Ifft_wrapper() {
//         clear();
//     }

//     int N() const {
//         return _N;
//     }

//     Complex* spectral_field() const {
//         return _spectral_field;
//     }

//     float const* spatial_field() const {
//         return _spatial_field;
//     }

//     void execute() {
//         fftwf_execute(_plan);
//     }
// };

// struct Global_module_state {
//     std::unordered_map<int, Ifft_wrapper> N_ifft_wrapper_map;

//     Global_module_state() = default;
//     ~Global_module_state() = default;
// };

// static std::unique_ptr<Global_module_state> g_module_state;

// Global_module_state& get_module_state() {
//     if (!g_module_state) { g_module_state = std::make_unique<Global_module_state>(); }
//     return *g_module_state;
// }

// Ifft_wrapper& get_ifft_wrapper(int const N) {
//     auto& module_state = get_module_state();
//     auto it = module_state.N_ifft_wrapper_map.find(N);
//     if (it == module_state.N_ifft_wrapper_map.end()) {
//         auto result = module_state.N_ifft_wrapper_map.emplace(N, Ifft_wrapper(N));
//         return result.first->second;
//     } else {
//         return it->second;
//     }
// }

#endif  // ENCINO_WAVES_H_INCLUDED