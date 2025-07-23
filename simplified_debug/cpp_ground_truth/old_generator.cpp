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

#include <fftw3.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if !defined(__CUDA_ARCH__)
#if defined(_WIN32)
#define ENCINO_WAVES_API __declspec(dllexport)
#else
#define ENCINO_WAVES_API __attribute__((visibility("default")))
#endif
#else
#define ENCINO_WAVES_API
#endif

extern "C" {

// C-style struct for all parameters. This is easy to map in ctypes.
struct Initial_state_parameters {
    // Resolution of the waves.
    // Must be a power of two, between 4 and 13.
    int resolution = 512;

    // Domain of the waves. - this is the size of the world space
    // that they occupy.
    float domain = 100.0f;  // in meters

    // Some physical parameters.
    float gravity = 9.81f;           // in meters per second squared.
    float surface_tension = 0.074f;  // in Newtons per meter
    float density = 1000.0f;         // in kilograms per meter cubed
    float depth = 100.0f;            // in meters.

    // Wind stuff. It is assumed that wind travels along the positive
    // X axis, since we assume these fields can be externally transformed.
    // Wind speed is in meters per second.
    float wind_speed = 17.0f;  // in meters per second
    float fetch = 300.0f;      // in KILOMETERS

    // float troughDamping = 0.0f;
    // float troughDampingSmallWavelength = 1.0f;
    // float troughDampingBigWavelength = 4.0f;
    // float troughDampingSoftWidth = 2.0f;

    float swell = 0.0f;

    float filter_soft_width = 0.0f;
    float filter_small_wavelength = 0.0f;
    float filter_big_wavelength = 1000000.0f;
    float filter_min = 0.0f;
    bool filter_invert = false;
    bool filter = false;

    int random_seed = 54321;
};

struct Propagation_parameters {
    float time = 0.0f;
    float pinch = 0.75f;          // lateral displacement
    float amplitude_gain = 1.0f;  // vertical displacement
};

ENCINO_WAVES_API int encino_waves_create_initial_state(Initial_state_parameters const* const params, int* const out_id);

ENCINO_WAVES_API int encino_waves_destroy_initial_state(int const initial_state_id);

ENCINO_WAVES_API int encino_waves_propagate(int const initial_state_id,
                                            Propagation_parameters const* const params,
                                            int const out_rank,
                                            int const* const out_shape,
                                            float* const out_height_field);

ENCINO_WAVES_API void encino_waves_shutdown();

}  // extern "C"

namespace encino_waves {

using Complex = std::complex<float>;

constexpr float TAU = 6.2831853071795f;
constexpr float PI = 3.1415926535897f;

constexpr float MIN_WAVELENGTH = 0.01f;
constexpr float MAX_WAVELENGTH = 100000.0f;

inline float ABS(float const x) {
    return std::abs(x);
}

inline float IS_FINITE(float const x) {
    return std::isfinite(x);
}

inline float SQRT(float const x) {
    return std::sqrt(x);
}

inline float TANH(float const x) {
    return std::tanh(x);
}

inline float COSH(float const x) {
    return std::cosh(x);
}

inline float COS(float const x) {
    return std::cos(x);
}

inline float SIN(float const x) {
    return std::sin(x);
}

inline float LERP(float const a, float const b, float const t) {
    return (1.0f - t) * a + t * b;
}

inline float CLAMP(float const x, float const min, float const max) {
    return std::max(min, std::min(x, max));
}

inline float SMOOTHSTEP(float const edge0, float const edge1, float const x) {
    float const t = CLAMP((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float ATAN2(float const y, float const x) {
    return std::atan2(y, x);
}

inline float POW(float const x, float const y) {
    return std::pow(x, y);
}

inline float EXP(float const x) {
    return std::exp(x);
}

inline float SQR(float const x) {
    return x * x;
}

inline float TGAMMA(float const x) {
    return std::tgamma(x);
}

using RANDGEN_ENGINE = std::minstd_rand;
using NORMAL_DISTRIBUTION = std::normal_distribution<float>;
using UNIFORM_DISTRIBUTION = std::uniform_real_distribution<float>;

// Error codes
enum Error_code {
    ERR_OK = 0,
    ERR_UNKNOWN = 1,
    ERR_INVALID_ARGUMENT = 2,
    ERR_OUT_OF_MEMORY = 3,
    // ... add more as needed
};

// Exception type for internal use
class Encino_waves_exception : public std::exception {
public:
    Encino_waves_exception(int const code, std::string const& msg)
      : _code(code)
      , _msg(msg) {
    }

    int code() const noexcept {
        return _code;
    }
    char const* what() const noexcept override {
        return _msg.c_str();
    }

private:
    int _code;
    std::string _msg;
};

// Macro for checking conditions and throwing exceptions
#define CHECK(cond, code, msg)                                        \
    do {                                                              \
        if (!(cond)) { throw Encino_waves_exception((code), (msg)); } \
    } while (0)

//-----------------------------------------------------------------------------------------------
// Though in general everything is built to be immutable, value-semantic, to
// support pure functions - this is a backend helper that's not explicitly
// visible to the user-facing API. FFTw is designed to work on memory allocated
// in this way, so we use this as a kind of "working scratch space" for the FFT.
class Ifft_wrapper {
private:
    int _N = 512;
    Complex* _spectral_field = nullptr;
    float* _spatial_field = nullptr;
    fftwf_plan _plan = nullptr;

    void nullify() noexcept {
        _spectral_field = nullptr;
        _spatial_field = nullptr;
        _plan = nullptr;
        _N = 0;
    }

    void clear() noexcept {
        try {
            if (_spectral_field) { fftwf_free(_spectral_field); }
            if (_spatial_field) { fftwf_free(_spatial_field); }
            if (_plan) { fftwf_destroy_plan(_plan); }
        } catch (...) {}

        nullify();
    }

public:
    // Delete copy constructor and copy assignment operator
    Ifft_wrapper(Ifft_wrapper const&) = delete;
    Ifft_wrapper& operator=(Ifft_wrapper const&) = delete;

    // Move constructor
    Ifft_wrapper(Ifft_wrapper&& other) noexcept
      : _N(other._N)
      , _spectral_field(other._spectral_field)
      , _spatial_field(other._spatial_field)
      , _plan(other._plan) {
        other.nullify();
    }

    // Move assignment operator
    Ifft_wrapper& operator=(Ifft_wrapper&& other) noexcept {
        if (this != &other) {
            // Free existing resources
            clear();
            // Move resources
            _N = other._N;
            _spectral_field = other._spectral_field;
            _spatial_field = other._spatial_field;
            _plan = other._plan;

            // Null out other's pointers
            other.nullify();
        }
        return *this;
    }

    Ifft_wrapper(int const in_N) {
        static bool did_init_fftw_threads = false;
        if (!did_init_fftw_threads) {
            fftwf_init_threads();
            fftwf_plan_with_nthreads(omp_get_max_threads());
            did_init_fftw_threads = true;
        }

        _N = in_N;

        _spectral_field = reinterpret_cast<Complex*>(fftwf_malloc(sizeof(Complex) * _N * (_N / 2 + 1)));
        _spatial_field = reinterpret_cast<float*>(fftwf_malloc(sizeof(float) * _N * _N));

        _plan = fftwf_plan_dft_c2r_2d(
          _N, _N, reinterpret_cast<fftwf_complex*>(_spectral_field), _spatial_field, FFTW_ESTIMATE);
    }

    ~Ifft_wrapper() {
        clear();
    }

    int N() const {
        return _N;
    }

    Complex* spectral_field() const {
        return _spectral_field;
    }

    float const* spatial_field() const {
        return _spatial_field;
    }

    void execute() {
        fftwf_execute(_plan);
    }
};

struct Initial_state {
    Initial_state_parameters params;
    int N;
    int size_i;
    int size_j;
    int count;
    std::vector<float> data;
};

struct Global_module_state {
    std::unordered_map<int, Ifft_wrapper> N_ifft_wrapper_map;

    int next_id = 1;
    std::unordered_map<int, Initial_state> id_initial_state_map;

    Global_module_state() = default;
    ~Global_module_state() = default;
};

static std::unique_ptr<Global_module_state> g_module_state;

Global_module_state& get_module_state() {
    if (!g_module_state) { g_module_state = std::make_unique<Global_module_state>(); }
    return *g_module_state;
}

Ifft_wrapper& get_ifft_wrapper(int const N) {
    auto& module_state = get_module_state();
    auto it = module_state.N_ifft_wrapper_map.find(N);
    if (it == module_state.N_ifft_wrapper_map.end()) {
        auto result = module_state.N_ifft_wrapper_map.emplace(N, Ifft_wrapper(N));
        return result.first->second;
    } else {
        return it->second;
    }
}

int store_initial_state(Initial_state&& initial_state) {
    auto& module_state = get_module_state();
    auto id = module_state.next_id++;
    module_state.id_initial_state_map.emplace(id, std::move(initial_state));
    return id;
}

Initial_state& get_initial_state(int const id) {
    auto& module_state = get_module_state();
    auto it = module_state.id_initial_state_map.find(id);
    CHECK(it != module_state.id_initial_state_map.end(), ERR_INVALID_ARGUMENT, "Initial state not found");
    return it->second;
}

void destroy_initial_state(int const id) {
    auto& module_state = get_module_state();
    auto it = module_state.id_initial_state_map.find(id);
    CHECK(it != module_state.id_initial_state_map.end(), ERR_INVALID_ARGUMENT, "Initial state not found");
    module_state.id_initial_state_map.erase(id);
}

Initial_state original_initial_state(Initial_state_parameters const& params) {
    Initial_state initial_state;

    // Check that params.resolution is a power of two between 4 and 8192 (inclusive)
    CHECK(params.resolution >= 4 && params.resolution <= 8192 && (params.resolution & (params.resolution - 1)) == 0,
          ERR_INVALID_ARGUMENT,
          "resolution must be a power of two between 4 and 8192");

    int const N = params.resolution;
    int const size_i = (N / 2) + 1;
    int const size_j = N;
    int const count = size_i * size_j;

    initial_state.params = params;
    initial_state.N = N;
    initial_state.size_i = size_i;
    initial_state.size_j = size_j;
    initial_state.count = count;
    initial_state.data.resize(count * 5);
    auto* const data = initial_state.data.data();

    auto const L = CLAMP(ABS(params.domain), MIN_WAVELENGTH, MAX_WAVELENGTH);
    auto const g = CLAMP(ABS(params.gravity), 0.1f, 100.0f);

    // Plausible physical range of surface tensions are 0.008 (liquid
    // nitrogen) to 0.485 (mercury).
    auto const surface_tension = CLAMP(ABS(params.surface_tension), 0.001f, 1.0f);

    // Plausible physical range of densities are 0.001 (liquid helium) to
    // 10000 (osmium).
    auto const rho = CLAMP(ABS(params.density), 0.001f, 10000.0f);

    // Deepest ocean depth is 10,984 meters, at the Challenger Deep in the
    // Mariana Trench.
    auto const depth = CLAMP(ABS(params.depth), 0.01f, 20000.0f);

    // Highest windspeed ever recorded on earth (over an ocean) is 113.3
    // m/s, on April 10, 1996, near Barrow Island, Australia, as part of
    // Severe Tropical Cyclone Olivia.
    auto const wind_speed = CLAMP(ABS(params.wind_speed), 0.01f, 300.0f);

    // The longest fetch for a single storm is between 4000 and 5000 km, for
    // a massive extratropical cyclone. The theoretical maximum fetch is
    // 10,000 km, but that's for the persistent global wind belt of the
    // Southern Ocean Westerlies.
    auto const fetch = CLAMP(ABS(params.fetch), 0.01f, 20000.0f);
    auto const swell = CLAMP(params.swell, -1.0f, 1.0f);
    auto const filter_soft_width = CLAMP(ABS(params.filter_soft_width), 0.0001f, 100000.0f);
    auto const filter_small_wavelength = CLAMP(ABS(params.filter_small_wavelength), MIN_WAVELENGTH, MAX_WAVELENGTH);
    auto const filter_big_wavelength = CLAMP(ABS(params.filter_big_wavelength), MIN_WAVELENGTH, MAX_WAVELENGTH);
    auto const filter_min = CLAMP(ABS(params.filter_min), 0.0f, 1.0f);
    auto const filter_invert = params.filter_invert;
    auto const filter = params.filter;
    auto const random_seed = params.random_seed;

    // Derived constants
    auto const dK = TAU / L;
    auto const dK2 = dK * dK;

    auto const max_k_mag = (N / 2) * dK;

    auto const sigma_over_rho = surface_tension / rho;

    float tma_gamma;
    {
        thread_local RANDGEN_ENGINE gamma_randgen_engine{static_cast<uint64_t>(params.random_seed) + 191819};
        thread_local NORMAL_DISTRIBUTION gamma_normal_distribution{3.30f, SQRT(0.67f)};
        tma_gamma = CLAMP(gamma_normal_distribution(gamma_randgen_engine), 1.0f, 6.0f);
    }

    auto const fetch_m = fetch * 1000.0f;
    auto const dimensionless_fetch = ABS(g * fetch_m / SQR(wind_speed));
    auto const tma_alpha = 0.076f * POW(dimensionless_fetch, -0.22f);
    auto const peak_omega = TAU * 3.5f * ABS(g / wind_speed) * POW(dimensionless_fetch, -0.33f);
    auto const tma_kd_gain = SQRT(depth / g);

    // Hasselmann Directional Spreading
    auto const modal_shape = 11.5f * POW(peak_omega * wind_speed / g, -2.5f);
    auto const modal_celerity = g / peak_omega;
    auto const wind_speed_over_celerity = wind_speed / modal_celerity;

    auto const filter_edge0 = filter_small_wavelength - filter_soft_width;
    auto const filter_edge1 = filter_small_wavelength;
    auto const filter_edge2 = filter_big_wavelength;
    auto const filter_edge3 = filter_big_wavelength + filter_soft_width;

    NORMAL_DISTRIBUTION amp_distribution{0.0f, 1.0f};
    UNIFORM_DISTRIBUTION phase_distribution{0.0f, TAU};

    // #pragma omp parallel for
    for (int wave_number_index = 0; wave_number_index < count; ++wave_number_index) {
        int j = wave_number_index / size_i;
        int real_j = j < (N / 2) ? j : j - N;
        int i = wave_number_index % size_i;

        // ki, kj are the wave numbers in the i and j directions.
        float const ki = i * dK;
        float const kj = real_j * dK;
        float const k_mag = SQRT(ki * ki + kj * kj);
        float const k_mag_squared = k_mag * k_mag;

        // Get out for the DC component and the highest wave numbers, to
        // avoid aliasing.
        if ((i == 0 && real_j == 0) || (k_mag > max_k_mag)) {
            data[wave_number_index * 5 + 0] = 0.0f;
            data[wave_number_index * 5 + 1] = 0.0f;
            data[wave_number_index * 5 + 2] = 0.0f;
            data[wave_number_index * 5 + 3] = 0.0f;
            data[wave_number_index * 5 + 4] = 0.0f;
            continue;
        }

        // Seed the random number generator from the wave numbers.
        thread_local RANDGEN_ENGINE randgen_engine{1};
        {
            constexpr uint32_t p1 = 73856093;
            constexpr uint32_t p2 = 19349663;
            constexpr uint32_t p3 = 83492791;

            // Truncate the ks to some precision, and then make them into seeds.
            uint32_t const seed =
              (static_cast<uint32_t>(ki * 10000) * p1) ^ (static_cast<uint32_t>(kj * 10000) * p2) ^ (random_seed * p3);

            randgen_engine.seed(seed);
        }

        // get thetaPos and thetaNeg from k.
        float const theta_pos = ATAN2(-kj, ki);
        float const theta_neg = ATAN2(kj, -ki);
        CHECK(IS_FINITE(theta_pos) && IS_FINITE(theta_neg), ERR_INVALID_ARGUMENT, "NAN/INF theta pos or theta neg");

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
            auto const hk = depth * k_mag;
            auto const k2s = k_mag_squared * sigma_over_rho;
            auto const gpk2s = g + k2s;

            omega = SQRT(ABS(k_mag * gpk2s * TANH(hk)));

            auto const numer = ((gpk2s + k2s + k2s) * TANH(hk)) + (hk * gpk2s / SQR(COSH(hk)));

            domega_dk = ABS(numer) / (2.0f * omega);
            if (!IS_FINITE(omega) || !IS_FINITE(domega_dk)) {
                omega = 1.0f;
                domega_dk = 0.0f;
            }
        }

        float spectrum;
        {
            // The JONSWAP spectrum introduces the fetch parameter. It is an
            // AlphaBeta spectrum times a peak sharpening function.  The peak sharpening
            // coefficient, gamma, is chosen as a random draw.
            // alpha = 0.076 * pow( xbar, -0.22 )
            // sigma = 0.07 for w < wm, 0.09 for w > wm
            // wm = TAU * 3.5 * ( g / U ) * pow( xbar, -0.33 )
            // xbar = g F / U^2
            // F = fetch
            // U = wind speed
            // g = gravity
            // y = gaussian( mean=3.30, variance=0.62 ), clamped from 1 to 6
            // peakSharpening = pow( y, exp( -(w-wm)^2/2(sigma wm)^2 )
            // beta = 1.25
            auto const sigma = omega <= peak_omega ? 0.07f : 0.09f;
            auto const peak_sharpening = POW(tma_gamma, EXP(-SQR((omega - peak_omega) / (sigma * peak_omega)) / 2.0f));
            auto const jonswap_alpha_beta =
              (tma_alpha * SQR(g) / POW(omega, 5.0f)) * EXP(-1.25f * POW(peak_omega / omega, 4.0f));

            auto const wh = omega * tma_kd_gain;
            auto const kitaigorodskii_depth = 0.5f + 0.5f * TANH(1.8f * (wh - 1.125f));

            spectrum = peak_sharpening * jonswap_alpha_beta * kitaigorodskii_depth;
            if (!IS_FINITE(spectrum)) { spectrum = 0.0f; }
        }

        // Attenuate by directional spreading
        float dir_spread_pos;
        float dir_spread_neg;
        {
            // Hasselmann Directional Spreading
            float shape_bias = 0.0;
            if (swell >= 0.0f) { shape_bias = 16.1f * TANH(peak_omega / omega) * SQR(swell); }
            float shape;
            if (omega > peak_omega) {
                shape = 9.77f * POW(omega / peak_omega, -2.33f - (1.45f * (wind_speed_over_celerity - 1.17f)));
            } else {
                shape = 6.97f * POW(omega / peak_omega, 4.06f);
            }
            shape += shape_bias;
            auto const factor_a = POW(2.0f, (2.0f * shape) - 1.0f) / PI;
            auto const factor_b = SQR(TGAMMA(shape + 1.0f)) / TGAMMA((2.0f * shape) + 1.0f);
            auto const factor_c_pos = POW(ABS(COS(theta_pos / 2.0f)), 2.0f * shape);
            auto const factor_c_neg = POW(ABS(COS(theta_neg / 2.0f)), 2.0f * shape);

            dir_spread_pos = factor_a * factor_b * factor_c_pos;
            dir_spread_neg = factor_a * factor_b * factor_c_neg;

            if (swell < 0.0f) {
                dir_spread_pos = LERP(dir_spread_pos, 1.0f / TAU, -swell);
                dir_spread_neg = LERP(dir_spread_neg, 1.0f / TAU, -swell);
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
        auto const change_of_variables_factor = (dK * dK) * ABS(domega_dk / k_mag);
        auto const delta_s_pos = spectrum * dir_spread_pos * change_of_variables_factor;
        auto const delta_s_neg = spectrum * dir_spread_neg * change_of_variables_factor;

        float amp_pos = amp_distribution(randgen_engine) * SQRT(ABS(delta_s_pos * 2.0f));
        if (!IS_FINITE(amp_pos)) { amp_pos = 0.0f; }
        float amp_neg = amp_distribution(randgen_engine) * SQRT(ABS(delta_s_neg * 2.0f));
        if (!IS_FINITE(amp_neg)) { amp_neg = 0.0f; }

        // Filter amplitudes. Filter has to be outside the sqrt so that it
        // is properly invertible.
        if (filter) {
            auto const band =
              SMOOTHSTEP(filter_edge0, filter_edge1, k_mag) - SMOOTHSTEP(filter_edge2, filter_edge3, k_mag);
            auto filt = LERP(filter_min, 1.0f, band);
            if (filter_invert) { filt = 1.0f - filt; }
            amp_pos *= filt;
            amp_neg *= filt;
        }

        // Final results!
        // Get random draws for phase.
        auto const phase_pos = phase_distribution(randgen_engine);
        auto const phase_neg = phase_distribution(randgen_engine);

        auto const h_pos_real = amp_pos * COS(phase_pos);
        auto const h_pos_imag = amp_pos * -SIN(phase_pos);
        auto const h_neg_real = amp_neg * COS(phase_neg);
        auto const h_neg_imag = amp_neg * -SIN(phase_neg);

        data[wave_number_index * 5 + 0] = h_pos_real;
        data[wave_number_index * 5 + 1] = h_pos_imag;
        data[wave_number_index * 5 + 2] = h_neg_real;
        data[wave_number_index * 5 + 3] = h_neg_imag;
        data[wave_number_index * 5 + 4] = omega;
    }

    return initial_state;
}

void propagate(Initial_state const& initial_state,
               Propagation_parameters const& params,
               float* const out_height_field) {
    auto& ifft = get_ifft_wrapper(initial_state.N);
    auto* const spectral = reinterpret_cast<Complex*>(ifft.spectral_field());

    // #pragma omp parallel for
    for (int i = 0; i < initial_state.count; ++i) {
        auto const h_pos_real = initial_state.data[i * 5 + 0];
        auto const h_pos_imag = initial_state.data[i * 5 + 1];
        auto const h_neg_real = initial_state.data[i * 5 + 2];
        auto const h_neg_imag = initial_state.data[i * 5 + 3];
        auto const omega = initial_state.data[i * 5 + 4];

        Complex const h_pos{h_pos_real, h_pos_imag};
        Complex const h_neg{h_neg_real, h_neg_imag};

        auto const cos_omega_t = COS(omega * params.time);
        auto const sin_omega_t = SIN(omega * params.time);
        Complex const fwd{cos_omega_t, -sin_omega_t};
        Complex const bkwd{cos_omega_t, sin_omega_t};

        spectral[i] = (h_pos * fwd) + (h_neg * bkwd);
    }

    ifft.execute();

    auto* const spatial = ifft.spatial_field();

    auto const count = initial_state.N * initial_state.N;

#pragma omp parallel for
    for (int i = 0; i < count; ++i) { out_height_field[i] = params.amplitude_gain * spatial[i]; }
}

}  // namespace encino_waves

extern "C" {

ENCINO_WAVES_API int encino_waves_create_initial_state(Initial_state_parameters const* const params,
                                                       int* const out_id) {
    using namespace encino_waves;

    if (!params || !out_id) { return ERR_INVALID_ARGUMENT; }

    int id = 0;
    try {
        id = store_initial_state(original_initial_state(*params));
    } catch (Encino_waves_exception const& e) { return e.code(); } catch (...) {
        return ERR_UNKNOWN;
    }

    *out_id = id;
    return ERR_OK;
}

ENCINO_WAVES_API int encino_waves_destroy_initial_state(int const initial_state_id) {
    using namespace encino_waves;
    if (initial_state_id < 1) { return ERR_INVALID_ARGUMENT; }

    try {
        destroy_initial_state(initial_state_id);
    } catch (Encino_waves_exception const& e) { return e.code(); } catch (...) {
        return ERR_UNKNOWN;
    }

    return ERR_OK;
}

ENCINO_WAVES_API int encino_waves_propagate(int const initial_state_id,
                                            Propagation_parameters const* const params,
                                            int const out_rank,
                                            int const* const out_shape,
                                            float* const out_height_field) {
    using namespace encino_waves;
    if (initial_state_id < 1 || !params || !out_height_field || !out_shape) { return ERR_INVALID_ARGUMENT; }

    if (out_rank != 2) { return ERR_INVALID_ARGUMENT; }

    try {
        auto& initial_state = get_initial_state(initial_state_id);

        if (out_shape[0] != initial_state.N || out_shape[1] != initial_state.N) { return ERR_INVALID_ARGUMENT; }

        propagate(initial_state, *params, out_height_field);
    } catch (Encino_waves_exception const& e) { return e.code(); } catch (...) {
        return ERR_UNKNOWN;
    }

    return ERR_OK;
}

ENCINO_WAVES_API void encino_waves_shutdown() {
    using namespace encino_waves;
    fftwf_cleanup_threads();
    g_module_state.reset();
}

}  // extern "C"