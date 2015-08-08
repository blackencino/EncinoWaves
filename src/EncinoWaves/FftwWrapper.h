//-*****************************************************************************
// Copyright 2015 Christopher Jon Horvath
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
//-*****************************************************************************

//-*****************************************************************************
// The basic architecture of these Waves is based on the TweakWaves application
// written by Chris Horvath for Tweak Films in 2001.  This, in turn, was based
// on the SIGGRAPH papers and courses by Jerry Tessendorf, and by the paper
// "A Simple Fluid Solver based on the FTT" by Jos Stam.
//
// The TMA, JONSWAP, and Pierson Moskowitz Wave Spectra, as well as the
// directional spreading functions are formulated based on the descriptions
// given in "Ocean Waves: The Stochastic Approach",
// by Michel K. Ochi, published by Cambridge Ocean Technology Series, 1998,2005.
//
// This library is written as a working implementation of the paper:
// Christopher J. Horvath. 2015.
// Empirical directional wave spectra for computer graphics.
// In Proceedings of the 2015 Symposium on Digital Production (DigiPro '15),
// Los Angeles, Aug. 8, 2015, pp. 29-39.
//-*****************************************************************************

#ifndef _EncinoWaves_FftwWrapper_h_
#define _EncinoWaves_FftwWrapper_h_

#include "Foundation.h"

namespace EncinoWaves {

//-*****************************************************************************
// Template basis - never used directly, specialized explicitly below.
template <typename T>
struct FftwWrapperT;

//-*****************************************************************************
// SINGLE PRECISION
// This just wraps fftw function calls, nothing more.
template <>
struct FftwWrapperT<float>
{
    typedef float               real_type;
    typedef std::complex<float> complex_type;
    typedef fftwf_plan          plan_type;
    typedef fftwf_iodim         iodim_type;

    // Init threads. Only call once at the very begining.
    static int init_threads( void )
    { return fftwf_init_threads(); }

    // Configure next plan's number of threads.
    static void plan_with_nthreads( int i_nthreads )
    { fftwf_plan_with_nthreads( i_nthreads ); }

    // Create a complex-to-real, 2d plan.
    static plan_type plan_dft_c2r_2d( int i_width, int i_height,
                                      complex_type* i_in, real_type* o_out,
                                      unsigned int i_flags )
    { return fftwf_plan_dft_c2r_2d( i_width, i_height,
                                    reinterpret_cast<fftwf_complex*>( i_in ),
                                    o_out, i_flags ); }

    // Use the guru interface to create the same plan.
    static plan_type plan_guru_dft_c2r( int i_width, int i_height,
                                        complex_type* i_in, real_type* o_out,
                                        unsigned int i_flags )
    {
        int rank = 2;
        iodim_type dims[2];
        dims[0].n = i_height;
        dims[0].is = (i_width/2)+1;
        dims[0].os = i_width;

        dims[1].n = i_width;
        dims[1].is = 1;
        dims[1].os = 1;

        int howmany = 1;
        iodim_type howmanydims[1];
        howmanydims[0].n = 1;
        howmanydims[0].is = ((i_width/2)+1)*i_height;
        howmanydims[0].os = i_width*i_height;

        return fftwf_plan_guru_dft_c2r(
            rank, dims, howmany, howmanydims,
            reinterpret_cast<fftwf_complex*>( i_in ), o_out, i_flags );
    }

    // Use the guru interface to provide a padded output
    static plan_type plan_guru_dft_c2r_output_padded(
        int i_width, int i_height, int i_widthPad, int i_heightPad,
        complex_type* i_in, real_type* o_out, unsigned int i_flags )
    {
        int rank = 2;
        iodim_type dims[2];
        dims[0].n = i_height;
        dims[0].is = (i_width/2)+1;
        dims[0].os = i_width+i_widthPad;

        dims[1].n = i_width;
        dims[1].is = 1;
        dims[1].os = 1;

        int howmany = 1;
        iodim_type howmanydims[1];
        howmanydims[0].n = 1;
        howmanydims[0].is = ((i_width/2)+1)*i_height;
        howmanydims[0].os = (i_width+i_widthPad)*(i_height+i_heightPad);

        return fftwf_plan_guru_dft_c2r(
            rank, dims, howmany, howmanydims,
            reinterpret_cast<fftwf_complex*>( i_in ), o_out, i_flags );
    }

    // Malloc some data. Capitalized to make very sure it isn't
    // confused with system malloc.
    static void* Malloc( size_t i_size )
    { return fftwf_malloc( i_size ); }

    // Free some data. Capitalized to make very sure it isn't confused
    // with system free.
    static void Free( void* i_data )
    { fftwf_free( i_data ); }

    // Execute a plan.
    static void execute( const plan_type i_plan )
    { fftwf_execute( i_plan ); }

    // Execute a plan on other data.
    static void execute_dft_c2r( const plan_type i_plan,
                                 complex_type* i_in, real_type* o_out )
    { fftwf_execute_dft_c2r( i_plan,
                             reinterpret_cast<fftwf_complex*>( i_in ),
                             o_out ); }

    // Destroy a plan.
    static void destroy_plan( const plan_type i_plan )
    { fftwf_destroy_plan( i_plan ); }

    // Cleanup threads. Only call at the very end.
    static void cleanup_threads( void )
    { fftwf_cleanup_threads(); }

    // Cleanup.
    static void cleanup( void )
    { fftwf_cleanup(); }
};

//-*****************************************************************************
// DOUBLE PRECISION
// This just wraps fftw function calls, nothing more.
template <>
struct FftwWrapperT<double>
{
    typedef double                  real_type;
    typedef std::complex<double>    complex_type;
    typedef fftw_plan               plan_type;
    typedef fftw_iodim              iodim_type;

    // Init threads. Only call once at the very begining.
    static int init_threads( void )
    { return fftw_init_threads(); }

    // Configure next plan's number of threads.
    static void plan_with_nthreads( int i_nthreads )
    { fftw_plan_with_nthreads( i_nthreads ); }

    // Create a complex-to-real, 2d plan.
    static plan_type plan_dft_c2r_2d( int i_width, int i_height,
                                      complex_type* i_in, real_type* o_out,
                                      unsigned int i_flags )
    { return fftw_plan_dft_c2r_2d( i_width, i_height,
                                   reinterpret_cast<fftw_complex*>( i_in ),
                                   o_out, i_flags ); }

    // Use the guru interface to create the same plan.
    static plan_type plan_guru_dft_c2r( int i_width, int i_height,
                                        complex_type* i_in, real_type* o_out,
                                        unsigned int i_flags )
    {
        int rank = 2;
        iodim_type dims[2];
        dims[0].n = i_height;
        dims[0].is = (i_width/2)+1;
        dims[0].os = i_width;

        dims[1].n = i_width;
        dims[1].is = 1;
        dims[1].os = 1;

        int howmany = 1;
        iodim_type howmanydims[1];
        howmanydims[0].n = 1;
        howmanydims[0].is = ((i_width/2)+1)*i_height;
        howmanydims[0].os = i_width*i_height;

        return fftw_plan_guru_dft_c2r(
            rank, dims, howmany, howmanydims,
            reinterpret_cast<fftw_complex*>( i_in ), o_out, i_flags );
    }

    // Use the guru interface to provide a padded output
    static plan_type plan_guru_dft_c2r_output_padded(
        int i_width, int i_height, int i_widthPad, int i_heightPad,
        complex_type* i_in, real_type* o_out, unsigned int i_flags )
    {
        int rank = 2;
        iodim_type dims[2];
        dims[0].n = i_height;
        dims[0].is = (i_width/2)+1;
        dims[0].os = i_width+i_widthPad;

        dims[1].n = i_width;
        dims[1].is = 1;
        dims[1].os = 1;

        int howmany = 1;
        iodim_type howmanydims[1];
        howmanydims[0].n = 1;
        howmanydims[0].is = ((i_width/2)+1)*i_height;
        howmanydims[0].os = (i_width+i_widthPad)*(i_height+i_heightPad);

        return fftw_plan_guru_dft_c2r(
            rank, dims, howmany, howmanydims,
            reinterpret_cast<fftw_complex*>( i_in ), o_out, i_flags );
    }

    // Malloc some data. Capitalized to make very sure it isn't
    // confused with system malloc.
    static void* Malloc( size_t i_size )
    { return fftw_malloc( i_size ); }

    // Free some data. Capitalized to make very sure it isn't confused
    // with system free.
    static void Free( void* i_data )
    { fftw_free( i_data ); }

    // Execute a plan.
    static void execute( const plan_type i_plan )
    { fftw_execute( i_plan ); }

    // Execute a plan on other data.
    static void execute_dft_c2r( const plan_type i_plan,
                                 complex_type* i_in, real_type* o_out )
    { fftw_execute_dft_c2r( i_plan,
                            reinterpret_cast<fftw_complex*>( i_in ),
                            o_out ); }

    // Destroy a plan.
    static void destroy_plan( const plan_type i_plan )
    { fftw_destroy_plan( i_plan ); }

    // Cleanup threads. Only call at the very end.
    static void cleanup_threads()
    { fftw_cleanup_threads(); }

    // Cleanup.
    static void cleanup( void )
    { fftw_cleanup(); }
};

//-*****************************************************************************
//-*****************************************************************************
// GLOBAL THREAD INIT NONSENSE.
//-*****************************************************************************
//-*****************************************************************************

template <typename T>
struct FftwInitThreadsT_InitHelper
{
    typedef FftwWrapperT<T> FFT;

    FftwInitThreadsT_InitHelper()
    {
        int err = FFT::init_threads();
        EWAV_ASSERT( err != 0, "FFTW thread init error." );
    }
    ~FftwInitThreadsT_InitHelper() { FFT::cleanup_threads(); }
};

template <typename T> struct __BaseFftwInitThreadsT;

template <>
struct __BaseFftwInitThreadsT<float>
{
    typedef FftwInitThreadsT_InitHelper<float> Init;
    static std::unique_ptr<Init> sm_init;
};

template <>
struct __BaseFftwInitThreadsT<double>
{
    typedef FftwInitThreadsT_InitHelper<double> Init;
    static std::unique_ptr<Init> sm_init;
};

//-*****************************************************************************
template <typename T>
inline void FftwInitThreadsT()
{
    typedef __BaseFftwInitThreadsT<T> Base;
    typedef typename Base::Init Init;

    if ( !Base::sm_init )
    { Base::sm_init.reset( new typename Base::Init ); }
};

} // namespace EncinoWaves

#endif
