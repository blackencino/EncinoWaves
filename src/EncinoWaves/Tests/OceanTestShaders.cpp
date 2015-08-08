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

#include "OceanTestShaders.h"

namespace OceanTest {

//-*****************************************************************************
// Header.
static const char* g_shaderHeader =
  R"(
    #version 150
)";

//-*****************************************************************************
static const char* g_vertexShaderBase =
  R"(
    in float g_h;
    in float g_dx;
    in float g_dy;
    in float g_minE;
    in vec2 g_vertex;
    in vec3 g_normal;

    uniform float g_pinch;
    uniform float g_amplitude;
    uniform float g_gainMinE;
    uniform float g_biasMinE;

    uniform mat4 projection_matrix;
    uniform mat4 modelview_matrix;

    out vec3 g_Pwld;
    out vec3 g_Nwld;
    out float g_crest;

    void main()
    {
       mat4 m = mat4( projection_matrix );
       mat4 pmvMat = projection_matrix * modelview_matrix;
       vec4 vtx = vec4(
           ( g_vertex.x - g_pinch * g_dx ),
           ( g_vertex.y - g_pinch * g_dy ),
           ( g_amplitude * g_h ),
           1 );

        g_Pwld = vec3( vtx );
        g_Nwld = normalize( g_normal );
        g_crest = ( g_minE * g_gainMinE ) + g_biasMinE;

        //gl_Position = projection_matrix * modelview_matrix * vtx;
        gl_Position = vtx;
    }
)";

//-*****************************************************************************
static const char* g_simpleVertexShaderBase =
  R"(
    in vec2 g_vertex;
    uniform mat4 projection_matrix;
    uniform mat4 modelview_matrix;
    void main()
    {
       mat4 pmvMat = projection_matrix * modelview_matrix;
       vec4 vtx = vec4( g_vertex.x, g_vertex.y, 0, 1 );
       gl_Position = pmvMat * vtx;
    }

)";

//-*****************************************************************************
static const char* g_simpleFragmentShaderBase =
  R"(
    out vec4 g_fragmentColor;
    void main()
    {
       g_fragmentColor = vec4( 1.0, 0.5, 0.1, 1.0 );
    }
)";

//-*****************************************************************************
static const char* g_geometryShaderBase =
  R"(
layout(triangles) in;
layout(triangle_strip, max_vertices = 27) out;
in vec3 g_Pwld[3];
in vec3 g_Nwld[3];
in float g_crest[3];
out vec3 gs_Pwld;
out vec3 gs_Nwld;
out float gs_crest;

uniform mat4 projection_matrix;
uniform mat4 modelview_matrix;
uniform float g_domain;
uniform int g_repeat;

void main(void) {
    int r = clamp( g_repeat, 1, 3 );
    vec4 input_origin = vec4( -0.5 * g_domain, -0.5 * g_domain, 0, 0 );
    vec4 repeat_origin = vec4( -0.5 * float( r ) * g_domain,
                        -0.5 * float( r ) * g_domain, 0, 0 );
    vec4 offset_origin = repeat_origin - input_origin;
    mat4 pmvMat = projection_matrix * modelview_matrix;
    for ( int kj = 0; kj < r; ++kj ) {
    for ( int ki = 0; ki < r; ++ki ) {
        vec4 offset = offset_origin + vec4( g_domain * float( ki ),
                            g_domain * float( kj ), 0, 0 );
        for (int i = 0; i < gl_in.length(); ++i) {
                gl_Position = pmvMat * ( offset + gl_in[i].gl_Position );
                gs_Pwld = g_Pwld[i] + offset.xyz;
                gs_Nwld = g_Nwld[i];
                gs_crest = g_crest[i];
                EmitVertex();
        }
        EndPrimitive();
   }
   }
}
)";

//-*****************************************************************************
static const char* g_simpleGeometryShaderBase =
  R"(
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec3 g_Pwld[3];
in vec3 g_Nwld[3];
in float g_crest[3];
out vec3 gs_Pwld;
out vec3 gs_Nwld;
out float gs_crest;

uniform mat4 projection_matrix;
uniform mat4 modelview_matrix;
uniform vec4 g_offset;

void main(void) {
    mat4 pmvMat = projection_matrix * modelview_matrix;
    for (int i = 0; i < 3; ++i) {
        gl_Position = pmvMat * ( g_offset + gl_in[i].gl_Position );
        gs_Pwld = g_Pwld[i] + g_offset.xyz;
        gs_Nwld = g_Nwld[i];
        gs_crest = g_crest[i];
        EmitVertex();
    }
    EndPrimitive();
}
)";


//-*****************************************************************************
static const char* g_fragmentShaderBaseSimple =
  R"(
    out vec4 gl_FragColor;
    void main( void ) {
        gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
    }
)";

//-*****************************************************************************
static const char* g_fragmentShaderBase =
  R"(
    in vec3 PREFIXIN_Pwld;
    in vec3 PREFIXIN_Nwld;
    in float PREFIXIN_crest;
    out vec4 g_fragmentColor;

    uniform vec3 g_eyeWld;
    uniform float g_domain;

    //-***************************************************************
    float sqr( float a ) { return a * a; }

    //-***************************************************************
    float linstep( float edge0, float edge1, float t )
    {
        return clamp( ( t - edge0 ) / ( edge1 - edge0 ),
                      0, 1 );
    }

    //-***************************************************************
    void myRefract( in vec3 In, in vec3 Nn,
                    in float ni, in float nt,
                    out vec3 R, out vec3 T,
                    out float kr, out float kt )
    {
        float eta = ni / nt;
        vec3 Vn = -In;
        float cosThetaI = dot( Vn, Nn );
        vec3 a = Vn - ( cosThetaI * Nn );
        vec3 b = -eta * a;
        float sinThetaI = sqrt(
            clamp( 1.0 - sqr( cosThetaI ), 0.0, 1.0 ) );
        float sinThetaT = eta * sinThetaI;
        float cosThetaT = sqrt(
            clamp( 1.0 - sqr( sinThetaT ), 0.0, 1.0 ) );
        R = normalize( Vn - 2.0 * a );
        T = normalize( b - cosThetaT * Nn );

        float rs = ( (ni*cosThetaI) - (nt*cosThetaT) ) /
           ( (ni*cosThetaI) + (nt*cosThetaT) );

        float rp = ( (ni*cosThetaT) - (nt*cosThetaI) ) /
           ( (ni*cosThetaT) + (nt*cosThetaI) );

        kr = ( sqr( rp ) + sqr( rs ) ) / 2.0;
        kt = ( 1.0 - kr ) / sqr( eta );
    }

    //-***************************************************************
    vec3 gammaCorrect( in vec3 col, in float g )
    {
        return vec3( pow( col.r, 1.0/g ),
                     pow( col.g, 1.0/g ),
                     pow( col.b, 1.0/g ) );
    }

    //-***************************************************************
    // Help out the phase function below.
    float HGPhaseSingle( float PhaseG, float CosTheta )
    {
        return ( 1.0 - PhaseG*PhaseG ) /
            pow( 1.0 + PhaseG*PhaseG - 2.0*PhaseG*CosTheta,
                 3.0/2.0 );
    }

    //-***************************************************************
    // This returns how much the light bounces back at the
    // light source in a participating media.
    vec3 HGPhase( vec3 PhaseG, float Theta )
    {
        float cosTheta = cos( Theta );
        return vec3( HGPhaseSingle( PhaseG.x, cosTheta ),
                     HGPhaseSingle( PhaseG.y, cosTheta ),
                     HGPhaseSingle( PhaseG.z, cosTheta ) )
            / ( 4.0 * 3.141592 );
    }

    //-***************************************************************
    float FullExtinctionChannel( float sigma,
                                 float InZ,
                                 float LnZ,
                                 float startDepth )
    {
        return exp( -startDepth * sigma / LnZ ) * LnZ /
            ( ( InZ + LnZ ) * sigma );
    }

    //-***************************************************************
    float AngleBetweenUnitVectors( vec3 A, vec3 B )
    {
        return acos( clamp( dot( A, B ), -1.0, 1.0 ) );
    }

    //-***************************************************************
    vec3 FullExtinctionColor( vec3 i_LightColor,
                              vec3 i_Extinction,
                              vec3 i_Scattering,
                              vec3 i_PhaseG,

                              vec3 i_DirectionToLight,
                              vec3 i_IncidentVector,

                              float i_OceanSurfaceHeight,

                              vec3 i_RayOrigin )
    {
        vec3 In = normalize( i_IncidentVector );
        vec3 Ln = normalize( i_DirectionToLight );

        float startDepth = max( i_OceanSurfaceHeight - i_RayOrigin.z,
                                0.01 ) / g_domain;

        float theta = AngleBetweenUnitVectors( -In, -Ln );

        vec3 Hgp = HGPhase( i_PhaseG, theta );
        //Hgp = mix( Hgp, vec3( 1.0, 1.0, 1.0 ), 0.01 );
        float InZ = 0.1 * abs( In.z );
        float LnZ = 0.1 * abs( Ln.z );

        vec3 extincted = vec3(
            FullExtinctionChannel( i_Extinction.x, InZ, LnZ, startDepth ),
            FullExtinctionChannel( i_Extinction.y, InZ, LnZ, startDepth ),
            FullExtinctionChannel( i_Extinction.z, InZ, LnZ, startDepth ) );
        float hackGain = linstep( -2.8 * i_OceanSurfaceHeight,
                                   i_OceanSurfaceHeight,
                                   i_RayOrigin.z );

        return 10.0* 5.25 * hackGain * i_LightColor * Hgp *
               i_Scattering * extincted;
    }

    //-***************************************************************
    uniform float g_ThetaSun;
    uniform float g_PhiSun;
    uniform vec3 g_Zenith;
    uniform vec3 g_A;
    uniform vec3 g_B;
    uniform vec3 g_C;
    uniform vec3 g_D;
    uniform vec3 g_E;
    uniform float g_BigHeight;
    uniform float g_minClipE;
    uniform float g_maxClipE;

    //-***************************************************************
    void VecToThetaPhi( in vec3 V,
                        out float theta,
                        out float phi )
    {
        // Phi is measured clockwise with zero at
        // the negative-x axis. (south)
        // Theta is measured away from positive-z axis.
        phi = atan( -V.y, -V.x );
        float xy = sqrt( sqr( V.x ) + sqr( V.y ) );
        theta = atan( xy, V.z );
    }

    //-***************************************************************
    vec3 ThetaPhiToVec( float Theta, float Phi )
    {
        // Phi is measured clockwise with zero at
        // the negative-x axis. ( south )
        // Theta is measured away from positive-z axis.
        return vec3( sin( Theta ) * -cos( Phi ),
                     sin( Theta ) * -sin( Phi ),
                     cos( Theta ) );
    }

    //-***************************************************************
    float AngleBetween( float Atheta, float Aphi,
                        float Btheta, float Bphi )
    {
        vec3 A = ThetaPhiToVec( Atheta, Aphi );
        vec3 B = ThetaPhiToVec( Btheta, Bphi );
        float cosAng = clamp( dot( A, B ), -1, 1 );
        return acos( cosAng );
    }

    //-***************************************************************
    float PerezFunction( float A,
                         float B,
                         float C,
                         float D,
                         float E,
                         float Theta,
                         float Gamma )
    {
        float cosGamma = cos( Gamma );
        float cosTheta = cos( Theta );
        float d = ( 1.0 +
            A * exp( B / cosTheta ) ) *
            ( 1.0 + C * exp( D * Gamma ) +
              E * sqr( cosGamma ) );
        return d;
    }

    //-***************************************************************
    float Distribution( float A,
                        float B,
                        float C,
                        float D,
                        float E,
                        float Theta,
                        float Gamma )
    {
        float f0 = PerezFunction( A,B,C,D,E,Theta,Gamma );
        float f1 = PerezFunction( A,B,C,D,E,0,g_ThetaSun );
        return (f0/f1);
    }

    //-***************************************************************
    vec3 xyYtoXYZ( float x, float y, float Y )
    {
        return vec3( Y * x / y,
                     Y,
                     Y * ( 1.0 - x - y ) / y );
    }

    //-***************************************************************
    vec3 XYZtoRGB( vec3 xyz )
    {
        vec3 dest;
        dest.r =  3.240479 * xyz.x - 1.537150 * xyz.y  - 0.498535 * xyz.z;
        dest.g = -0.969256 * xyz.x + 1.875991 * xyz.y  + 0.041556 * xyz.z;
        dest.b =  0.055648 * xyz.x - 0.204043 * xyz.y  + 1.057311 * xyz.z;
        return dest;
    }

    //-***************************************************************
    vec3 xyYtoRGB( float x, float y, float Y )
    {
        return XYZtoRGB( xyYtoXYZ( x, y, Y ) );
    }

    //-***************************************************************
    vec3 DaySkyColor( float theta, float phi )
    {
        float gamma = AngleBetween( theta, phi,
                                    g_ThetaSun, g_PhiSun );
        vec3 skyColor;
        skyColor.x = g_Zenith.x *
            Distribution( g_A.x, g_B.x, g_C.x, g_D.x, g_E.x,
                          theta, gamma );
        skyColor.y = g_Zenith.y *
            Distribution( g_A.y, g_B.y, g_C.y, g_D.y, g_E.y,
                          theta, gamma );
        skyColor.z = g_Zenith.z *
            Distribution( g_A.z, g_B.z, g_C.z, g_D.z, g_E.z,
                          theta, gamma );
        // CJH: The 0.045 was tuned by hand, an exposure.
        return
            0.06 * xyYtoRGB( skyColor.x,
                             skyColor.y, skyColor.z );
    }

    //-***************************************************************
    float kSpecular( vec3 In, vec3 Nn, vec3 Ln, float m )
    {
        vec3 Vn = -In;
        vec3 H = normalize( Ln + Vn );
        float d = dot( Nn, H ); d *= d;
        return pow( d, m/2 );
    }

    //----------------------------------------------------------------
    vec3 cheesyFog(vec3 Ci, float z, vec3 density, vec3 Cfog) {
        vec3 f = vec3(exp(-density[0] * z),
                      exp(-density[1] * z),
                      exp(-density[2] * z));
        return vec3((f[0] * Ci[0]) + ((1.0 - f[0])*Cfog[0]),
                    (f[1] * Ci[1]) + ((1.0 - f[1])*Cfog[1]),
                    (f[2] * Ci[2]) + ((1.0 - f[2])*Cfog[2]));
    }

    //--------------------------------------------------------------------------
    vec3 layeredFog(vec3  Ci,
               float z,
               vec3 E,
               vec3 In,
               float beta,
               vec3 density,
               vec3 Cfog)
    {
        float k = (1.0 - exp(-beta * In.z * z)) * exp(-beta * max(0.0, (E.z+0.4*g_BigHeight))) / (beta * In.z);
        vec3 T = vec3(exp(-k * density[0]),
                      exp(-k * density[1]),
                      exp(-k * density[2]) );

        return vec3(mix(Cfog[0], Ci[0], T[0]),
                    mix(Cfog[1], Ci[1], T[1]),
                    mix(Cfog[2], Ci[2], T[2]));
    }

    //-***************************************************************
    void main()
    {
        vec3 I = PREFIXIN_Pwld - g_eyeWld;
        float lenI = length(I);
        vec3 In = normalize( I );
        vec3 Nn = normalize( PREFIXIN_Nwld );
        vec3 R = vec3( 1, 0, 0 );
        vec3 T = vec3( 0, 1, 0 );
        float kr = 1;
        float kt = 1;
        myRefract( In, Nn, 1.0, 1.3, R, T, kr, kt );

        if ( R.z < 0 ) { R.z = -R.z; };
        float Rtheta, Rphi;
        VecToThetaPhi( R, Rtheta, Rphi );

        vec3 ToSun = ThetaPhiToVec( g_ThetaSun, g_PhiSun );
        vec3 ToBack = vec3( -ToSun.x, -ToSun.y, ToSun.z );
        vec3 SunCol = DaySkyColor( g_ThetaSun,
                                   g_PhiSun );
        vec3 BackCol = DaySkyColor( g_ThetaSun,
                                    g_PhiSun + 3.141592 );

        float diffD = dot( ToSun, Nn );
        vec3 Cdiff = 0.95 * SunCol * max( diffD, 0.0 );
        diffD = dot( ToBack, Nn );
        Cdiff += 0.35 * BackCol * max( diffD, 0.0 );

        vec3 sky = DaySkyColor( Rtheta, Rphi );
        const vec3 PacificSigma = vec3( 3.3645, 3.158, 3.2428 );
        const vec3 PacificBeta = vec3( 0.1800, 0.1834, 0.2281 );
        const vec3 PacificG = vec3( 0.902, 0.825, 0.914 );

        vec3 deep2 = FullExtinctionColor( SunCol,
            PacificSigma, PacificBeta, PacificG,
            ToSun, In, g_BigHeight, PREFIXIN_Pwld ) +
                     FullExtinctionColor( BackCol,
            PacificSigma, PacificBeta, PacificG,
            ToBack, In, g_BigHeight, PREFIXIN_Pwld );

        vec3 deep = ( 1.0 - 0.75 * abs( T.z ) ) *
                    ( linstep( -1.8 * g_BigHeight,
                                   g_BigHeight,
                                   PREFIXIN_Pwld.z ) ) *
                    SunCol *
                    PacificBeta *
                    HGPhase( PacificG, 0.1*3.141592 );

        vec3 Cspec = SunCol * kr * 1 *
            kSpecular( In, Nn, ToSun, 800 );

        vec3 finalCol = ( kr * sky )       +
                        ( Cspec )          +
                        ( 10 * kt * deep2 )     +
                        ( 0.075 * kt * deep ) +
                        ( 0.001 * Cdiff );
        if ( g_minClipE < g_maxClipE )
        {
            float clipE = smoothstep( g_minClipE,
                                      g_maxClipE, PREFIXIN_crest );
            finalCol = mix( finalCol, Cdiff, clipE );
        }
        finalCol = layeredFog(finalCol, 1000 *pow(length(I)/1000,1.125),
                              g_eyeWld, In, .1,
                              1.5 * vec3(0.0005, 0.0004, 0.00045),
                              gammaCorrect(vec3(.75, .75, .75), 1.0/2.2));
        //finalCol = Cdiff;
        //float cc = smoothstep( g_minClipE, g_maxClipE, PREFIXIN_crest );
        //finalCol = (0.001*finalCol) + vec3( cc,cc,cc );
        finalCol = gammaCorrect( finalCol, 2.2 );
        g_fragmentColor = vec4( finalCol, 1.0 );
        //g_fragmentColor = vec4( 1.0 );
    }
)";

//-*****************************************************************************
static const char* g_fragmentShaderTextureSkyBase =
  R"(
    in vec3 PREFIXIN_Pwld;
    in vec3 PREFIXIN_Nwld;
    in float PREFIXIN_crest;
    out vec4 g_fragmentColor;

    uniform vec3 g_eyeWld;
    uniform float g_domain;
    uniform sampler2D g_sky_texture;
    uniform vec3 g_to_sun;
    uniform vec3 g_sun_color;
    uniform vec3 g_to_moon;
    uniform vec3 g_moon_color;

    float M_PI = 3.14159265358979323846;
    float M_PI_2 = 1.57079632679489661923;
    float M_PI_4 = 0.785398163397448309616;

    //-***************************************************************
    float sqr( float a ) { return a * a; }

    //-***************************************************************
    float linstep( float edge0, float edge1, float t )
    {
        return clamp( ( t - edge0 ) / ( edge1 - edge0 ),
                      0, 1 );
    }

    //-***************************************************************
    void myRefract( in vec3 In, in vec3 Nn,
                    in float ni, in float nt,
                    out vec3 R, out vec3 T,
                    out float kr, out float kt )
    {
        float eta = ni / nt;
        vec3 Vn = -In;
        float cosThetaI = dot( Vn, Nn );
        vec3 a = Vn - ( cosThetaI * Nn );
        vec3 b = -eta * a;
        float sinThetaI = sqrt(
            clamp( 1.0 - sqr( cosThetaI ), 0.0, 1.0 ) );
        float sinThetaT = eta * sinThetaI;
        float cosThetaT = sqrt(
            clamp( 1.0 - sqr( sinThetaT ), 0.0, 1.0 ) );
        R = normalize( Vn - 2.0 * a );
        T = normalize( b - cosThetaT * Nn );

        float rs = ( (ni*cosThetaI) - (nt*cosThetaT) ) /
           ( (ni*cosThetaI) + (nt*cosThetaT) );

        float rp = ( (ni*cosThetaT) - (nt*cosThetaI) ) /
           ( (ni*cosThetaT) + (nt*cosThetaI) );

        kr = ( sqr( rp ) + sqr( rs ) ) / 2.0;
        kt = ( 1.0 - kr ) / sqr( eta );
    }

    //-***************************************************************
    vec3 gammaCorrect( in vec3 col, in float g )
    {
        return vec3( pow( col.r, 1.0/g ),
                     pow( col.g, 1.0/g ),
                     pow( col.b, 1.0/g ) );
    }

    //-***************************************************************
    // Help out the phase function below.
    float HGPhaseSingle( float PhaseG, float CosTheta )
    {
        return ( 1.0 - PhaseG*PhaseG ) /
            pow( 1.0 + PhaseG*PhaseG - 2.0*PhaseG*CosTheta,
                 3.0/2.0 );
    }

    //-***************************************************************
    // This returns how much the light bounces back at the
    // light source in a participating media.
    vec3 HGPhase( vec3 PhaseG, float Theta )
    {
        float cosTheta = cos( Theta );
        return vec3( HGPhaseSingle( PhaseG.x, cosTheta ),
                     HGPhaseSingle( PhaseG.y, cosTheta ),
                     HGPhaseSingle( PhaseG.z, cosTheta ) )
            / ( 4.0 * 3.141592 );
    }

    //-***************************************************************
    float FullExtinctionChannel( float sigma,
                                 float InZ,
                                 float LnZ,
                                 float startDepth )
    {
        return exp( -startDepth * sigma / LnZ ) * LnZ /
            ( ( InZ + LnZ ) * sigma );
    }

    //-***************************************************************
    float AngleBetweenUnitVectors( vec3 A, vec3 B )
    {
        return acos( clamp( dot( A, B ), -1.0, 1.0 ) );
    }

    //-***************************************************************
    vec3 FullExtinctionColor( vec3 i_LightColor,
                              vec3 i_Extinction,
                              vec3 i_Scattering,
                              vec3 i_PhaseG,

                              vec3 i_DirectionToLight,
                              vec3 i_IncidentVector,

                              float i_OceanSurfaceHeight,

                              vec3 i_RayOrigin )
    {
        vec3 In = normalize( i_IncidentVector );
        vec3 Ln = normalize( i_DirectionToLight );

        float startDepth = max( i_OceanSurfaceHeight - i_RayOrigin.z,
                                0.01 ) / g_domain;

        float theta = AngleBetweenUnitVectors( -In, -Ln );

        vec3 Hgp = HGPhase( i_PhaseG, theta );
        //Hgp = mix( Hgp, vec3( 1.0, 1.0, 1.0 ), 0.01 );
        float InZ = 0.1 * abs( In.z );
        float LnZ = 0.1 * abs( Ln.z );

        vec3 extincted = vec3(
            FullExtinctionChannel( i_Extinction.x, InZ, LnZ, startDepth ),
            FullExtinctionChannel( i_Extinction.y, InZ, LnZ, startDepth ),
            FullExtinctionChannel( i_Extinction.z, InZ, LnZ, startDepth ) );
        float hackGain = linstep( -2.8 * i_OceanSurfaceHeight,
                                   i_OceanSurfaceHeight,
                                   i_RayOrigin.z );

        return 10.0* 5.25 * hackGain * i_LightColor * Hgp *
               i_Scattering * extincted;
    }

    //-***************************************************************
    uniform float g_BigHeight;
    uniform float g_minClipE;
    uniform float g_maxClipE;

    //-***************************************************************
    vec2 VecToEnvTexCoord( in vec3 V )
    {
        float theta = atan(V.y, V.x);
        float xy = sqrt(sqr(V.x) + sqr(V.y));
        float phi = atan(V.z, xy);

        return vec2((theta + M_PI) / (2.0 * M_PI),
                    (phi + M_PI_2) / M_PI);
    }

    //-***************************************************************
    vec3 SkyColor( in vec3 V )
    {
        return texture(g_sky_texture, VecToEnvTexCoord(V)).rgb;
    }

    //-***************************************************************
    float kSpecular( vec3 In, vec3 Nn, vec3 Ln, float m )
    {
        vec3 Vn = -In;
        vec3 H = normalize( Ln + Vn );
        float d = dot( Nn, H ); d *= d;
        return pow( d, m/2 );
    }

    //----------------------------------------------------------------
    vec3 cheesyFog(vec3 Ci, float z, vec3 density, vec3 Cfog) {
        vec3 f = vec3(exp(-density[0] * z),
                      exp(-density[1] * z),
                      exp(-density[2] * z));
        return vec3((f[0] * Ci[0]) + ((1.0 - f[0])*Cfog[0]),
                    (f[1] * Ci[1]) + ((1.0 - f[1])*Cfog[1]),
                    (f[2] * Ci[2]) + ((1.0 - f[2])*Cfog[2]));
    }

    //--------------------------------------------------------------------------
    vec3 layeredFog(vec3  Ci,
               float z,
               vec3 E,
               vec3 In,
               float beta,
               vec3 density,
               vec3 Cfog)
    {
        float k = (1.0 - exp(-beta * In.z * z)) * exp(-beta * max(0.0, (E.z+0.4*g_BigHeight))) / (beta * In.z);
        vec3 T = vec3(exp(-k * density[0]),
                      exp(-k * density[1]),
                      exp(-k * density[2]) );

        return vec3(mix(Cfog[0], Ci[0], T[0]),
                    mix(Cfog[1], Ci[1], T[1]),
                    mix(Cfog[2], Ci[2], T[2]));
    }

    //-***************************************************************
    void main()
    {
        vec3 I = PREFIXIN_Pwld - g_eyeWld;
        float lenI = length(I);
        vec3 In = normalize( I );
        vec3 Nn = normalize( PREFIXIN_Nwld );
        vec3 R = vec3( 1, 0, 0 );
        vec3 T = vec3( 0, 1, 0 );
        float kr = 1;
        float kt = 1;
        myRefract( In, Nn, 1.0, 1.3, R, T, kr, kt );

        if ( R.z < 0 ) { R.z = -R.z; };


        float diffD = dot( g_to_sun, Nn );
        vec3 Cdiff = 0.95 * g_sun_color * max( diffD, 0.0 );
        diffD = dot( g_to_moon, Nn );
        Cdiff += 0.35 * g_moon_color * max( diffD, 0.0 );

        vec3 sky = SkyColor( R );
        const vec3 PacificSigma = vec3( 3.3645, 3.158, 3.2428 );
        const vec3 PacificBeta = vec3( 0.1800, 0.1834, 0.2281 );
        const vec3 PacificG = vec3( 0.902, 0.825, 0.914 );

        vec3 deep2 = FullExtinctionColor( g_sun_color,
            PacificSigma, PacificBeta, PacificG,
            g_to_sun, In, g_BigHeight, PREFIXIN_Pwld ) +

                     FullExtinctionColor( g_moon_color,
            PacificSigma, PacificBeta, PacificG,
            g_to_moon, In, g_BigHeight, PREFIXIN_Pwld );

        vec3 deep = 0.25 * ( 1.0 - 0.75 * abs( T.z ) ) *
                    ( linstep( -1.8 * g_BigHeight,
                                   g_BigHeight,
                                   PREFIXIN_Pwld.z ) ) *
                    g_sun_color *
                    PacificBeta *
                    HGPhase( PacificG, .1*3.141592 );

        vec3 Cspec = g_sun_color * kr * .01 *
            kSpecular( In, Nn, g_to_sun, 800 );

        vec3 finalCol = ( kr * sky )       +
                        ( Cspec )          +
                        ( kt * deep2 )     +
                        ( 0.375 * kt * deep ) +
                        ( 0.001 * Cdiff );
        if ( g_minClipE < g_maxClipE )
        {
            float clipE = smoothstep( g_minClipE,
                                      g_maxClipE, PREFIXIN_crest );
            finalCol = mix( finalCol, Cdiff, clipE );
        }
        finalCol = layeredFog(finalCol, 1000 *pow(length(I)/1000,1.125),
                              g_eyeWld, In, .1,
                              1.5 * vec3(0.0005, 0.0004, 0.00045),
                              gammaCorrect(vec3(.75, .75, .75), 1.0/2.2));
        //finalCol = Cdiff;
        //float cc = smoothstep( g_minClipE, g_maxClipE, PREFIXIN_crest );
        //finalCol = (0.001*finalCol) + vec3( cc,cc,cc );
        finalCol = gammaCorrect( finalCol, 2.2 );
        g_fragmentColor = vec4( finalCol, 1.0 );
        //g_fragmentColor = vec4( 1.0 );
    }
)";

//-*****************************************************************************
static const char* g_envVertexShaderBase =
  R"(
    in vec3 g_vertex;
    uniform mat4 projection_matrix;
    uniform mat4 modelview_matrix;
    out vec3 gv_Pwld;
    void main()
    {
       mat4 pmvMat = projection_matrix * modelview_matrix;
       vec4 vtx = vec4( g_vertex.x, g_vertex.y, g_vertex.z, 1 );
       gl_Position = pmvMat * vtx;
       gv_Pwld = g_vertex;
    }
)";

static const char* g_envFragmentShaderBase =
  R"(
    in vec3 gv_Pwld;
    out vec4 g_fragmentColor;

    uniform sampler2D g_sky_texture;

    float M_PI = 3.14159265358979323846;
    float M_PI_2 = 1.57079632679489661923;
    float M_PI_4 = 0.785398163397448309616;

    float sqr( float a ) { return a * a; }

    vec3 gammaCorrect( in vec3 col, in float g )
    {
        return vec3( pow( col.r, 1.0/g ),
                     pow( col.g, 1.0/g ),
                     pow( col.b, 1.0/g ) );
    }

    vec2 VecToEnvTexCoord( in vec3 V )
    {
        float theta = atan(V.y, V.x);
        float xy = sqrt(sqr(V.x) + sqr(V.y));
        float phi = atan(V.z, xy);

        return vec2((theta + M_PI) / (2.0 * M_PI),
                    (phi + M_PI_2) / M_PI);
    }

    vec3 SkyColor( in vec3 V )
    {
        return texture(g_sky_texture, VecToEnvTexCoord(V)).rgb;
    }

    void main() {
        vec3 R = normalize(vec3(gv_Pwld));
        vec2 tex = VecToEnvTexCoord(R);
        vec3 sky = texture(g_sky_texture, tex).rgb;
        vec3 finalColor = gammaCorrect(sky, 2.2);
        g_fragmentColor = vec4(finalColor, 1.0);
    }
)";

//-*****************************************************************************
std::string VertexShader() {
  return std::string(g_shaderHeader) + std::string(g_vertexShaderBase);
}

//-*****************************************************************************
std::string GeometryShader() {
  return std::string(g_shaderHeader) + std::string(g_simpleGeometryShaderBase);
}

//-*****************************************************************************
void ReplaceAll(std::string& io_str, const std::string& i_search,
                const std::string& i_replace) {
  for (std::size_t pos = 0; true; pos += i_replace.length()) {
    // Locate the substring to replace
    pos = io_str.find(i_search, pos);

    if (pos == std::string::npos) {
      break;
    }

    // Replace by erasing and inserting
    io_str.erase(pos, i_search.length());
    io_str.insert(pos, i_replace);
  }
}

//-*****************************************************************************
std::string FragmentShader(const std::string& i_prefixIn, bool texture_sky) {
  std::string frag =
    texture_sky ? g_fragmentShaderTextureSkyBase : g_fragmentShaderBase;

  ReplaceAll(frag, "PREFIXIN", i_prefixIn);

  return std::string(g_shaderHeader) + frag;
}

//-*****************************************************************************
std::string EnvVertexShader() {
  return std::string(g_shaderHeader) + std::string(g_envVertexShaderBase);
}

//-*****************************************************************************
std::string EnvFragmentShader() {
  return std::string(g_shaderHeader) + std::string(g_envFragmentShaderBase);
}

}  // namespace OceanTest
