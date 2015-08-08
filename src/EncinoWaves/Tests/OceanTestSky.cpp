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

#include "OceanTestSky.h"

namespace OceanTest {

using namespace GeepGLFW;

//-*****************************************************************************
//-*****************************************************************************
// This rendering code is an implementation of the paper,
// "A Practical Analytic Model for Daylight"
// Preetham, Shirley, Smits, 1999
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
static void DistributionCoefficients(double T, V3d &A, V3d &B, V3d &C, V3d &D,
                                     V3d &E) {
  A = V3d(-0.0193 * T - 0.2592, -0.0167 * T - 0.2608, 0.1787 * T - 1.4630);

  B = V3d(-0.0665 * T - 0.0008, -0.0950 * T + 0.0092, -0.3554 * T + 0.4275);

  C = V3d(-0.0004 * T + 0.2125, -0.0079 * T + 0.2102, -0.0227 * T + 5.3251);

  D = V3d(-0.0641 * T - 0.8989, -0.0441 * T - 1.6537, 0.1206 * T - 2.5771);

  E = V3d(-0.0033 * T + 0.0452, -0.0109 * T + 0.0529, -0.0670 * T + 0.3703);
}

//-*****************************************************************************
static V3d Zenith_xyY(double T, double thetaS) {
  V3d xyY;

  double chi = ((4.0 / 9.0) - (T / 120.0)) * (M_PI - 2.0 * thetaS);

  xyY[2] = ((4.0453 * T) - 4.9710) * tan(chi) - (0.2155 * T) + 2.4192;

  double T2 = T * T;
  V3d T2T1(T2, T, 1.0);
  double S1 = thetaS;
  double S2 = S1 * S1;
  double S3 = S1 * S1 * S1;

  V3d zenx((0.00166 * S3) + (-0.00375 * S2) + (0.00209 * S1) + (0.0),

           (-0.02903 * S3) + (0.06377 * S2) + (-0.03202 * S1) + (0.00394),

           (0.11693 * S3) + (-0.21196 * S2) + (0.06052 * S1) + (0.25886));

  xyY[0] = T2T1.dot(zenx);

  V3d zeny((0.00275 * S3) + (-0.00610 * S2) + (0.00317 * S1) + (0.0),

           (-0.04214 * S3) + (0.08970 * S2) + (-0.04153 * S1) + (0.00516),

           (0.15346 * S3) + (-0.26756 * S2) + (0.06670 * S1) + (0.26688));

  xyY[1] = T2T1.dot(zeny);

  return xyY;
}

//-*****************************************************************************
// Computing sun phi & theta from
static void CalcSunPosition(double pTime, double pDay, double pLat,
                            double pLong, double GMT_offset,

                            double &o_phi, double &o_theta) {
  double latRad  = radians(pLat);
  double longRad = radians(pLong);

  double solarTime = pTime + +0.170 * sin(4.0 * M_PI * (pDay - 80.0) / 373.0) -
                     0.129 * sin(2.0 * M_PI * (pDay - 8.0) / 355.0) +
                     12.0 * (((GMT_offset / 12.0) * M_PI) - longRad) / M_PI;

  double solarDeclination = 0.4093 * sin(2.0 * M_PI * (pDay - 81.0) / 368.0);

  double sinL = std::sin(latRad);
  double cosL = std::cos(latRad);

  double sinD = std::sin(solarDeclination);
  double cosD = std::cos(solarDeclination);

  double solarTimeRad = M_PI * solarTime / 12.0;
  double sinSTR       = std::sin(solarTimeRad);
  double cosSTR       = std::cos(solarTimeRad);

  o_theta = M_PI_2 - std::asin((sinL * sinD) - (cosL * cosD * cosSTR));
  o_phi   = std::atan2(-(cosD * sinSTR), (cosL * sinD) - (sinL * cosD * cosSTR));
}

//-*****************************************************************************
void Sky::setUniforms(Program &i_program) const {
  double d_phi;
  double d_theta;
  CalcSunPosition(m_params.time, m_params.day, m_params.latitude,
                  m_params.longitude, m_params.GMT_offset, d_phi, d_theta);
  // std::cout << "Sun phi & theta: " << d_phi << ", " << d_theta << std::endl;

  double TURB = 1.7 + std::max(m_params.turbidity, 0.0);

  V3d zenith = Zenith_xyY(TURB, d_theta);

  V3d d_A, d_B, d_C, d_D, d_E;
  DistributionCoefficients(TURB, d_A, d_B, d_C, d_D, d_E);

  i_program(Uniform("g_ThetaSun", (float)d_theta));
  i_program(Uniform("g_PhiSun", (float)d_phi));
  i_program(Uniform("g_Zenith", zenith));
  i_program(Uniform("g_A", d_A));
  i_program(Uniform("g_B", d_B));
  i_program(Uniform("g_C", d_C));
  i_program(Uniform("g_D", d_D));
  i_program(Uniform("g_E", d_E));
}

}  // namespace OceanTest
