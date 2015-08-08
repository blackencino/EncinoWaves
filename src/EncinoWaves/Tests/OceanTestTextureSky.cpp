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

#include "OceanTestTextureSky.h"
#include <tuple>
#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>

namespace OceanTest {

using namespace Imf;
using namespace Imath;

static void readRgba1(const std::string& filename, Array2D<Rgba>& pixels,
                      int& width, int& height) {
  RgbaInputFile file{filename.c_str()};
  Box2i dw = file.dataWindow();
  width    = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;
  pixels.resizeErase(height, width);
  file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
  file.readPixels(dw.min.y, dw.max.y);
}

TextureSky::TextureSky(const std::string& filename) {
  int width  = 0;
  int height = 0;
  Array2D<Rgba> pixels;
  readRgba1(filename, pixels, width, height);
  std::cout << "Loaded texture sky: " << filename << std::endl;

  float gain     = 2.0f;
  float exponent = 1.0f;
  auto brighten  = [gain, exponent](Rgba& rgba) {
    rgba.r       = std::pow(float(rgba.r), exponent) * gain;
    rgba.g       = std::pow(float(rgba.g), exponent) * gain;
    rgba.b       = std::pow(float(rgba.b), exponent) * gain;
  };

  for (int y = 0, y2 = height - 1; y < y2; ++y, --y2) {
    for (int x = 0; x < width; ++x) {
      std::swap(pixels[y][x], pixels[y2][x]);
      brighten(pixels[y][x]);
      brighten(pixels[y2][x]);
    }
  }

  auto luminance = [](const Rgba& rgba) -> float {
    return (0.2126f * rgba.r) + (0.7152f * rgba.g) + (0.0722f * rgba.b);
  };

  // find brightest pixel in image
  int brightest_x           = -1;
  int brightest_y           = -1;
  float brightest_luminance = std::numeric_limits<float>::lowest();
  auto brightest = std::tie(brightest_x, brightest_y, brightest_luminance);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float lum = luminance(pixels[y][x]);
      if (lum > std::get<2>(brightest)) {
        std::get<0>(brightest) = x;
        std::get<1>(brightest) = y;
        std::get<2>(brightest) = lum;
      }
    }
  }
  std::cout << "Brightest x, y: " << brightest_x << ", " << brightest_y
            << std::endl;
  float brightest_theta =
    lerp(-M_PI, M_PI, (0.5f + static_cast<double>(brightest_x)) /
                        static_cast<double>(width));
  float brightest_phi =
    lerp(-M_PI_2, M_PI_2, (0.5f + static_cast<double>(brightest_y)) /
                            static_cast<double>(height));
  float cos_theta = std::cos(brightest_theta);
  float sin_theta = std::sin(brightest_theta);
  float cos_phi   = std::cos(brightest_phi);
  float sin_phi   = std::sin(brightest_phi);
  m_to_sun        = V3f{cos_phi * cos_theta, cos_phi * sin_theta, sin_phi};
  auto sun_rgba = pixels[brightest_y][brightest_x];
  m_sun_color =
    V3f{static_cast<float>(sun_rgba.r), static_cast<float>(sun_rgba.g),
        static_cast<float>(sun_rgba.b)};
  m_to_moon      = V3f{-m_to_sun.x, -m_to_sun.y, m_to_sun.z};
  auto moon_rgba = pixels[brightest_y][wrap(brightest_x + (width / 2), width)];
  m_moon_color =
    V3f{static_cast<float>(moon_rgba.r), static_cast<float>(moon_rgba.g),
        static_cast<float>(moon_rgba.b)};

  // create texture
  m_tex_id = 0;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &m_tex_id);

  // load into texture
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_tex_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(GL_TEXTURE_2D,   // target
               0,               // level
               GL_RGBA16F_ARB,  // internalformat
               width,           // width
               height,          // height
               0,               // border
               GL_RGBA,         // format
               GL_HALF_FLOAT_ARB, reinterpret_cast<const GLvoid*>(pixels[0]));
  std::cout << "Created GL Texture from Sky Pixels" << std::endl;
}

TextureSky::~TextureSky() {
  if (m_tex_id > 0) {
    glDeleteTextures(1, &m_tex_id);
  }
}

void TextureSky::bind(GLuint program_id) const {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_tex_id);
  glUniform1i(glGetUniformLocation(program_id, "g_sky_texture"), 0);
  glUniform3f(glGetUniformLocation(program_id, "g_to_sun"), m_to_sun.x,
              m_to_sun.y, m_to_sun.z);
  glUniform3f(glGetUniformLocation(program_id, "g_sun_color"), m_sun_color.x,
              m_sun_color.y, m_sun_color.z);
  glUniform3f(glGetUniformLocation(program_id, "g_to_moon"), m_to_moon.x,
              m_to_moon.y, m_to_moon.z);
  glUniform3f(glGetUniformLocation(program_id, "g_moon_color"), m_moon_color.x,
              m_moon_color.y, m_moon_color.z);
}

}  // namespace OceanTest