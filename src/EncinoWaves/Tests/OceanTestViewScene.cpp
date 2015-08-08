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

#include "OceanTestViewScene.h"

namespace OceanTest {

//-*****************************************************************************
ViewScene::ViewScene(const ewav::Parametersf& i_params,
                     const Sky::Parameters& i_skyParams,
                     const Mesh::DrawParameters& i_dparams,
                     const Parameters& vparams)
  : EncinoWaves::SimpleSimViewer::Sim3D()
  , m_params(i_params)
  , m_skyParams(i_skyParams)
  , m_drawParams(i_dparams)
  , m_viewSceneParams(vparams) {
  // Make param editors.
  ParamEditorSptr pedit;

  // Domain
  pedit.reset(new FloatParamEditor("domain", &m_params.domain, 1.0f, 10000.0f,
                                   1.0f, 50.0f));
  m_paramEdits.push_back(pedit);

  // Domain
  pedit.reset(new FloatParamEditor("depth", &m_params.depth, 0.25f, 1000.0f,
                                   0.5f, 10.0f));
  m_paramEdits.push_back(pedit);

  // Wind Speed
  pedit.reset(new FloatParamEditor("wind speed", &m_params.windSpeed, 1.0f,
                                   500.0f, 1.0f, 10.0f));
  m_paramEdits.push_back(pedit);

  // Fetch
  pedit.reset(new FloatParamEditor("fetch", &m_params.fetch, 1.0f, 5000.0f,
                                   10.0f, 50.0f));
  m_paramEdits.push_back(pedit);

  // Pinch
  pedit.reset(
    new FloatParamEditor("pinch", &m_params.pinch, -3.0f, 3.0f, 0.05f, 0.25f));
  m_paramEdits.push_back(pedit);

#if 0
  // Amplitude
  pedit.reset(new FloatParamEditor("amplitude", &m_params.amplitudeGain, 0.01f,
                                   10.0f, 0.05f, 0.25f));
  m_paramEdits.push_back(pedit);

  // Random
  {
    typedef EnumParamEditor<ewav::RandomType> RandPE;
    RandPE::NamedEnumVector nev;
    nev.push_back(RandPE::NamedEnum("Normal", ewav::kNormalRandom));
    nev.push_back(
      RandPE::NamedEnum("Log Normal Random", ewav::kLogNormalRandom));
    pedit.reset(new RandPE("random distribution", &m_params.random.type, nev));
    m_paramEdits.push_back(pedit);
  }

  // Dispersion
  {
    typedef EnumParamEditor<ewav::DispersionType> DispPE;
    DispPE::NamedEnumVector nev;
    nev.push_back(DispPE::NamedEnum("Deep", ewav::kDeepDispersion));
    nev.push_back(
      DispPE::NamedEnum("Finite Depth", ewav::kFiniteDepthDispersion));
    nev.push_back(DispPE::NamedEnum("Capillary", ewav::kCapillaryDispersion));
    pedit.reset(new DispPE("dispersion", &m_params.dispersion.type, nev));
    m_paramEdits.push_back(pedit);
  }
#endif

  // Spectrum
  {
    typedef EnumParamEditor<ewav::SpectrumType> SpecPE;
    SpecPE::NamedEnumVector nev;
    nev.push_back(
      SpecPE::NamedEnum("Pierson-Moskowitz", ewav::kPiersonMoskowitzSpectrum));
    nev.push_back(SpecPE::NamedEnum("JONSWAP", ewav::kJONSWAPSpectrum));
    nev.push_back(SpecPE::NamedEnum("Texel-Marsen-Arsloe", ewav::kTMASpectrum));
    pedit.reset(new SpecPE("spectrum", &m_params.spectrum.type, nev));
    m_paramEdits.push_back(pedit);
  }

  // Directional Spreading
  {
    typedef EnumParamEditor<ewav::DirectionalSpreadingType> DirPE;
    DirPE::NamedEnumVector nev;
    nev.push_back(DirPE::NamedEnum("Pos Cos-Theta Squared",
                                   ewav::kPosCosThetaSqrDirectionalSpreading));
    nev.push_back(
      DirPE::NamedEnum("Mitsuyasu", ewav::kMitsuyasuDirectionalSpreading));
    nev.push_back(
      DirPE::NamedEnum("Hasselmann", ewav::kHasselmannDirectionalSpreading));
    nev.push_back(DirPE::NamedEnum("Donelan Banner",
                                   ewav::kDonelanBannerDirectionalSpreading));
    pedit.reset(new DirPE("directional spreading",
                          &m_params.directionalSpreading.type, nev));
    m_paramEdits.push_back(pedit);
  }

  pedit.reset(new FloatParamEditor("spread swell",
                                   &m_params.directionalSpreading.swell, -1.0f,
                                   2.0f, 0.01f, 0.05f));
  m_paramEdits.push_back(pedit);

#if 0
  // Filter
  {
    typedef EnumParamEditor<ewav::FilterType> FiltPE;
    FiltPE::NamedEnumVector nev;
    nev.push_back(FiltPE::NamedEnum("Null", ewav::kNullFilter));
    nev.push_back(FiltPE::NamedEnum("Smooth Invertible Band Pass",
                                    ewav::kSmoothInvertibleBandPassFilter));
    pedit.reset(new FiltPE("filter", &m_params.filter.type, nev));
    m_paramEdits.push_back(pedit);
  }

  // Filter stuff
  pedit.reset(new FloatParamEditor("filter soft width",
                                   &m_params.filter.softWidth, 0.1f, 1000.0f,
                                   0.25f, 10.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("filter small wavelength",
                                   &m_params.filter.smallWavelength, 0.0f,
                                   1000.0f, 0.25f, 10.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("filter big wavelength",
                                   &m_params.filter.bigWavelength, 0.0f,
                                   1000.0f, 0.25f, 10.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("filter min", &m_params.filter.min, 0.0f,
                                   1.0f, 0.01f, 0.05f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new BoolParamEditor("filter invert", &m_params.filter.invert));
  m_paramEdits.push_back(pedit);
#endif

  // Trough stuff
  pedit.reset(new FloatParamEditor("trough damping", &m_params.troughDamping,
                                   0.0f, 1.0f, 0.05f, 0.1f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("trough damping small wavelength",
                                   &m_params.troughDampingSmallWavelength, 0.0f,
                                   1000.0f, 0.25f, 2.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("trough damping big wavelength",
                                   &m_params.troughDampingBigWavelength, 0.0f,
                                   1000.0f, 0.25f, 2.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("trough damping soft width",
                                   &m_params.troughDampingSoftWidth, 0.1f,
                                   1000.0f, 0.25f, 2.0f));
  m_paramEdits.push_back(pedit);

  pedit.reset(new FloatParamEditor("wind rotation",
                                   &m_drawParams.wind_rotation, -360.0f, 360.0f,
                                   1.0f, 10.0f));
  m_paramEdits.push_back(pedit);

  // Set edit to wind speed.
  m_paramEditPosition = 2;

  // Enable/Disable
  enableParamEdits();
}

//-*****************************************************************************
std::string ViewScene::getName() const {
  std::string ret = std::string("EncinoWaves 2013 : ") +
                    m_paramEdits[m_paramEditPosition]->description();

  if (!(m_paramEdits[m_paramEditPosition]->enabled())) {
    ret += " DISABLED";
  }

  return ret;
}

//-*****************************************************************************
void ViewScene::keyboard(int i_key, int i_scancode, int i_action, int i_mods,
                         int i_x, int i_y) {
  // Don't bother with releases.
  if (i_action == GLFW_RELEASE) {
    return;
  }

  // Shift
  bool shift = (bool)(i_mods & GLFW_MOD_SHIFT);

  // Control
  bool ctrl = (bool)(i_mods & GLFW_MOD_CONTROL);

  // Alt
  bool alt = (bool)(i_mods & GLFW_MOD_ALT);

  // Super
  bool super = (bool)(i_mods & GLFW_MOD_SUPER);

  bool doEditMesh = false;
  switch (i_key) {
  // Right arrow
  case 262:
    if (!m_paramEdits[m_paramEditPosition]->enabled()) {
      break;
    }
    if (shift) {
      m_paramEdits[m_paramEditPosition]->action(ParamEditor::kUpBig);
    } else {
      m_paramEdits[m_paramEditPosition]->action(ParamEditor::kUpSmall);
    }
    doEditMesh = true;
    break;

  // Left arrow
  case 263:
    if (!m_paramEdits[m_paramEditPosition]->enabled()) {
      break;
    }
    if (shift) {
      m_paramEdits[m_paramEditPosition]->action(ParamEditor::kDownBig);
    } else {
      m_paramEdits[m_paramEditPosition]->action(ParamEditor::kDownSmall);
    }
    doEditMesh = true;
    break;

  // TAB
  case 258:
    if (shift) {
      --m_paramEditPosition;
    } else {
      ++m_paramEditPosition;
    }
    m_paramEditPosition = wrap(m_paramEditPosition, (int)m_paramEdits.size());
    break;
  };

  if (doEditMesh && m_mesh) {
    enableParamEdits();
    m_mesh->setWavesParams(m_params);
    m_mesh->setDrawParams(m_drawParams);
  }

  SimpleSimViewer::Sim3D::keyboard(i_key, i_scancode, i_action, i_mods, i_x,
                                   i_y);
}

//-*****************************************************************************
ParamEditorSptr ViewScene::findParamEdit(const std::string& i_name) {
  ParamEditorSptr nullRet;

  for (ParamEditorSptrVector::iterator iter = m_paramEdits.begin();
       iter != m_paramEdits.end(); ++iter) {
    if ((*iter)->name() == i_name) {
      return (*iter);
    }
  }

  return nullRet;
}

//-*****************************************************************************
void ViewScene::enableParamEdits() {
  for (ParamEditorSptrVector::iterator iter = m_paramEdits.begin();
       iter != m_paramEdits.end(); ++iter) {
    (*iter)->enable(true);
  }

  switch (m_params.spectrum.type) {
  case ewav::kPiersonMoskowitzSpectrum:
    findParamEdit("fetch")->enable(false);
    break;
  default:
    break;
  };

#if 0
  switch (m_params.filter.type) {
  case ewav::kNullFilter:
    findParamEdit("filter soft width")->enable(false);
    findParamEdit("filter small wavelength")->enable(false);
    findParamEdit("filter big wavelength")->enable(false);
    findParamEdit("filter min")->enable(false);
    findParamEdit("filter invert")->enable(false);
    break;
  default:
    break;
  };
#endif
}

//------------------------------------------------------------------------------
void ViewScene::grabFrame() {
  // CJH HACK the 2 comes from multisample
  GLsizei width         = 2 * camera().width();
  m_rgbGrabBufferWidth  = width;
  GLsizei height        = 2 * camera().height();
  m_rgbGrabBufferHeight = height;
  m_rgbGrabBuffer.resize(width * height * 3);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE,
               (GLvoid*)m_rgbGrabBuffer.data());
}

//------------------------------------------------------------------------------
void ViewScene::writeFrame() {
  std::stringstream sstr;
  sstr << m_viewSceneParams.outputFileBase << "." << m_writeFrame << ".exr";
  std::string outputFileName = sstr.str();
  const unsigned char* c = m_rgbGrabBuffer.data();
  m_rgbaWriteBuffer.resize(m_rgbGrabBufferWidth * m_rgbGrabBufferHeight);
  Imf::Rgba* p = m_rgbaWriteBuffer.data();
  for (int y = 0; y < m_rgbGrabBufferHeight; ++y) {
    for (int x = 0; x < m_rgbGrabBufferWidth; ++x, ++p, c += 3) {
      p->r = float(c[0]) / 255.0f;
      p->g = float(c[1]) / 255.0f;
      p->b = float(c[2]) / 255.0f;
      p->a = 1.0f;
    }
  }
  Imf::RgbaOutputFile file(outputFileName.c_str(), m_rgbGrabBufferWidth,
                           m_rgbGrabBufferHeight, Imf::WRITE_RGBA);
  file.setFrameBuffer(m_rgbaWriteBuffer.data(), 1, m_rgbGrabBufferWidth);
  file.writePixels(m_rgbGrabBufferHeight);
  std::cout << "Wrote: " << outputFileName << std::endl;
}

}  // namespace OceanTest
