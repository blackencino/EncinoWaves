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

#ifndef _EncinoWaves_OceanTest_ViewScene_h_
#define _EncinoWaves_OceanTest_ViewScene_h_

#include "OceanTestFoundation.h"
#include "OceanTestMesh.h"
#include "OceanTestSky.h"

namespace OceanTest {

//-*****************************************************************************
#define EWAV_USE_VERTEX_BUFFERS 0

//-*****************************************************************************
class ParamEditor {
public:
  ParamEditor(const std::string& i_name)
    : m_name(i_name)
    , m_enabled(true) {}
  virtual ~ParamEditor() {}

  enum ActionType { kDownSmall, kDownBig, kUpSmall, kUpBig, kReset };

  virtual std::string description() const = 0;
  virtual void action(ActionType i_type) = 0;

  const std::string& name() const { return m_name; }

  bool enabled() const { return m_enabled; }
  void enable(bool tf) { m_enabled = tf; }

protected:
  std::string m_name;
  bool m_enabled;
};

//-*****************************************************************************
template <typename T>
class NumParamEditor : public ParamEditor {
public:
  NumParamEditor(const std::string& i_name, T* o_value, T i_min, T i_max,
                 T i_smallIncrement, T i_bigIncrement)
    : ParamEditor(i_name)
    , m_value(o_value)
    , m_min(i_min)
    , m_max(i_max)
    , m_default((*o_value))
    , m_smallIncrement(i_smallIncrement)
    , m_bigIncrement(i_bigIncrement) {}

  virtual std::string description() const {
    std::stringstream sstr;
    sstr << m_name << " = " << *m_value;
    return sstr.str();
  }

  virtual void action(ActionType i_type) {
    T i_val = *m_value;

    switch (i_type) {
    case kDownSmall:
      i_val -= m_smallIncrement;
      break;
    case kDownBig:
      i_val -= m_bigIncrement;
      break;
    case kUpSmall:
      i_val += m_smallIncrement;
      break;
    case kUpBig:
      i_val += m_bigIncrement;
      break;
    case kReset:
      i_val = m_default;
      break;
    };

    i_val = Imath::clamp(i_val, m_min, m_max);

    if (i_val != *m_value) {
      *m_value = i_val;
      std::cout << description() << std::endl;
    }
  }

public:
  T* m_value;

  T m_min;
  T m_max;
  T m_default;
  T m_smallIncrement;
  T m_bigIncrement;
};

//-*****************************************************************************
typedef NumParamEditor<int> IntParamEditor;
typedef NumParamEditor<float> FloatParamEditor;
typedef NumParamEditor<double> DoubleParamEditor;

//-*****************************************************************************

//-*****************************************************************************
class BoolParamEditor : public ParamEditor {
public:
  BoolParamEditor(const std::string& i_name, bool* o_value)
    : ParamEditor(i_name)
    , m_value(o_value)
    , m_default((*o_value)) {}

  virtual std::string description() const {
    std::stringstream sstr;
    sstr << m_name << " = " << *m_value;
    return sstr.str();
  }

  virtual void action(ActionType i_type) {
    bool i_val = *m_value;

    switch (i_type) {
    case kDownSmall:
    case kDownBig:
    case kUpSmall:
    case kUpBig:
      i_val = !i_val;
      break;
    case kReset:
      i_val = m_default;
      break;
    };

    if (i_val != *m_value) {
      *m_value = i_val;
      std::cout << description() << std::endl;
    }
  }

public:
  bool* m_value;

  bool m_default;
};

//-*****************************************************************************
template <typename ENUM_T>
class EnumParamEditor : public ParamEditor {
public:
  typedef std::pair<std::string, ENUM_T> NamedEnum;
  typedef std::vector<NamedEnum> NamedEnumVector;

  EnumParamEditor(const std::string& i_name, ENUM_T* o_value,
                  const NamedEnumVector& i_namedEnums)
    : ParamEditor(i_name)
    , m_value(o_value)
    , m_namedEnums(i_namedEnums) {
    // Need to find the position of our default.
    m_defaultPosition = -1;
    for (int i = 0; i < m_namedEnums.size(); ++i) {
      if (m_namedEnums[i].second == *m_value) {
        m_defaultPosition = i;
        break;
      }
    }
    EWAV_ASSERT(m_defaultPosition >= 0, "Bad enum param editor.");
    m_position = m_defaultPosition;
  }

  virtual std::string description() const {
    const NamedEnum& ne = m_namedEnums[m_position];
    std::stringstream sstr;
    sstr << m_name << " = " << ne.first;
    return sstr.str();
  }

  virtual void action(ActionType i_type) {
    switch (i_type) {
    case kDownSmall:
    case kDownBig:
      --m_position;
      break;
    case kUpSmall:
    case kUpBig:
      --m_position;
      break;
    case kReset:
      m_position = m_defaultPosition;
      break;
    };

    m_position = wrap(m_position, (int)m_namedEnums.size());

    const NamedEnum& ne = m_namedEnums[m_position];

    if (ne.second != *m_value) {
      *m_value = ne.second;
      std::cout << description() << std::endl;
    }
  }

protected:
  ENUM_T* m_value;

  NamedEnumVector m_namedEnums;
  int m_position;
  int m_defaultPosition;
};

//-*****************************************************************************
typedef std::shared_ptr<ParamEditor> ParamEditorSptr;
typedef std::vector<ParamEditorSptr> ParamEditorSptrVector;

//-*****************************************************************************
class ViewScene : public EncinoWaves::SimpleSimViewer::Sim3D {
public:
  struct Parameters {
    std::string outputFileBase = "EncinoWaves";
  };
  ViewScene(const ewav::Parametersf& i_params,
            const Sky::Parameters& i_skyParams,
            const Mesh::DrawParameters& i_dparams, const Parameters& i_vparams);

  virtual std::string getName() const override;

  virtual void step() override {
    if (m_mesh) {
      m_mesh->step();
    }
    if (m_writing && m_rgbGrabBufferWidth > 0 && m_rgbGrabBufferHeight > 0 &&
        m_rgbGrabBuffer.size() > 0) {
      writeFrame();
      ++m_writeFrame;
    }
  }

  virtual Box3d getBounds() const override {
    Box3d ret;
    ret.makeEmpty();
    if (m_mesh) {
      ret = m_mesh->getBounds();
    } else {
      ret = Box3d(V3d(-0.125 * m_params.domain, -0.125 * m_params.domain,
                      -0.125 * m_params.domain),
                  V3d(0.125 * m_params.domain));
    }
    return ret;
  }

  virtual bool overrideClipping(double& o_near, double& o_far) const override {
    // m_camera.autoSetClippingPlanes(getBounds());
    V2d clip = m_camera.clippingPlanes();
    o_near   = std::min(0.1, clip[0]);
    o_far    = std::max(3.0 * m_params.domain, clip[1]);
    o_far    = std::max(o_far, 5000.0);
    return true;
  }

  virtual void draw() override {
    glClearColor(.75, .75, .75, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!m_mesh) {
      m_mesh.reset(new Mesh(m_params, m_skyParams, m_drawParams));
    }
    m_mesh->draw(m_camera);
    if (m_writing) {
      grabFrame();
    }
  }

  virtual void character(unsigned int key, int x, int y) override {
    if (key == 'w') {
      toggleWriting();
      // std::cout << "W key pressed" << std::endl
      //    << "Camera w, h: " << camera().width() << ", "
      //    << camera().height() << std::endl
      //    << "Out file base: " << m_viewSceneParams.outputFileBase
      //    << std::endl;
    }
  }

  virtual void keyboard(int key, int i_scancode, int i_action, int i_mods,
                        int x, int y) override;

protected:
  void toggleWriting() {
    m_writeFrame = 1;
    m_writing    = !m_writing;
  }
  void grabFrame();
  void writeFrame();
  ParamEditorSptr findParamEdit(const std::string& i_name);
  void enableParamEdits();

  ewav::Parametersf m_params;
  Sky::Parameters m_skyParams;
  Mesh::DrawParameters m_drawParams;
  Parameters m_viewSceneParams;
  std::unique_ptr<Mesh> m_mesh;

  ParamEditorSptrVector m_paramEdits;
  int m_paramEditPosition;

  bool m_writing   = false;
  int m_writeFrame = 1;
  std::vector<unsigned char> m_rgbGrabBuffer;
  int m_rgbGrabBufferWidth  = 0;
  int m_rgbGrabBufferHeight = 0;
  std::vector<Imf::Rgba> m_rgbaWriteBuffer;
};

}  // namespace OceanTest

#endif
