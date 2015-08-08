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

#ifndef _EncinoWaves_GeepGLFW_Program_h_
#define _EncinoWaves_GeepGLFW_Program_h_

#include "Foundation.h"
#include "UtilGL.h"

namespace EncinoWaves {
namespace GeepGLFW {

//-*****************************************************************************
class Uniform {
public:
  enum Type { kUniformFloat, kUniformInt, kUniformUint, kUniformMatrix };

  enum Requirement { kRequireOptional, kRequireWarning, kRequireError };

  Uniform()
    : m_name("")
    , m_type(kUniformFloat)
    , m_required(kRequireOptional) {
    m_size[0] = 0;
    m_size[1] = 0;
  }

  Uniform(const Uniform& i_copy)
    : m_name(i_copy.m_name)
    , m_type(i_copy.m_type)
    , m_required(i_copy.m_required) {
    m_size[0] = i_copy.m_size[0];
    m_size[1] = i_copy.m_size[1];
    for (int i = 0; i < 16; ++i) {
      m_data.f[i] = i_copy.m_data.f[i];
    }
  }

  Uniform& operator=(const Uniform& i_copy) {
    m_name         = i_copy.m_name;
    m_type         = i_copy.m_type;
    m_required     = i_copy.m_required;

    m_size[0] = i_copy.m_size[0];
    m_size[1] = i_copy.m_size[1];
    for (int i = 0; i < 16; ++i) {
      m_data.f[i] = i_copy.m_data.f[i];
    }
    return *this;
  }

  const std::string& name() const { return m_name; }

  //-*************************************************************************
  // FLOATS
  //-*************************************************************************

  // 1 float
  Uniform(const std::string& i_name, float i_f,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_f;
    m_size[0]   = 1;
    m_size[1]   = 0;
  }

  // 2 floats
  Uniform(const std::string& i_name, float i_f0, float i_f1,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_f0;
    m_data.f[1] = i_f1;
    m_size[0]   = 2;
    m_size[1]   = 0;
  }

  // V2f
  Uniform(const std::string& i_name, const Imath::Vec2<float>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_v[0];
    m_data.f[1] = i_v[1];
    m_size[0]   = 2;
    m_size[1]   = 0;
  }

  // V2d
  Uniform(const std::string& i_name, const Imath::Vec2<double>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = GLfloat(i_v[0]);
    m_data.f[1] = GLfloat(i_v[1]);
    m_size[0]   = 2;
    m_size[1]   = 0;
  }

  // 3 floats
  Uniform(const std::string& i_name, float i_f0, float i_f1, float i_f2,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_f0;
    m_data.f[1] = i_f1;
    m_data.f[2] = i_f2;
    m_size[0]   = 3;
    m_size[1]   = 0;
  }

  // V3f
  Uniform(const std::string& i_name, const Imath::Vec3<float>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_v[0];
    m_data.f[1] = i_v[1];
    m_data.f[2] = i_v[2];
    m_size[0]   = 3;
    m_size[1]   = 0;
  }

  // V3d
  Uniform(const std::string& i_name, const Imath::Vec3<double>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = GLfloat(i_v[0]);
    m_data.f[1] = GLfloat(i_v[1]);
    m_data.f[2] = GLfloat(i_v[2]);
    m_size[0]   = 3;
    m_size[1]   = 0;
  }

  // 4 floats
  Uniform(const std::string& i_name, float i_f0, float i_f1, float i_f2,
          float i_f3, Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_f0;
    m_data.f[1] = i_f1;
    m_data.f[2] = i_f2;
    m_data.f[3] = i_f3;
    m_size[0]   = 4;
    m_size[1]   = 0;
  }

  // V4f
  Uniform(const std::string& i_name, const Imath::Vec4<float>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = i_v[0];
    m_data.f[1] = i_v[1];
    m_data.f[2] = i_v[2];
    m_data.f[3] = i_v[3];
    m_size[0]   = 4;
    m_size[1]   = 0;
  }

  // V4d
  Uniform(const std::string& i_name, const Imath::Vec4<double>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformFloat)
    , m_required(i_req) {
    m_data.f[0] = GLfloat(i_v[0]);
    m_data.f[1] = GLfloat(i_v[1]);
    m_data.f[2] = GLfloat(i_v[2]);
    m_data.f[3] = GLfloat(i_v[3]);
    m_size[0]   = 4;
    m_size[1]   = 0;
  }

  //-*************************************************************************
  // INTS
  //-*************************************************************************

  // 1 int
  Uniform(const std::string& i_name, int i_i,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_i;
    m_size[0]   = 1;
    m_size[1]   = 0;
  }

  // 2 ints
  Uniform(const std::string& i_name, int i_i0, int i_i1,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_i0;
    m_data.i[1] = i_i1;
    m_size[0]   = 2;
    m_size[1]   = 0;
  }

  // V2i
  Uniform(const std::string& i_name, const Imath::Vec2<int>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_v[0];
    m_data.i[1] = i_v[1];
    m_size[0]   = 2;
    m_size[1]   = 0;
  }

  // 3 ints
  Uniform(const std::string& i_name, int i_i0, int i_i1, int i_i2,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_i0;
    m_data.i[1] = i_i1;
    m_data.i[2] = i_i2;
    m_size[0]   = 3;
    m_size[1]   = 0;
  }

  // V3i
  Uniform(const std::string& i_name, const Imath::Vec3<int>& i_v,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_v[0];
    m_data.i[1] = i_v[1];
    m_data.i[2] = i_v[2];
    m_size[0]   = 3;
    m_size[1]   = 0;
  }

  // 4 ints
  Uniform(const std::string& i_name, int i_i0, int i_i1, int i_i2, int i_i3,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformInt)
    , m_required(i_req) {
    m_data.i[0] = i_i0;
    m_data.i[1] = i_i1;
    m_data.i[2] = i_i2;
    m_data.i[3] = i_i3;
    m_size[0]   = 4;
    m_size[1]   = 0;
  }

  //-*************************************************************************
  // UNSIGNED INTS
  //-*************************************************************************

  // 1 unsigned int
  Uniform(const std::string& i_name, unsigned int i_ui,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformUint)
    , m_required(i_req) {
    m_data.ui[0] = i_ui;
    m_size[0]    = 1;
    m_size[1]    = 0;
  }

  // 2 unsigned ints
  Uniform(const std::string& i_name, unsigned int i_ui0, unsigned int i_ui1,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformUint)
    , m_required(i_req) {
    m_data.ui[0] = i_ui0;
    m_data.ui[1] = i_ui1;
    m_size[0]    = 2;
    m_size[1]    = 0;
  }

  // 3 unsigned ints
  Uniform(const std::string& i_name, unsigned int i_ui0, unsigned int i_ui1,
          unsigned int i_ui2, Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformUint)
    , m_required(i_req) {
    m_data.ui[0] = i_ui0;
    m_data.ui[1] = i_ui1;
    m_data.ui[2] = i_ui2;
    m_size[0]    = 3;
    m_size[1]    = 0;
  }

  // 4 unsigned ints
  Uniform(const std::string& i_name, unsigned int i_ui0, unsigned int i_ui1,
          unsigned int i_ui2, unsigned int i_ui3,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformUint)
    , m_required(i_req) {
    m_data.ui[0] = i_ui0;
    m_data.ui[1] = i_ui1;
    m_data.ui[2] = i_ui2;
    m_data.ui[3] = i_ui3;
    m_size[0]    = 4;
    m_size[1]    = 0;
  }

  //-*************************************************************************
  // MATRICES
  //-*************************************************************************

  // 3x3 matrix
  template <typename T>
  Uniform(const std::string& i_name, const Imath::Matrix33<T>& i_mat,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformMatrix)
    , m_required(i_req) {
    const T* m = &(i_mat[0][0]);
    for (int i = 0; i < 9; ++i) {
      m_data.f[i] = float(m[i]);
    }
    m_size[0] = 3;
    m_size[1] = 3;
  }

  // 4x4 matrix
  template <typename T>
  Uniform(const std::string& i_name, const Imath::Matrix44<T>& i_mat,
          Requirement i_req = kRequireOptional)
    : m_name(i_name)
    , m_type(kUniformMatrix)
    , m_required(i_req) {
    const T* m = &(i_mat[0][0]);
    for (int i = 0; i < 16; ++i) {
      m_data.f[i] = float(m[i]);
    }
    m_size[0] = 4;
    m_size[1] = 4;
  }

  //-*************************************************************************
  // SET
  //-*************************************************************************
  void set(GLuint i_progId) const;

protected:
  std::string m_name;
  union {
    GLfloat f[16];
    GLint i[16];
    GLuint ui[16];
  } m_data;
  Type m_type;
  int m_size[2];
  Requirement m_required;
};

//-*****************************************************************************
class Program {
public:
  typedef std::pair<GLuint, std::string> Binding;
  typedef std::vector<Binding> Bindings;

  // The reason the Program constructor takes a vertex array object
  // specifier is because some OpenGL drivers (OSX) require a VAO to be
  // bound in order to validate a program, and without one bound,
  // glValidateProgram will return an error. This constructor will bind
  // and unbind the VAO if it is greater than zero.
  Program(const std::string& i_name, const std::string& i_vtx,
          const std::string& i_geom, const std::string& i_frg,
          const Bindings& i_vtxBindingsIn, const Bindings& i_frgBindingsOut,
          GLuint i_vertexArrayObject);
  ~Program();

  void operator()(const Uniform& i_uniform);

  void use() const;
  void unuse() const;

  void setUniforms() const;

  GLuint id() const { return m_progId; }
  GLuint vertexShaderId() const { return m_vtxId; }
  GLuint geometryShaderId() const { return m_geomId; }
  GLuint fragmentShaderId() const { return m_frgId; }

protected:
  GLuint initShader(const std::string& i_shaderName, GLenum i_type,
                    const std::vector<std::string>& i_sources);

  std::string m_name;

  GLuint m_progId;
  GLuint m_vtxId;
  GLuint m_geomId;
  GLuint m_frgId;

  typedef std::map<std::string, Uniform> Uniforms;
  Uniforms m_uniforms;
};

}  // namespace GeepGLFW
}  // namespace EncinoWaves

#endif
