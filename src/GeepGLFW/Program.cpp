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

#include "Program.h"

namespace EncinoWaves {
namespace GeepGLFW {

//-*****************************************************************************
//-*****************************************************************************
// UNIFORM
//-*****************************************************************************
//-*****************************************************************************
void Uniform::set(GLuint i_progId) const {
  // Get out of nonexistent uniforms.
  if (m_size[0] < 1 || m_name == "") {
    return;
  }

  // Find the uniform location, and complain if necessary.
  GLint loc = glGetUniformLocation(i_progId, m_name.c_str());
  if (loc < 0) {
    switch (m_required) {
    default:
    case kRequireOptional:
      break;
    case kRequireWarning:
      std::cout << "WARNING: Couldn't find uniform: " << m_name
                << " in program." << std::endl;
      break;
    case kRequireError:
      EWAV_THROW("Couldn't find uniform: " << m_name << " in program.");
      break;
    }

    // No uniform location, bug out.
    return;
  }

  // Set the uniform by type.
  switch (m_type) {
  case kUniformFloat:
    switch (m_size[0]) {
    default:
    case 1:
      glUniform1f(loc, (GLfloat)m_data.f[0]);
      break;
    case 2:
      glUniform2fv(loc, 1, (const GLfloat*)m_data.f);
      break;
    case 3:
      glUniform3fv(loc, 1, (const GLfloat*)m_data.f);
      break;
    case 4:
      glUniform4fv(loc, 1, (const GLfloat*)m_data.f);
      break;
    }
    break;
  case kUniformInt:
    switch (m_size[0]) {
    default:
    case 1:
      glUniform1i(loc, (GLint)m_data.i[0]);
      break;
    case 2:
      glUniform2iv(loc, 1, (const GLint*)m_data.i);
      break;
    case 3:
      glUniform3iv(loc, 1, (const GLint*)m_data.i);
      break;
    case 4:
      glUniform4iv(loc, 1, (const GLint*)m_data.i);
      break;
    }
    break;
  case kUniformUint:
    switch (m_size[0]) {
    default:
    case 1:
      glUniform1ui(loc, (GLuint)m_data.ui[0]);
      break;
    case 2:
      glUniform2uiv(loc, 1, (const GLuint*)m_data.ui);
      break;
    case 3:
      glUniform3uiv(loc, 1, (const GLuint*)m_data.ui);
      break;
    case 4:
      glUniform4uiv(loc, 1, (const GLuint*)m_data.ui);
      break;
    }
    break;
  case kUniformMatrix:
    switch (m_size[0]) {
    default:
    case 3:
      glUniformMatrix3fv(loc, 1, GL_FALSE, (const GLfloat*)m_data.f);
      break;
    case 4:
      glUniformMatrix4fv(loc, 1, GL_FALSE, (const GLfloat*)m_data.f);
      break;
    };
    break;
  };
  UtilGL::CheckErrors("glUniform setting");
}

//-*****************************************************************************
//-*****************************************************************************
// PROGRAM
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
Program::Program(const std::string& i_name, const std::string& i_vtx,
                 const std::string& i_geom, const std::string& i_frg,
                 const Bindings& i_vtxBindingsIn,
                 const Bindings& i_frgBindingsOut, GLuint i_vertexArrayObject)
  : m_name(i_name) {
  // Bind the vertex array object
  if (i_vertexArrayObject > 0) {
    glBindVertexArray(i_vertexArrayObject);
    UtilGL::CheckErrors("glBindVertexArray");
  }

  // Do the program creation.
  m_progId = glCreateProgram();
  if (m_progId == 0) {
    EWAV_THROW("Couldn't allocate GLSL program: " << m_name);
  }
  std::cout << "Created GLSL program" << std::endl;

  //-*************************************************************************
  // VERTEX SHADER
  //-*************************************************************************
  m_vtxId = 0;
  std::vector<std::string> vtxSources;
  vtxSources.push_back(i_vtx);
  if (vtxSources.size() > 0) {
    m_vtxId = initShader(m_name + "::vertex", GL_VERTEX_SHADER, vtxSources);
    glAttachShader(m_progId, m_vtxId);
  }
  std::cout << "Created GLSL vertex shader" << std::endl;

  //-*************************************************************************
  // GEOMETRY SHADER (they're optional!)
  //-*************************************************************************
  m_geomId = 0;
  if (i_geom != "") {
    std::vector<std::string> geomSources;
    geomSources.push_back(i_geom);
    if (geomSources.size() > 0) {
      m_geomId =
        initShader(m_name + "::geometry", GL_GEOMETRY_SHADER, geomSources);
      glAttachShader(m_progId, m_geomId);
    }
    std::cout << "Created GLSL geometry shader" << std::endl;
  }

  //-*************************************************************************
  // FRAGMENT SHADER
  //-*************************************************************************
  m_frgId = 0;
  std::vector<std::string> frgSources;
  frgSources.push_back(i_frg);
  if (frgSources.size() > 0) {
    m_frgId = initShader(m_name + "::fragment", GL_FRAGMENT_SHADER, frgSources);
    glAttachShader(m_progId, m_frgId);
  }
  std::cout << "Created GLSL fragment shader" << std::endl;

  //-*************************************************************************
  // Bind vertex attributes
  //-*************************************************************************
  for (Bindings::const_iterator biter = i_vtxBindingsIn.begin();
       biter != i_vtxBindingsIn.end(); ++biter) {
    glBindAttribLocation(m_progId, (*biter).first,
                         (const GLchar*)((*biter).second.c_str()));
    UtilGL::CheckErrors("glBindAttribLocation");
  }

  //-*************************************************************************
  // Bind fragment data
  //-*************************************************************************
  for (Bindings::const_iterator biter = i_frgBindingsOut.begin();
       biter != i_frgBindingsOut.end(); ++biter) {
    glBindFragDataLocation(m_progId, (*biter).first,
                           (const GLchar*)((*biter).second.c_str()));
  }

  //-*************************************************************************
  // Link program
  //-*************************************************************************
  glLinkProgram(m_progId);
  std::cout << "Linked GLSL program." << std::endl;

  GLint linked = 0;
  glGetProgramiv(m_progId, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    GLint length = 0;
    glGetProgramiv(m_progId, GL_INFO_LOG_LENGTH, &length);

    std::vector<GLchar> log(length + 1);
    glGetProgramInfoLog(m_progId, length, &length, &(log[0]));
    std::string logStr = (const std::string&)&(log[0]);
    EWAV_THROW("Linking error in program: " << m_name << std::endl << logStr);
  }

  //-*************************************************************************
  // Check vertex attribute bindings
  //-*************************************************************************
  for (Bindings::const_iterator biter = i_vtxBindingsIn.begin();
       biter != i_vtxBindingsIn.end(); ++biter) {
    GLint result =
      glGetAttribLocation(m_progId, (const GLchar*)((*biter).second.c_str()));
    UtilGL::CheckErrors("glGetAttribLocation");
    EWAV_ASSERT(result == (*biter).first,
                "Did not successfully bind attribute: "
                  << (*biter).second << ", got result: " << result
                  << ", but wanted: " << (*biter).first);
  }

  GLint validate = 0;
  glValidateProgram(m_progId);
  glGetProgramiv(m_progId, GL_VALIDATE_STATUS, &validate);
  if (validate != GL_TRUE) {
    GLint length = 0;
    glGetProgramiv(m_progId, GL_INFO_LOG_LENGTH, &length);

    std::vector<GLchar> log(length + 1);
    glGetProgramInfoLog(m_progId, length, &length, &(log[0]));
    std::string logStr = (const std::string&)&(log[0]);

    EWAV_THROW("Given vertex/fragment program: "
               << m_name << " won't run on this hardware" << std::endl
               << logStr);
  }
  std::cout << "Validated GLSL program." << std::endl;

  // Bind the vertex array object
  if (i_vertexArrayObject > 0) {
    glBindVertexArray(0);
    UtilGL::CheckErrors("glBindVertexArray");
  }
}

//-*****************************************************************************
Program::~Program() {
  if (m_progId > 0) {
    glDeleteProgram(m_progId);
    m_progId = 0;
  }

  if (m_vtxId > 0) {
    glDeleteShader(m_vtxId);
    m_vtxId = 0;
  }

  if (m_frgId > 0) {
    glDeleteShader(m_frgId);
    m_frgId = 0;
  }

  if (m_geomId > 0) {
    glDeleteShader(m_geomId);
    m_geomId = 0;
  }
}

//-*****************************************************************************
// Set uniforms on the program.
void Program::operator()(const Uniform& i_uniform) {
  m_uniforms[i_uniform.name()] = i_uniform;
}

//-*****************************************************************************
void Program::use() const {
  EWAV_ASSERT(m_progId > 0, "Cannot use program 0");

  // Use the program
  glUseProgram(m_progId);

  setUniforms();
}

//-*****************************************************************************
void Program::unuse() const {
  EWAV_ASSERT(m_progId > 0, "Cannot unuse program 0");

  // Unuse program
  glUseProgram(0);
}

//-*****************************************************************************
void Program::setUniforms() const {
  // Set the uniforms.
  for (Uniforms::const_iterator uiter = m_uniforms.begin();
       uiter != m_uniforms.end(); ++uiter) {
    (*uiter).second.set(m_progId);
  }
}

//-*****************************************************************************
GLuint Program::initShader(const std::string& i_shaderName, GLenum i_type,
                           const std::vector<std::string>& i_sources) {
  GLuint id = 0;

  static const GLchar* shaderSources[32];
  GLsizei numSources = i_sources.size();
  assert(numSources > 0);

  if (numSources > 32) {
    EWAV_THROW("Can't compile shader: " << i_shaderName << std::endl
                                        << "Too many shader sources: "
                                        << numSources << ". Max = 32");
  }

  std::vector<std::string> newSources(i_sources.size());

  for (int i = 0; i < numSources; ++i) {
    shaderSources[i] = (const GLchar*)(i_sources[i].c_str());
  }

  for (int i = numSources; i < 32; ++i) {
    shaderSources[i] = NULL;
  }

  id = glCreateShader(i_type);
  if (id == 0) {
    EWAV_THROW("Could not create shader: " << i_shaderName);
  }

  glShaderSource(id, numSources, shaderSources, NULL);
  glCompileShader(id);
  GLint compiled = 0;
  glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    GLint length = 0;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
    std::vector<GLchar> log(length + 1);
    glGetShaderInfoLog(id, length, &length, &(log[0]));
    std::string strLog = (const char*)&(log[0]);
    EWAV_THROW("Compilation error in shader: " << i_shaderName << std::endl
                                               << strLog);
  }

  return id;
}

}  // namespace GeepGLFW
}  // namespace EncinoWaves
