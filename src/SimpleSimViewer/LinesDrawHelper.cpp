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

#include "LinesDrawHelper.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
LinesDrawHelper::LinesDrawHelper(bool i_dynamic, std::size_t i_numPoints,
                                 const V3f* i_vtxPosData,
                                 const V3f* i_vtxNormData,
                                 const V3f* i_vtxColData,
                                 const V2f* i_vtxUvData)
  : m_numPoints(i_numPoints)
  , m_vtxPosData(i_vtxPosData)
  , m_vtxNormData(i_vtxNormData)
  , m_vtxColData(i_vtxColData)
  , m_vtxUvData(i_vtxUvData)
  , m_vertexArrayObject(0)
  , m_posVboIdx(-1)
  , m_normVboIdx(-1)
  , m_colVboIdx(-1)
  , m_uvVboIdx(-1) {
  //-*************************************************************************
  // OPENGL INIT
  //-*************************************************************************
  UtilGL::CheckErrors("line draw helper init before anything");

  // Create and bind VAO
  glGenVertexArrays(1, &m_vertexArrayObject);
  UtilGL::CheckErrors("glGenVertexArrays");
  EWAV_ASSERT(m_vertexArrayObject > 0, "Failed to create VAO");

  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  // Figure out how many VBOs to make.
  EWAV_ASSERT(m_vtxPosData != NULL, "Must have vertex data.");
  m_numVBOs = 1;
  if (m_vtxNormData) {
    ++m_numVBOs;
  }
  if (m_vtxColData) {
    ++m_numVBOs;
  }
  if (m_vtxUvData) {
    ++m_numVBOs;
  }

  // Create vertex buffers.
  glGenBuffers(m_numVBOs, m_vertexBuffers);
  UtilGL::CheckErrors("glGenBuffers");
  EWAV_ASSERT(m_vertexBuffers[0] > 0, "Failed to create VBOs");

  GLenum vtxDataDyn = GL_STATIC_DRAW;
  if (i_dynamic) {
    vtxDataDyn = GL_DYNAMIC_DRAW;
  }

  // POS buffer
  int vboIdx  = 0;
  m_posVboIdx = vboIdx;
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
  UtilGL::CheckErrors("glBindBuffer POS");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numPoints,
               (const GLvoid*)m_vtxPosData, vtxDataDyn);
  UtilGL::CheckErrors("glBufferData POS");
  glVertexAttribPointer(vboIdx, 3, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer POS");
  glEnableVertexAttribArray(vboIdx);
  UtilGL::CheckErrors("glEnableVertexAttribArray POS");

  // If NORM.
  m_normVboIdx = -1;
  if (m_vtxNormData) {
    ++vboIdx;
    m_normVboIdx = vboIdx;
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
    UtilGL::CheckErrors("glBindBuffer NORM");
    glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numPoints,
                 (const GLvoid*)m_vtxNormData, vtxDataDyn);
    UtilGL::CheckErrors("glBufferData NORM");
    glVertexAttribPointer(vboIdx, 3, GL_FLOAT, GL_FALSE, 0, 0);
    UtilGL::CheckErrors("glVertexAttribPointer NORM");
    glEnableVertexAttribArray(vboIdx);
    UtilGL::CheckErrors("glEnableVertexAttribArray NORM");
  }

  // If COLOR.
  m_colVboIdx = -1;
  if (m_vtxColData) {
    ++vboIdx;
    m_colVboIdx = vboIdx;
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
    UtilGL::CheckErrors("glBindBuffer COLOR");
    glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numPoints,
                 (const GLvoid*)m_vtxColData, vtxDataDyn);
    UtilGL::CheckErrors("glBufferData COLOR");
    glVertexAttribPointer(vboIdx, 3, GL_FLOAT, GL_FALSE, 0, 0);
    UtilGL::CheckErrors("glVertexAttribPointer COLOR");
    glEnableVertexAttribArray(vboIdx);
    UtilGL::CheckErrors("glEnableVertexAttribArray COLOR");
  }

  // If UVs.
  m_uvVboIdx = -1;
  if (m_vtxUvData) {
    ++vboIdx;
    m_uvVboIdx = vboIdx;
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
    UtilGL::CheckErrors("glBindBuffer UV");
    glBufferData(GL_ARRAY_BUFFER, sizeof(V2f) * m_numPoints,
                 (const GLvoid*)m_vtxUvData, vtxDataDyn);
    UtilGL::CheckErrors("glBufferData UV");
    glVertexAttribPointer(vboIdx, 2, GL_FLOAT, GL_FALSE, 0, 0);
    UtilGL::CheckErrors("glVertexAttribPointer UV");
    glEnableVertexAttribArray(vboIdx);
    UtilGL::CheckErrors("glEnableVertexAttribArray UV");
  }

  EWAV_ASSERT(vboIdx == m_numVBOs - 1, "Mismatched vbo idx.");

  // Unbind VAO.
  glBindVertexArray(0);
  UtilGL::CheckErrors("Unbind VAO");
}

//-*****************************************************************************
LinesDrawHelper::~LinesDrawHelper() {
  if (m_vertexArrayObject > 0) {
    glDeleteVertexArrays(1, &m_vertexArrayObject);
    m_vertexArrayObject = 0;
  }

  if (m_numVBOs > 0 && m_vertexBuffers[0] > 0) {
    glDeleteBuffers(m_numVBOs, m_vertexBuffers);
    for (int i = 0; i < m_numVBOs; ++i) {
      m_vertexBuffers[i] = 0;
    }
  }
}

//-*****************************************************************************
void LinesDrawHelper::update(std::size_t i_numPoints, const V3f* i_vtxPosData,
                             const V3f* i_vtxNormData, const V3f* i_vtxColData,
                             const V2f* i_vtxUvData) {
  m_numPoints = i_numPoints;

  // Activate VAO
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  if (i_vtxPosData) {
    m_vtxPosData = i_vtxPosData;
    updateFloatVertexBuffer<V3f>(m_vtxPosData, m_posVboIdx);
  }
  if (i_vtxNormData) {
    m_vtxNormData = i_vtxNormData;
    updateFloatVertexBuffer<V3f>(m_vtxNormData, m_normVboIdx);
  }
  if (i_vtxColData) {
    m_vtxColData = i_vtxColData;
    updateFloatVertexBuffer<V3f>(m_vtxColData, m_colVboIdx);
  }
  if (i_vtxUvData) {
    m_vtxUvData = i_vtxUvData;
    updateFloatVertexBuffer<V2f>(m_vtxUvData, m_uvVboIdx);
  }

  // Unbind
  glBindVertexArray(0);
}

//-*****************************************************************************
void LinesDrawHelper::draw(const GLCamera& i_cam) const {
  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Draw the arrays
  glDrawArrays(GL_LINES, 0, m_numPoints);
  UtilGL::CheckErrors("glDrawArrays");

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

//-*****************************************************************************
void LinesDrawHelper::draw() const {
  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Draw the arrays
  glDrawArrays(GL_LINES, 0, m_numPoints);
  UtilGL::CheckErrors("glDrawArrays");

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves
