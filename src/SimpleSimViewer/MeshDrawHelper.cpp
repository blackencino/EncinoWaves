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

#include "MeshDrawHelper.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
MeshDrawHelper::MeshDrawHelper(
  DeformType i_deformType, std::size_t i_numTriangles,
  std::size_t i_numVertices, const V3ui* i_triIndices, const V3f* i_vtxPosData,
  const V3f* i_vtxNormData, const V3f* i_vtxColData, const V2f* i_vtxUvData)
  : m_deformType(i_deformType)
  , m_numTriangles(i_numTriangles)
  , m_numVertices(i_numVertices)
  , m_triIndices(i_triIndices)
  , m_vtxPosData(i_vtxPosData)
  , m_vtxNormData(i_vtxNormData)
  , m_vtxColData(i_vtxColData)
  , m_vtxUvData(i_vtxUvData)
  , m_vertexArrayObject(0)
  , m_posVboIdx(-1)
  , m_normVboIdx(-1)
  , m_colVboIdx(-1)
  , m_uvVboIdx(-1)
  , m_indicesVboIdx(-1) {
  //-*************************************************************************
  // OPENGL INIT
  //-*************************************************************************
  UtilGL::CheckErrors("mesh draw helper init before anything");

  // Create and bind VAO
  glGenVertexArrays(1, &m_vertexArrayObject);
  UtilGL::CheckErrors("glGenVertexArrays");
  EWAV_ASSERT(m_vertexArrayObject > 0, "Failed to create VAO");

  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  // Figure out how many VBOs to make.
  EWAV_ASSERT(m_vtxPosData != NULL && m_triIndices != NULL,
              "Must have vertex and index data.");
  m_numVBOs = 2;
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
  if (i_deformType != kStaticDeform) {
    vtxDataDyn = GL_DYNAMIC_DRAW;
  }
  GLenum indexDyn = GL_STATIC_DRAW;
  if (i_deformType == kInconsistentDeform) {
    indexDyn = GL_DYNAMIC_DRAW;
  }

  // POS buffer
  int vboIdx  = 0;
  m_posVboIdx = vboIdx;
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
  UtilGL::CheckErrors("glBindBuffer POS");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numVertices,
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numVertices,
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * m_numVertices,
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(V2f) * m_numVertices,
                 (const GLvoid*)m_vtxUvData, vtxDataDyn);
    UtilGL::CheckErrors("glBufferData UV");
    glVertexAttribPointer(vboIdx, 2, GL_FLOAT, GL_FALSE, 0, 0);
    UtilGL::CheckErrors("glVertexAttribPointer UV");
    glEnableVertexAttribArray(vboIdx);
    UtilGL::CheckErrors("glEnableVertexAttribArray UV");
  }

  // Indices buffer.
  ++vboIdx;
  m_indicesVboIdx = vboIdx;
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexBuffers[vboIdx]);
  UtilGL::CheckErrors("glBindBuffer INDICES");
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(V3ui) * m_numTriangles,
               (const GLvoid*)m_triIndices, indexDyn);
  UtilGL::CheckErrors("glBufferData INDICES");

  EWAV_ASSERT(vboIdx == m_numVBOs - 1, "Mismatched vbo idx.");

  // Unbind VAO.
  glBindVertexArray(0);
  UtilGL::CheckErrors("Unbind VAO");
}

//-*****************************************************************************
MeshDrawHelper::~MeshDrawHelper() {
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
// Static topology
void MeshDrawHelper::update(const V3f* i_vtxPosData, const V3f* i_vtxNormData,
                            const V3f* i_vtxColData, const V2f* i_vtxUvData) {
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
// Dynamic topology
void MeshDrawHelper::update(std::size_t i_numTriangles,
                            std::size_t i_numVertices, const V3ui* i_triIndices,
                            const V3f* i_vtxPosData, const V3f* i_vtxNormData,
                            const V3f* i_vtxColData, const V2f* i_vtxUvData) {
  m_numTriangles = i_numTriangles;
  m_numVertices  = i_numVertices;
  m_triIndices   = i_triIndices;

  // Activate VAO
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  if (m_numTriangles > 0) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexBuffers[m_indicesVboIdx]);
    UtilGL::CheckErrors("glBindBuffer INDICES");
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(V3ui) * m_numTriangles,
                 (const GLvoid*)m_triIndices, GL_DYNAMIC_DRAW);
    UtilGL::CheckErrors("glBufferData INDICES");
  }

  // Update mesh stuff.
  update(i_vtxPosData, i_vtxNormData, i_vtxColData, i_vtxUvData);

  // Unbind
  glBindVertexArray(0);
}

//-*****************************************************************************
void MeshDrawHelper::draw(const GLCamera& i_cam) const {
  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Draw the elements
  glDrawElements(GL_TRIANGLES, 3 * m_numTriangles, GL_UNSIGNED_INT, 0);
  UtilGL::CheckErrors("glDrawElements");

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

//-*****************************************************************************
void MeshDrawHelper::draw() const {
  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Draw the elements
  glDrawElements(GL_TRIANGLES, 3 * m_numTriangles, GL_UNSIGNED_INT, 0);
  UtilGL::CheckErrors("glDrawElements");

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves
