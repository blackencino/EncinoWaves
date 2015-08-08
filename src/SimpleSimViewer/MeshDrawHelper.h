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

#ifndef _EncinoWaves_SimpleSimViewer_MeshDrawHelper_h_
#define _EncinoWaves_SimpleSimViewer_MeshDrawHelper_h_

#include "Foundation.h"
#include "GLCamera.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
class MeshDrawHelper {
public:
  enum DeformType {
    // No changes at all
    kStaticDeform,

    // Vertex data (positions, colors, normals, etc) change, but
    // not indices
    kConsistentDeform,

    // Everything changes
    kInconsistentDeform
  };

  MeshDrawHelper(DeformType i_deformType, std::size_t i_numTriangles,
                 std::size_t i_numVertices, const V3ui* i_triIndices,
                 const V3f* i_vtxPosData, const V3f* i_vtxNormData,
                 const V3f* i_vtxColData, const V2f* i_vtxUvData);

  virtual ~MeshDrawHelper();

  // Update only the vertex data, leave topology alone
  void update(const V3f* i_vtxPosData, const V3f* i_vtxNormData,
              const V3f* i_vtxColData, const V2f* i_vtxUvData);

  // Update EVERYTHING.
  void update(std::size_t i_numTriangles, std::size_t i_numVertices,
              const V3ui* i_triIndices, const V3f* i_vtxPosData,
              const V3f* i_vtxNormData, const V3f* i_vtxColData,
              const V2f* i_vtxUvData);

  // Draw.
  void draw(const GLCamera& i_cam) const;

  // Draw without camera.
  void draw() const;

  GLint posVboIdx() const { return m_posVboIdx; }
  GLint normVboIdx() const { return m_normVboIdx; }
  GLint colVboIdx() const { return m_colVboIdx; }
  GLint uvVboIdx() const { return m_uvVboIdx; }
  GLint indicesVboIdx() const { return m_indicesVboIdx; }

  GLuint vertexArrayObject() const { return m_vertexArrayObject; }

protected:
  template <typename T>
  void updateFloatVertexBuffer(const T* i_vtxData, int i_vboIdx) {
    if (i_vboIdx < 0 || i_vtxData == NULL || m_numVertices == 0) {
      return;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[i_vboIdx]);
    UtilGL::CheckErrors("glBindBuffer");
    glBufferData(GL_ARRAY_BUFFER, sizeof(T) * m_numVertices,
                 (const GLvoid*)i_vtxData, GL_DYNAMIC_DRAW);
    UtilGL::CheckErrors("glBufferData");
  }

  DeformType m_deformType;
  std::size_t m_numTriangles;
  std::size_t m_numVertices;
  const V3ui* m_triIndices;
  const V3f* m_vtxPosData;
  const V3f* m_vtxNormData;
  const V3f* m_vtxColData;
  const V2f* m_vtxUvData;

  // The VAO and VBOs
  GLuint m_vertexArrayObject;
  // At most we have indices, pos, norm, col, uv
  GLuint m_vertexBuffers[5];
  int m_numVBOs;
  GLint m_posVboIdx;
  GLint m_normVboIdx;
  GLint m_colVboIdx;
  GLint m_uvVboIdx;
  GLint m_indicesVboIdx;
};

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
