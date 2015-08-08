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

#include "OceanTestEnvSphere.h"
#include "OceanTestShaders.h"

namespace OceanTest {

class MeshSphereBuilder {
public:
  MeshSphereBuilder(std::vector<V3f>& vertices, std::vector<V3f>& normals,
                    std::vector<uint32_t>& tri_indices, float radius,
                    int num_lat_steps, int num_long_steps);

private:
  void makeCenterVertex(float position_z);
  void makeRingOfVertices(float latitude);
  void makeBottomCap();
  void makeRib(int first_bottom_vertex_index);
  void makeTopCap(int first_bottom_vertex_index);

  std::vector<V3f>& m_vertices;
  std::vector<V3f>& m_normals;
  std::vector<uint32_t>& m_tri_indices;
  float m_radius;
  int m_num_lat_steps;
  int m_num_long_steps;
};

MeshSphereBuilder::MeshSphereBuilder(std::vector<V3f>& vertices,
                                     std::vector<V3f>& normals,
                                     std::vector<uint32_t>& tri_indices,
                                     float radius, int num_lat_steps,
                                     int num_long_steps)
  : m_vertices(vertices)
  , m_normals(normals)
  , m_tri_indices(tri_indices)
  , m_radius(radius)
  , m_num_lat_steps(num_lat_steps)
  , m_num_long_steps(num_long_steps) {
  // make bottom center point
  makeCenterVertex(-m_radius);

  // Latitude step radians
  float lat_angle_step          = radians(180.0f / float(m_num_lat_steps));
  static const float bottom_lat = radians(-90.0f);
  static const float top_lat    = radians(90.0f);

  // make bottom ring
  makeRingOfVertices(bottom_lat + lat_angle_step);

  // make bottom cap.
  makeBottomCap();

  // for each length step, make ring with normals pointing out, then
  // make open cylinder.
  for (int lat_step = 1; lat_step < m_num_lat_steps - 1; ++lat_step) {
    float lat =
      mix(bottom_lat, top_lat, float(lat_step + 1) / float(m_num_lat_steps));
    makeRingOfVertices(lat);

    makeRib((int)(m_vertices.size() - 2 * m_num_long_steps));
  }

  // make top center point.
  makeCenterVertex(m_radius);

  // make top cap.
  makeTopCap((int)(m_vertices.size() - (1 + m_num_long_steps)));
}

// Make a center vertex at some height.
void MeshSphereBuilder::makeCenterVertex(float position_z) {
  m_vertices.push_back(V3f(0.0f, 0.0f, position_z));
  m_normals.push_back(V3f(0.0f, 0.0f, std::copysign(1.0f, position_z)));
}

// Make a ring of vertices at some height, with a plane normal, or pointing
// outwards, with some angle step.
void MeshSphereBuilder::makeRingOfVertices(float latitude) {
  float cos_lat = std::cos(latitude);
  float sin_lat = std::sin(latitude);

  float long_step = radians(360.0f / float(m_num_long_steps));
  for (int i = 0; i < m_num_long_steps; ++i) {
    float longitude = long_step * float(i);
    V3f n(cos_lat * std::cos(longitude), cos_lat * std::sin(longitude),
          sin_lat);
    m_vertices.push_back(m_radius * n);
    m_normals.push_back(n);
  }
}

// Make the bottom cap, starting at the 0th vertex.
void MeshSphereBuilder::makeBottomCap() {
  for (int i = 0; i < m_num_long_steps; ++i) {
    m_tri_indices.push_back(0);
    m_tri_indices.push_back(1 + wrap(i + 1, m_num_long_steps));
    m_tri_indices.push_back(1 + i);
  }
}

// Make a rib from two rings of vertices.
void MeshSphereBuilder::makeRib(int first_bottom_vertex_index) {
  for (int i = 0; i < m_num_long_steps; ++i) {
    int down_i      = first_bottom_vertex_index + i;
    int next_down_i = first_bottom_vertex_index + wrap(i + 1, m_num_long_steps);
    int up_i        = down_i + m_num_long_steps;
    int next_up_i = next_down_i + m_num_long_steps;
    m_tri_indices.push_back(down_i);
    m_tri_indices.push_back(next_up_i);
    m_tri_indices.push_back(up_i);
    m_tri_indices.push_back(down_i);
    m_tri_indices.push_back(next_down_i);
    m_tri_indices.push_back(next_up_i);
  }
}

// Make the top cap, starting at the given vertex for the bottom ring.
void MeshSphereBuilder::makeTopCap(int first_bottom_vertex_index) {
  for (int i = 0; i < m_num_long_steps; ++i) {
    m_tri_indices.push_back(first_bottom_vertex_index + i);
    m_tri_indices.push_back(first_bottom_vertex_index +
                            wrap(i + 1, m_num_long_steps));
    m_tri_indices.push_back(first_bottom_vertex_index + m_num_long_steps);
  }
}

void makeMeshSphere(std::vector<V3f>& vertices, std::vector<V3f>& normals,
                    std::vector<uint32_t>& tri_indices, float radius,
                    int num_lat_steps, int num_long_steps) {
  vertices.clear();
  normals.clear();
  tri_indices.clear();
  MeshSphereBuilder builder{vertices, normals,       tri_indices,
                            radius,   num_lat_steps, num_long_steps};
}

EnvSphere::EnvSphere() {
  std::vector<V3f> positions;
  std::vector<V3f> normals;
  std::vector<uint32_t> tri_indices;
  makeMeshSphere(positions, normals, tri_indices, 2000, 32, 16);
  m_num_indices = tri_indices.size();

  UtilGL::CheckErrors("mesh init before anything");

  // Create and bind VAO
  glGenVertexArrays(1, &m_vertexArrayObject);
  UtilGL::CheckErrors("glGenVertexArrays");
  EWAV_ASSERT(m_vertexArrayObject > 0, "Failed to create VAO");

  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  // Create vertex buffers.
  glGenBuffers(2, m_vertexBuffers);
  UtilGL::CheckErrors("glGenBuffers");
  EWAV_ASSERT(m_vertexBuffers[0] > 0, "Failed to create VBOs");

  // Data size.
  std::size_t dataSize = positions.size();

  // xy buffer
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[0]);
  UtilGL::CheckErrors("glBindBuffer Vertices");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * dataSize, positions.data(),
               GL_STATIC_DRAW);
  UtilGL::CheckErrors("glBufferData Vertices");
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer Vertices");
  glEnableVertexAttribArray(0);
  UtilGL::CheckErrors("glEnableVertexAttribArray Vertices");

  // Indices buffer.
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexBuffers[1]);
  UtilGL::CheckErrors("glBindBuffer INDICES");
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * tri_indices.size(),
               tri_indices.data(), GL_STATIC_DRAW);
  UtilGL::CheckErrors("glBufferData INDICES");

  createProgram();

  // Unbind vao.
  glBindVertexArray(0);
  UtilGL::CheckErrors("Unbind VAO");
}

EnvSphere::~EnvSphere() {
  if (m_vertexArrayObject > 0) {
    glDeleteVertexArrays(1, &m_vertexArrayObject);
    m_vertexArrayObject = 0;
  }

  if (m_vertexBuffers[0] > 0) {
    glDeleteBuffers(2, m_vertexBuffers);
    for (int i = 0; i < 2; ++i) {
      m_vertexBuffers[i] = 0;
    }
  }
}

void EnvSphere::createProgram() {
  // Make the bindings
  Program::Bindings vtxBindings;
  vtxBindings.push_back(Program::Binding(0, "g_vertex"));

  Program::Bindings frgBindings;
  frgBindings.push_back(Program::Binding(0, "g_fragmentColor"));

  m_program.reset(new Program("OceanTestEnvDraw", EnvVertexShader(), "",
                              EnvFragmentShader(), vtxBindings, frgBindings,
                              m_vertexArrayObject));
}

void EnvSphere::setCameraUniforms(
  const EncinoWaves::SimpleSimViewer::GLCamera& i_cam) {
  M44d pm  = i_cam.projectionMatrix();
  M44d mvm = i_cam.modelViewMatrix();
  (*m_program)(Uniform("projection_matrix", pm));
  (*m_program)(Uniform("modelview_matrix", mvm));
}

void EnvSphere::draw(const EncinoWaves::SimpleSimViewer::GLCamera& i_cam,
                     const TextureSky& sky) {
  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Use the program
  m_program->use();

  // Set the uniforms
  sky.bind(m_program->id());
  setCameraUniforms(i_cam);
  m_program->setUniforms();

  // Draw the elements
  glDrawElements(GL_TRIANGLES, m_num_indices, GL_UNSIGNED_INT, 0);
  UtilGL::CheckErrors("glDrawElements");

  // Unuse the program
  m_program->unuse();

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

}  // namespace OceanTest
