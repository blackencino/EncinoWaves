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

#include "StdShaders.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
// Header.
static const char* g_shaderHeader =
  R"(
    #version 150
)";

//-*****************************************************************************
static const char* g_simpleVertexShaderBase =
  R"(
    in vec3 g_Pobj;
    in vec3 g_Nobj;

    out vec3 gv_Pobj;
    out vec3 gv_Nobj;

    void main()
    {
       gv_Pobj = g_Pobj;
       gv_Nobj = g_Nobj;
       gl_Position = vec4( g_Pobj, 1 );
    }
)";

//-*****************************************************************************
static const char* g_transformFunctions =
  "vec3 transform( vec3 p, mat4 m )                           \n"
  "{                                                          \n"
  "    return vec3( m * vec4( p, 1.0 ) );                            \n"
  "}                                                          \n"
  "vec3 vtransform( vec3 v, mat4 m )                          \n"
  "{                                                          \n"
  "    return vec3( m * vec4( v, 0.0 ) );                     \n"
  "}                                                          \n"
  "vec3 ntransform( vec3 n, mat4 m )                          \n"
  "{                                                          \n"
  "    return normalize( vtransform( n, m ) );                \n"
  "}                                                          \n"
  "                                                           \n";

//-*****************************************************************************
static const char* g_stdMatrices =
  "uniform mat4 projection_matrix;                            \n"
  "\n"
  "uniform mat4 world_to_rhc_matrix;                          \n"
  "uniform mat4 world_to_rhc_nmatrix;                         \n"
  "\n"
  "uniform mat4 rhc_to_world_matrix;                          \n"
  "uniform mat4 rhc_to_world_nmatrix;                         \n"
  "\n"
  "uniform mat4 obj_to_world_matrix;                          \n"
  "uniform mat4 obj_to_world_nmatrix;                         \n"
  "\n"
  "uniform mat4 world_to_obj_matrix;                          \n"
  "uniform mat4 world_to_obj_nmatrix;                         \n"
  "\n"
  "uniform vec3 g_eyeWld;                                     \n";

//-*****************************************************************************
static const char* g_simplePointsGeometryShaderBase =
  "layout(points) in;                                         \n"
  "layout(points, max_vertices = 1) out;                      \n"
  "in vec3 gv_Pobj[1];                                        \n"
  "in vec3 gv_Nobj[1];                                        \n"
  "out vec3 gg_Pwld;                                          \n"
  "out vec3 gg_Nwld;                                          \n"
  "                                                           \n"
  "uniform float g_pointSize;                                 \n"
  "                                                           \n"
  "void main( void )                                          \n"
  "{                                                          \n"
  "   mat4 modelview_matrix =                                 \n"
  "       world_to_rhc_matrix * obj_to_world_matrix;          \n"
  "   mat4 pmv = projection_matrix * modelview_matrix;        \n"
  "                                                           \n"
  "   gg_Pwld = transform( gv_Pobj[0], obj_to_world_matrix );        \n"
  "   gg_Nwld = ntransform( gv_Nobj[0], obj_to_world_nmatrix );       \n"
  "   gl_Position = pmv * vec4( gv_Pobj[0], 1 );              \n"
  "   gl_PointSize = g_pointSize;                             \n"
  "   EmitVertex(); //EndPrimitive();                           \n"
  "}                                                          \n"
  "                                                           \n";

//-*****************************************************************************
static const char* g_simpleTrianglesGeometryShaderBase =
  "layout(triangles) in;                                      \n"
  "layout(triangle_strip, max_vertices = 3) out;              \n"
  "in vec3 gv_Pobj[3];                                        \n"
  "in vec3 gv_Nobj[3];                                        \n"
  "out vec3 gg_Pwld;                                          \n"
  "out vec3 gg_Nwld;                                          \n"
  "                                                           \n"
  "void main( void )                                          \n"
  "{                                                          \n"
  "   mat4 modelview_matrix =                                 \n"
  "       world_to_rhc_matrix * obj_to_world_matrix;          \n"
  "   mat4 pmv = projection_matrix * modelview_matrix;        \n"
  "                                                           \n"
  "   for ( int i = 0; i < 3; ++i )                           \n"
  "   {                                                       \n"
  "       gg_Pwld = transform( gv_Pobj[i], obj_to_world_matrix );    \n"
  "       gg_Nwld = ntransform( gv_Nobj[i], obj_to_world_nmatrix );   \n"
  "       gl_Position = pmv * vec4( gv_Pobj[i], 1 );          \n"
  "       EmitVertex();                                       \n"
  "   }                                                       \n"
  "   EndPrimitive();                                         \n"
  "}                                                          \n"
  "                                                           \n";

//-*****************************************************************************
static const char* g_simpleTrianglesWireframeGeometryShaderBase =
  "layout(triangles) in;                                      \n"
  "layout(line_strip, max_vertices = 3) out;                  \n"
  "in vec3 gv_Pobj[3];                                        \n"
  "in vec3 gv_Nobj[3];                                        \n"
  "out vec3 gg_Pwld;                                          \n"
  "out vec3 gg_Nwld;                                          \n"
  "                                                           \n"
  "void main( void )                                          \n"
  "{                                                          \n"
  "   mat4 modelview_matrix =                                 \n"
  "       world_to_rhc_matrix * obj_to_world_matrix;          \n"
  "   mat4 pmv = projection_matrix * modelview_matrix;        \n"
  "                                                           \n"
  "   for ( int i = 0; i < 3; ++i )                           \n"
  "   {                                                       \n"
  "       gg_Pwld = transform( gv_Pobj[i], obj_to_world_matrix );    \n"
  "       gg_Nwld = ntransform( gv_Nobj[i], obj_to_world_nmatrix );   \n"
  "       gl_Position = pmv * vec4( gv_Pobj[i], 1 );          \n"
  "       EmitVertex();                                       \n"
  "   }                                                       \n"
  "   EndPrimitive();                                         \n"
  "}                                                          \n"
  "                                                           \n";

//-*****************************************************************************
static const char* g_specDiffuseGamma =
  "//-***************************************************************\n"
  "vec3 gammaCorrect( in vec3 col, in float g )                      \n"
  "{                                                                 \n"
  "    return vec3( pow( clamp( col.r, 0.0, 1.0 ), 1.0/g ),          \n"
  "                 pow( clamp( col.g, 0.0, 1.0 ), 1.0/g ),          \n"
  "                 pow( clamp( col.b, 0.0, 1.0 ), 1.0/g ) );        \n"
  "}                                                                 \n"
  "                                                                  \n"
  "                                                                  \n"
  "//-***************************************************************\n"
  "float kSpecular( vec3 In, vec3 Nn, vec3 Ln, float m )             \n"
  "{                                                                 \n"
  "    vec3 Vn = -In;                                                \n"
  "    vec3 H = normalize( Ln + Vn );                                \n"
  "    float d = dot( Nn, H ); d *= d;                               \n"
  "    d = max( d, 0.0 );                                            \n"
  "    return pow( d, m/2 );                                         \n"
  "}                                                                 \n"
  "                                                                  \n"
  "//-***************************************************************\n"
  "float kDiffuse( vec3 Nn, vec3 Ln )                                \n"
  "{                                                                 \n"
  "    float d = dot( Nn, Ln );                                      \n"
  "    return clamp( d, 0, 1 );                                      \n"
  "}                                                                 \n"
  "                                                                  \n";

//-*****************************************************************************
static const char* g_keyFillFragmentShaderBase =
  "in vec3 gg_Pwld;                                                  \n"
  "in vec3 gg_Nwld;                                                  \n"
  "out vec4 g_fragmentColor;                                         \n"
  "                                                                  \n"
  "uniform vec3 g_toKey;                                             \n"
  "uniform vec3 g_keyColor;                                          \n"
  "uniform vec3 g_toFill;                                            \n"
  "uniform vec3 g_fillColor;                                         \n"
  "uniform vec3 g_diffColor;                                         \n"
  "uniform vec3 g_specColor;                                         \n"
  "uniform float g_specExponent;                                     \n"
  "                                                                  \n"
  "//-***************************************************************\n"
  "float sqr( float a ) { return a * a; }                            \n"
  "                                                                  \n"
  "//-***************************************************************\n"
  "float linstep( float edge0, float edge1, float t )                \n"
  "{                                                                 \n"
  "    return clamp( ( t - edge0 ) / ( edge1 - edge0 ),              \n"
  "                  0, 1 );                                         \n"
  "}                                                                 \n"
  "                                                                  \n"
  "                                                                  \n"
  "//-***************************************************************\n"
  "void main()                                                       \n"
  "{                                                                 \n"
  "    vec3 I = gg_Pwld - g_eyeWld;                                  \n"
  "    vec3 In = normalize( I );                                     \n"
  "    vec3 Nn = normalize( gg_Nwld );                               \n"
  "                                                                  \n"
  "    vec3 ToKey = normalize( g_toKey );                            \n"
  "    vec3 ToFill = normalize( g_toFill );                          \n"
  "                                                                  \n"
  "    vec3 Cdiff = g_diffColor *                                    \n"
  "       ( ( g_keyColor * kDiffuse( Nn, ToKey ) ) +                 \n"
  "         ( g_fillColor * kDiffuse( Nn, ToFill ) ) );              \n"
  "    vec3 Cspec = g_specColor *                                    \n"
  "       ( ( g_keyColor * kSpecular( In, Nn, ToKey, g_specExponent ) ) +      "
  "  \n"
  "         ( g_fillColor * kSpecular( In, Nn, ToFill, g_specExponent ) ) );   "
  "  \n"
  "                                                                  \n"
  "    vec3 finalCol = gammaCorrect( Cdiff + Cspec, 2.2 );           \n"
  "    //finalCol = gammaCorrect( vec3(0.5,0.5,0.5) + 0.5*vec3(dot(In,Nn)), "
  "2.2 );           \n"
  "    //finalCol = Cspec; \n"
  "    g_fragmentColor = vec4( finalCol, 1.0 );                      \n"
  "}                                                                 \n"
  "                                                                  \n";

//-*****************************************************************************
static const char* g_constantRedFragmentBase =
  "out vec4 g_fragmentColor;                                         \n"
  "void main() { g_fragmentColor = vec4( 1, 0, 0, 1 ); } \n";

//-*****************************************************************************
static const char* g_constantWhiteFragmentBase =
  "out vec4 g_fragmentColor;                                         \n"
  "void main() { g_fragmentColor = vec4( 1, 1, 1, 1 ); } \n";

//-*****************************************************************************
std::string StdShaderHeader() { return std::string(g_shaderHeader); }

//-*****************************************************************************
std::string StdMatrices() { return std::string(g_stdMatrices); }

//-*****************************************************************************
std::string StdTransformFunctions() {
  return std::string(g_transformFunctions);
}

//-*****************************************************************************
std::string StdSpecDiffuseGammaFunctions() {
  return std::string(g_specDiffuseGamma);
}

//-*****************************************************************************
std::string SimpleVertexShader() {
  return std::string(g_shaderHeader) + StdMatrices() + StdTransformFunctions() +
         std::string(g_simpleVertexShaderBase);
}

//-*****************************************************************************
std::string SimplePointsGeometryShader() {
  return std::string(g_shaderHeader) + StdMatrices() + StdTransformFunctions() +
         std::string(g_simplePointsGeometryShaderBase);
}

//-*****************************************************************************
std::string SimpleTrianglesGeometryShader() {
  return std::string(g_shaderHeader) + StdMatrices() + StdTransformFunctions() +
         std::string(g_simpleTrianglesGeometryShaderBase);
}

//-*****************************************************************************
std::string SimpleTrianglesWireframeGeometryShader() {
  return std::string(g_shaderHeader) + StdMatrices() + StdTransformFunctions() +
         std::string(g_simpleTrianglesWireframeGeometryShaderBase);
}

//-*****************************************************************************
std::string KeyFillFragmentShader() {
  return std::string(g_shaderHeader) + StdMatrices() + StdTransformFunctions() +
         StdSpecDiffuseGammaFunctions() +
         std::string(g_keyFillFragmentShaderBase);
}

//-*****************************************************************************
std::string ConstantRedFragmentShader() {
  return std::string(g_shaderHeader) + std::string(g_constantRedFragmentBase);
}

//-*****************************************************************************
std::string ConstantWhiteFragmentShader() {
  return std::string(g_shaderHeader) + std::string(g_constantWhiteFragmentBase);
}

//-*****************************************************************************
static M44d nmatrix(const M44d& m) {
  V3d s;
  V3d h;
  Imath::Eulerd r;
  V3d t;
  bool status = Imath::extractSHRT(m, s, h, r, t);
  EWAV_ASSERT(status, "Should have good matrix");

  M44d S;
  S.setScale(s);
  M44d H;
  H.setShear(h);
  M44d R = r.toMatrix44();

  M44d nm = S * H * R;
  nm.gjInvert();
  nm.transpose();
  return nm;
}

//-*****************************************************************************
//  "uniform mat4 projection_matrix;                            \n"
//  "uniform mat4 world_to_rhc_matrix;                          \n"
//  "uniform mat4 obj_to_world_matrix;                          \n"
//  "uniform mat4 world_to_obj_matrix;                          \n"
//  "uniform mat4 rhc_to_world_matrix;                          \n"
//  "uniform vec3 g_eyeWld;                                     \n";
void SetStdMatrices(Program& o_program, const GLCamera& i_cam,
                    const M44d& i_objectToWorld) {
  M44d proj           = i_cam.projectionMatrix();
  M44d world_to_rhc   = i_cam.modelViewMatrix();
  M44d world_to_rhc_n = nmatrix(world_to_rhc);

  M44d rhc_to_world   = world_to_rhc.gjInverse();
  M44d rhc_to_world_n = nmatrix(rhc_to_world);

  M44d world_to_obj   = i_objectToWorld.gjInverse();
  M44d world_to_obj_n = nmatrix(world_to_obj);

  M44d obj_to_world_n = nmatrix(i_objectToWorld);

  o_program(Uniform("projection_matrix", proj));
  o_program(Uniform("world_to_rhc_matrix", world_to_rhc));
  o_program(Uniform("world_to_rhc_nmatrix", world_to_rhc_n));

  o_program(Uniform("rhc_to_world_matrix", rhc_to_world));
  o_program(Uniform("rhc_to_world_nmatrix", rhc_to_world_n));

  o_program(Uniform("world_to_obj_matrix", world_to_obj));
  o_program(Uniform("world_to_obj_nmatrix", world_to_obj_n));

  o_program(Uniform("obj_to_world_matrix", i_objectToWorld));
  o_program(Uniform("obj_to_world_nmatrix", obj_to_world_n));

  // Eye location.
  const V3d& eye = i_cam.translation();
  o_program(Uniform("g_eyeWld", eye /*, Uniform::kRequireWarning*/));
}

//-*****************************************************************************
// "uniform vec3 g_toKey;                                             \n"
//    "uniform vec3 g_keyColor;                                          \n"
//    "uniform vec3 g_toFill;                                            \n"
//    "uniform vec3 g_fillColor;                                         \n"
void SetKeyFillLights(Program& o_program, const V3f& i_toKey,
                      const V3f& i_keyColor, const V3f& i_toFill,
                      const V3f& i_fillColor) {
  o_program(Uniform("g_toKey", i_toKey));
  o_program(Uniform("g_keyColor", i_keyColor));
  o_program(Uniform("g_toFill", i_toFill));
  o_program(Uniform("g_fillColor", i_fillColor));
}

//-*****************************************************************************
void SetStdMaterial(Program& o_program, const V3f& i_diffColor,
                    const V3f& i_specColor, float i_specExponent) {
  o_program(Uniform("g_diffColor", i_diffColor));
  o_program(Uniform("g_specColor", i_specColor));
  o_program(Uniform("g_specExponent", i_specExponent));
}

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves
