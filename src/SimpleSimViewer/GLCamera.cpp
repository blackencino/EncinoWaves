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

#include "GLCamera.h"

#define Y_UP 0
#define Z_UP 1

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
GLCamera::GLCamera()
  : m_rotation(0.0, 0.0, 0.0)
  , m_scale(1.0, 1.0, 1.0)
  , m_translation(0.0, 0.0, 0.0)
  , m_centerOfInterest(15.0)
  , m_fovy(45.0)
  , m_clip(0.0, 1.0)
  , m_size(100, 100)
  , m_aspect(1.0) {
  // Nothing
}

//-*****************************************************************************
static inline void rotateVectorYup(double rx, double ry, V3d &v) {
  rx                = radians(rx);
  const double sinX = sin(rx);
  const double cosX = cos(rx);

  const V3d t(v.x, (v.y * cosX) - (v.z * sinX), (v.y * sinX) + (v.z * cosX));

  ry                = radians(ry);
  const double sinY = sin(ry);
  const double cosY = cos(ry);

  v.x = (t.x * cosY) + (t.z * sinY);
  v.y = t.y;
  v.z = (t.x * -sinY) + (t.z * cosY);
}

//-*****************************************************************************
static inline void rotateVectorZup(double rx, double rz, V3d &v) {
  rx                = radians(rx);
  const double sinX = sin(rx);
  const double cosX = cos(rx);

  const V3d t(v.x, (v.y * cosX) - (v.z * sinX), (v.y * sinX) + (v.z * cosX));

  rz                = radians(rz);
  const double sinZ = sin(rz);
  const double cosZ = cos(rz);

  v.x = (t.x * cosZ) - (t.y * sinZ);
  v.y = (t.x * sinZ) + (t.y * cosZ);
  v.z = t.z;
}

//-*****************************************************************************
void GLCamera::frame(const Box3d &bounds) {
  double r     = 0.5 * bounds.size().length();
  double fovyr = radians(m_fovy);

#if Y_UP

  double g = (1.1f * r) / sinf(fovyr * 0.5f);
  lookAt(bounds.center() + V3d(0, 0, g), bounds.center());

#else

  double g = (1.1 * r) / sin(fovyr * 0.5);
  lookAt(bounds.center() + V3d(1.0 * g, -1.0 * g, 2.0 * r), bounds.center());

// std::cout << "Bounds.center(): " << bounds.center() << std::endl
//          << "G: " << g << std::endl
//          << "Eye point: " << ( bounds.center() + V3d( 0, g, 0 ) )
//          << std::endl;

#endif  // Y_UP
}

//-*****************************************************************************
void GLCamera::autoSetClippingPlanes(const Box3d &bounds) {
  V3d bsize   = bounds.size();
  double tiny = 0.0001 * std::min(std::min(bsize.x, bsize.y), bsize.z);

#if Y_UP

  const double rotX = m_rotation.x;
  const double rotY = m_rotation.y;
  const V3d &eye    = m_translation;
  double clipNear   = FLT_MAX;
  double clipFar    = FLT_MIN;

  V3d v(0.0, 0.0, -m_centerOfInterest);
  rotateVectorYup(rotX, rotY, v);
  const V3d view = eye + v;
  v.normalize();

  V3d points[8];

  points[0] = V3d(bounds.min.x, bounds.min.y, bounds.min.z);
  points[1] = V3d(bounds.min.x, bounds.min.y, bounds.max.z);
  points[2] = V3d(bounds.min.x, bounds.max.y, bounds.min.z);
  points[3] = V3d(bounds.min.x, bounds.max.y, bounds.max.z);
  points[4] = V3d(bounds.max.x, bounds.min.y, bounds.min.z);
  points[5] = V3d(bounds.max.x, bounds.min.y, bounds.max.z);
  points[6] = V3d(bounds.max.x, bounds.max.y, bounds.min.z);
  points[7] = V3d(bounds.max.x, bounds.max.y, bounds.max.z);

  for (int p = 0; p < 8; ++p) {
    V3d dp      = points[p] - eye;
    double proj = dp.dot(v);
    clipNear    = std::min(proj, clipNear);
    clipFar     = std::max(proj, clipFar);
  }

  clipNear -= tiny;
  clipFar += tiny;
  clipNear = clamp(clipNear, tiny, 1.0e30);
  clipFar  = clamp(clipFar, tiny, 1.0e30);

  assert(clipFar > clipNear);

  m_clip[0] = clipNear;
  m_clip[1] = clipFar;

#else

  const double rotX = m_rotation.x;
  const double rotZ = m_rotation.z;
  const V3d &eye    = m_translation;
  double clipNear   = FLT_MAX;
  double clipFar    = FLT_MIN;

  V3d v(0.0, m_centerOfInterest, 0.0);
  rotateVectorZup(rotX, rotZ, v);
  const V3d view = eye + v;
  v.normalize();

  V3d points[8];

  points[0] = V3d(bounds.min.x, bounds.min.y, bounds.min.z);
  points[1] = V3d(bounds.min.x, bounds.min.y, bounds.max.z);
  points[2] = V3d(bounds.min.x, bounds.max.y, bounds.min.z);
  points[3] = V3d(bounds.min.x, bounds.max.y, bounds.max.z);
  points[4] = V3d(bounds.max.x, bounds.min.y, bounds.min.z);
  points[5] = V3d(bounds.max.x, bounds.min.y, bounds.max.z);
  points[6] = V3d(bounds.max.x, bounds.max.y, bounds.min.z);
  points[7] = V3d(bounds.max.x, bounds.max.y, bounds.max.z);

  for (int p = 0; p < 8; ++p) {
    V3d dp      = points[p] - eye;
    double proj = dp.dot(v);
    clipNear    = std::min(proj, clipNear);
    clipFar     = std::max(proj, clipFar);
  }

  clipNear -= tiny;
  clipFar += tiny;
  clipNear /= 2.0f;
  clipFar *= 2.0f;
  clipNear = clamp(clipNear, tiny, 1.0e30);
  clipFar  = clamp(clipFar, tiny, 1.0e30);

  assert(clipFar > clipNear);

  m_clip[0] = clipNear;
  m_clip[1] = clipFar;

#endif
}

//-*****************************************************************************
void GLCamera::lookAt(const V3d &eye, const V3d &at) {
#if Y_UP

  m_translation = eye;

  const V3d dt = at - eye;

  const double xzLen = sqrt((dt.x * dt.x) + (dt.z * dt.z));

  m_rotation.x = degrees(atan2(dt.y, xzLen));

  m_rotation.y = degrees(atan2(dt.x, -dt.z));

  m_rotation.z = 0.0;

  m_centerOfInterest = dt.length();

#else

  m_translation = eye;

  const V3d dt = at - eye;

  const double xyLen = sqrt((dt.x * dt.x) + (dt.y * dt.y));

  m_rotation.x = degrees(atan2(dt.z, xyLen));

  m_rotation.y = 0.0;

  m_rotation.z = degrees(atan2(-dt.x, dt.y));

  m_centerOfInterest = dt.length();

#endif
}

//-*****************************************************************************
void GLCamera::apply() const {
#if 0
#if Y_UP

    glViewport( 0, 0, ( GLsizei )m_size[0], ( GLsizei )m_size[1] );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( m_fovy,
                    ( ( GLdouble )m_size[0] ) /
                    ( ( GLdouble )m_size[1] ),
                    m_clip[0],
                    m_clip[1] );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    ::glScaled( 1.0 / m_scale[0], 1.0 / m_scale[1], 1.0 / m_scale[2] );
    ::glRotated( -m_rotation[2], 0.0, 0.0, 1.0 );
    ::glRotated( -m_rotation[0], 1.0, 0.0, 0.0 );
    ::glRotated( -m_rotation[1], 0.0, 1.0, 0.0 );
    ::glTranslated( -m_translation[0], -m_translation[1], -m_translation[2] );

#else

    glViewport( 0, 0, ( GLsizei )m_size[0], ( GLsizei )m_size[1] );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( m_fovy,
                    ( ( GLdouble )m_size[0] ) /
                    ( ( GLdouble )m_size[1] ),
                    m_clip[0],
                    m_clip[1] );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    // Switch to Y-up.
    ::glRotated( -90.0, 1.0, 0.0, 0.0 );

    // Here, I have put the world so that it lies down the positive
    // y axis.  Now, above, I must rotate the positive y axis to negative z.

    // Transform
    ::glScaled( 1.0 / m_scale[0], 1.0 / m_scale[1], 1.0 / m_scale[2] );
    ::glRotated( -m_rotation[1], 0.0, 1.0, 0.0 );
    ::glRotated( -m_rotation[0], 1.0, 0.0, 0.0 );
    ::glRotated( -m_rotation[2], 0.0, 0.0, 1.0 );
    ::glTranslated( -m_translation[0], -m_translation[1], -m_translation[2] );
#endif
#endif
}

//-*****************************************************************************
M44d GLCamera::modelViewMatrix() const {
#if Y_UP

  M44d m;
  M44d tmp;
  m.makeIdentity();

  tmp.setScale(V3d(1.0 / m_scale[0], 1.0 / m_scale[1], 1.0 / m_scale[2]));
  m = m * tmp;

  tmp.setAxisAngle(V3d(0.0, 0.0, 1.0), radians(-m_rotation[2]));
  m = m * tmp;
  tmp.setAxisAngle(V3d(1.0, 0.0, 0.0), radians(-m_rotation[0]));
  m = m * tmp;
  tmp.setAxisAngle(V3d(0.0, 1.0, 0.0), radians(-m_rotation[1]));
  m = m * tmp;

  tmp.setTranslation(
    V3d(-m_translation[0], -m_translation[1], -m_translation[2]));
  m = m * tmp;

  return m;

#else

  M44d ZupToYup;
  ZupToYup.setAxisAngle(V3d(1.0, 0.0, 0.0), radians(-90.0));

  M44d UnScale;
  UnScale.setScale(V3d(1.0 / m_scale[0], 1.0 / m_scale[1], 1.0 / m_scale[2]));

  M44d UnRotY;
  UnRotY.setAxisAngle(V3d(0.0, 1.0, 0.0), radians(-m_rotation[1]));

  M44d UnRotX;
  UnRotX.setAxisAngle(V3d(1.0, 0.0, 0.0), radians(-m_rotation[0]));

  M44d UnRotZ;
  UnRotZ.setAxisAngle(V3d(0.0, 0.0, 1.0), radians(-m_rotation[2]));

  M44d UnTranslate;
  UnTranslate.setTranslation(
    V3d(-m_translation[0], -m_translation[1], -m_translation[2]));

  M44d m = UnTranslate * UnRotZ * UnRotX * UnRotY * UnScale * ZupToYup;

  return m;

#endif
}

//-*****************************************************************************
M44d GLCamera::projectionMatrix() const {
#if 1
  Imath::Frustum<double> F;
  F.set(m_clip[0], m_clip[1], 0.0, radians(m_fovy),
        double(m_size[0]) / double(m_size[1]));

  return F.projectionMatrix();
#else

  double znear  = m_clip[0];
  double zfar   = m_clip[1];
  double aspect = double(m_size[0]) / double(m_size[1]);

  double ymax = znear * std::tan(radians(m_fovy) / 2.0);
  double ymin = -ymax;
  double xmax = ymax * aspect;
  double xmin = ymin * aspect;

  double width  = xmax - xmin;
  double height = ymax - ymin;

  double depth = zfar - znear;
  double q     = -(zfar + znear) / depth;
  double qn    = -2.0 * (zfar * znear) / depth;

  double w = 2.0 * znear / width;
  w        = w / aspect;
  double h = 2.0 * znear / height;

  M44d ret;
  double *m = reinterpret_cast<double *>(&(ret[0][0]));

  m[0] = w;
  m[1] = 0.0;
  m[2] = 0.0;
  m[3] = 0.0;

  m[4] = 0.0;
  m[5] = h;
  m[6] = 0.0;
  m[7] = 0.0;

  m[8]  = 0.0;
  m[9]  = 0.0;
  m[10] = q;
  m[11] = -1;

  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = qn;
  m[15] = 0.0;

  return ret;
#endif
}

//-*****************************************************************************
void GLCamera::track(const V2d &point) {
#if Y_UP

  // INIT
  const double rotX = m_rotation.x;
  const double rotY = m_rotation.y;

  V3d dS(1.0, 0.0, 0.0);
  rotateVectorYup(rotX, rotY, dS);

  V3d dT(0.0, 1.0, 0.0);
  rotateVectorYup(rotX, rotY, dT);

  double multS       = 2.0 * m_centerOfInterest * tanf(radians(fovy()) / 2.0);
  const double multT = multS / double(height());
  multS /= double(width());

  // TRACK
  const double s = -multS * point.x;
  const double t = multT * point.y;

  // ALTER
  setTranslation((m_translation + (s * dS) + (t * dT)));

#else

  // INIT
  const double rotX = m_rotation.x;
  const double rotZ = m_rotation.z;

  V3d dS(1.0, 0.0, 0.0);
  rotateVectorZup(rotX, rotZ, dS);

  V3d dT(0.0, 0.0, 1.0);
  rotateVectorZup(rotX, rotZ, dT);

  double multS       = 2.0 * m_centerOfInterest * tanf(radians(fovy()) / 2.0);
  const double multT = multS / double(height());
  multS /= double(width());

  // TRACK
  const double s = -multS * point.x;
  const double t = multT * point.y;

  // ALTER
  setTranslation((m_translation + (s * dS) + (t * dT)));

#endif
}

//-*****************************************************************************
void GLCamera::dolly(const V2d &point, double dollySpeed) {
#if Y_UP

  // INIT
  const double rotX = m_rotation.x;
  const double rotY = m_rotation.y;
  const V3d &eye    = m_translation;

  V3d v(0.0, 0.0, -m_centerOfInterest);
  rotateVectorYup(rotX, rotY, v);
  const V3d view = eye + v;
  v.normalize();

  // DOLLY
  const double t = point.x / double(width());

  // Magic dolly function
  double dollyBy = 1.0 - expf(-dollySpeed * t);

  assert(std::abs(dollyBy) < 1.0);
  dollyBy *= m_centerOfInterest;
  const V3d newEye = eye + (dollyBy * v);

  // ALTER
  setTranslation(newEye);
  v                  = newEye - view;
  m_centerOfInterest = v.length();

#else

  // INIT
  const double rotX = m_rotation.x;
  const double rotZ = m_rotation.z;
  const V3d &eye    = m_translation;

  V3d v(0.0, m_centerOfInterest, 0.0);
  rotateVectorZup(rotX, rotZ, v);
  const V3d view = eye + v;
  v.normalize();

  // DOLLY
  const double t = point.x / double(width());

  // Magic dolly function
  double dollyBy = 1.0 - expf(-dollySpeed * t);

  assert(std::abs(dollyBy) < 1.0);
  dollyBy *= m_centerOfInterest;
  const V3d newEye = eye + (dollyBy * v);

  // ALTER
  setTranslation(newEye);
  v                  = newEye - view;
  m_centerOfInterest = v.length();

#endif
}

//-*****************************************************************************
void GLCamera::rotate(const V2d &point, double rotateSpeed) {
#if Y_UP

  // INIT
  double rotX       = m_rotation.x;
  double rotY       = m_rotation.y;
  const double rotZ = m_rotation.z;
  V3d eye           = m_translation;

  V3d v(0.0, 0.0, -m_centerOfInterest);
  rotateVectorYup(rotX, rotY, v);

  const V3d view = eye + v;

  // ROTATE
  rotY += -rotateSpeed * (point.x / double(width()));
  rotX += -rotateSpeed * (point.y / double(height()));

  v[0] = 0.0;
  v[1] = 0.0;
  v[2] = m_centerOfInterest;
  rotateVectorYup(rotX, rotY, v);

  const V3d newEye = view + v;

  // ALTER
  setTranslation(view + v);
  setRotation(V3d(rotX, rotY, rotZ));

#else

  // INIT
  double rotX       = m_rotation.x;
  const double rotY = m_rotation.y;
  double rotZ       = m_rotation.z;
  V3d eye           = m_translation;

  V3d v(0.0, m_centerOfInterest, 0.0);
  rotateVectorZup(rotX, rotZ, v);

  const V3d view = eye + v;

  // ROTATE
  rotZ += rotateSpeed * (-point.x / double(width()));
  rotX += rotateSpeed * (-point.y / double(height()));

  v[0] = 0.0;
  v[1] = -m_centerOfInterest;
  v[2] = 0.0;
  rotateVectorZup(rotX, rotZ, v);

  const V3d newEye = view + v;

  // ALTER
  setTranslation(view + v);
  setRotation(V3d(rotX, rotY, rotZ));

#endif
}

//-*****************************************************************************
std::string GLCamera::RIB() const {
#if Y_UP

  // Then transpose and print.
  return (boost::format(
            "Format %d %d 1\n"
            "Clipping %f %f\n"
            "Projection \"perspective\" \"fov\" [%f]\n"
            "Scale 1 1 -1\n"
            "Scale %f %f %f\n"
            "Rotate %f 0 0 1\n"
            "Rotate %f 1 0 0\n"
            "Rotate %f 0 1 0\n"
            "Translate %f %f %f\n") %
          (int)m_size[0] % (int)m_size[1] % (float)m_clip[0] %
          (float)m_clip[1] % (float)m_fovy % (float)(1.0 / m_scale[0]) %
          (float)(1.0 / m_scale[1]) % (float)(1.0 / m_scale[2]) %
          (float)(-m_rotation[2]) % (float)(-m_rotation[0]) %
          (float)(-m_rotation[1]) % (float)(-m_translation[0]) %
          (float)(-m_translation[1]) % (float)(-m_translation[2]))
    .str();

#else

  // Then transpose and print.
  return (boost::format(
            "Format %d %d 1\n"
            "Clipping %f %f\n"
            "Projection \"perspective\" \"fov\" [%f]\n"
            "Scale 1 1 -1\n"
            "Rotate -90 1 0 0\n"
            "Scale %f %f %f\n"
            "Rotate %f 0 1 0\n"
            "Rotate %f 1 0 0\n"
            "Rotate %f 0 0 1\n"
            "Translate %f %f %f\n") %
          (int)m_size[0] % (int)m_size[1] % (float)m_clip[0] %
          (float)m_clip[1] % (float)m_fovy % (float)(1.0 / m_scale[0]) %
          (float)(1.0 / m_scale[1]) % (float)(1.0 / m_scale[2]) %
          (float)(-m_rotation[1]) % (float)(-m_rotation[0]) %
          (float)(-m_rotation[2]) % (float)(-m_translation[0]) %
          (float)(-m_translation[1]) % (float)(-m_translation[2]))
    .str();

#endif
}

//-*****************************************************************************
Imath::Line3d GLCamera::getRayThroughRasterPoint(const V2d &pt_raster) const {
  M44d rhc_to_world = modelViewMatrix().inverse();
  Imath::Frustum<double> F;
  F.set(m_clip[0], m_clip[1], 0.0, radians(m_fovy),
        double(m_size[0]) / double(m_size[1]));

  V2d pt_screen(
    2.0 * (pt_raster.x / double(m_size.x)) - 1.0,
    2.0 * ((double(m_size.y) - pt_raster.y) / double(m_size.y)) - 1.0);

  Imath::Line3d ray = F.projectScreenToRay(pt_screen);
  ray.pos           = m_translation;
  V3d new_dir;
  rhc_to_world.multDirMatrix(ray.dir, new_dir);
  new_dir.normalize();
  ray.dir = new_dir;
  return ray;
}

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves
