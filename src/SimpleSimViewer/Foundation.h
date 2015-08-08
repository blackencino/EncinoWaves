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

#ifndef _EncinoWaves_SimpleSimViewer_Foundation_h_
#define _EncinoWaves_SimpleSimViewer_Foundation_h_

#include <GeepGLFW/All.h>
#include <Util/All.h>
#include <Alembic/Util/All.h>

#include <ImathBox.h>
#include <ImathBoxAlgo.h>
#include <ImathColor.h>
#include <ImathFrustum.h>
#include <ImathFun.h>
#include <ImathLine.h>
#include <ImathMath.h>
#include <ImathMatrix.h>
#include <ImathMatrixAlgo.h>
#include <ImathQuat.h>
#include <ImathVec.h>

#include <boost/format.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
#ifdef PLATFORM_DARWIN

#define OSX_GLFW_VIEWPORT_BUG 1

#else

#define OSX_GLFW_VIEWPORT_BUG 0

#endif

using namespace EncinoWaves::Util;
using namespace EncinoWaves::GeepGLFW;

typedef Imath::Vec3<unsigned int> V3ui;

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
