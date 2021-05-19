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

#ifndef _EncinoWaves_Util_Foundation_h_
#define _EncinoWaves_Util_Foundation_h_

// #include <sys/time.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <OpenEXR/ImathMath.h>
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/half.h>

#include <Alembic/AbcGeom/All.h>
#include <Alembic/Util/All.h>

namespace EncinoWaves {
namespace Util {

// Bring in Alembic types!
namespace AbcG = Alembic::AbcGeom;

using AbcG::chrono_t;
using AbcG::uint8_t;
using AbcG::int8_t;
using AbcG::uint16_t;
using AbcG::int16_t;
using AbcG::uint32_t;
using AbcG::int32_t;
using AbcG::uint64_t;
using AbcG::int64_t;
using AbcG::float16_t;
using AbcG::float32_t;
using AbcG::float64_t;
using AbcG::bool_t;
using AbcG::byte_t;
using AbcG::index_t;

using AbcG::V2s;
using AbcG::V2i;
using AbcG::V2f;
using AbcG::V2d;

using AbcG::V3s;
using AbcG::V3i;
using AbcG::V3f;
using AbcG::V3d;

using Imath::V4s;
using Imath::V4i;
using Imath::V4f;
using Imath::V4d;

using AbcG::Box2s;
using AbcG::Box2i;
using AbcG::Box2f;
using AbcG::Box2d;

using AbcG::Box3s;
using AbcG::Box3i;
using AbcG::Box3f;
using AbcG::Box3d;

using AbcG::M33f;
using AbcG::M33d;
using AbcG::M44f;
using AbcG::M44d;

using AbcG::Quatf;
using AbcG::Quatd;

using AbcG::C3h;
using AbcG::C3f;
using AbcG::C3c;

using AbcG::C4h;
using AbcG::C4f;
using AbcG::C4c;

using AbcG::N3f;
using AbcG::N3d;

}  // namespace Util
}  // namespace EncinoWaves

#endif
