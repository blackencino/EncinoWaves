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

#include <Util/All.h>

#include <iostream>
#include <typeinfo>

namespace EncinoWaves {
namespace Util {

//-*****************************************************************************
template <typename T>
void testWrapT(T x, T lb, T ub, T expected, T tol) {
  T k = wrap<T>(x, lb, ub);
  std::cout << "wrap<" << typeid(T).name() << ">( " << x << ", " << lb << ", "
            << ub << " ) = " << k << ", expecting: " << expected << std::endl;
  EWAV_ASSERT(std::abs(k - expected) <= tol, "test wrap");
}

//-*****************************************************************************
void testWrap() {
  testWrapT<int>(15, 7, 12, 9, 0);
  testWrapT<int>(6, -4, 3, -2, 0);
  testWrapT<int>(-19, 9, 13, 11, 0);

  // Try floats.
  testWrapT<float>(-741.325, 1.4151, 19.7333, 9.72113, 0.0001);

  float e   = 32.4;
  int block = -844;
  float lb  = 1.8;
  float ub  = 66.7113;

  float x = lb + ((ub - lb) * float(block)) + (e - lb);
  testWrapT<float>(x, lb, ub, e, 0.01);
}

}  // namespace Util
}  // namespace EncinoWaves

//-*****************************************************************************
int main(int argc, char* argv[]) {
  EncinoWaves::Util::testWrap();
  return 0;
}
