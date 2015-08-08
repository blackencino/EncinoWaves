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

#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>

#include <random>
#include <cstdint>

typedef uint64_t seed_type;
typedef std::linear_congruential_engine<uint64_t, uint64_t(0xDEECE66DUL) |
                                                    (uint64_t(0x5) << 32),
                                        0xB, uint64_t(1) << 48> Rand48_Engine;

//-*****************************************************************************
// Let's see whether we can use fancy cmath stuff.
void testFancyCmathStuff() {
  // Gamma.
  std::cout << "Gamma of 0.5 = " << std::tgamma(0.5) << std::endl;

  // Lround.
  std::cout << "Long round of 191381.1356: " << std::lround(191381.1356)
            << std::endl;

  // Erf & Erfc
  std::cout << "Erf of 18.881: " << std::erf(18.881) << std::endl
            << "Erfc of 18.881: " << std::erfc(18.881) << std::endl;

  // Random stuff
  Rand48_Engine rnd;
  rnd.seed(12345);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::cout << "First draw from Rand48: " << dist(rnd) << std::endl;
}

//-*****************************************************************************
template <typename ITER, typename F>
void applyToAll(ITER i_begin, ITER i_end, F i_func) {
  int N    = i_end - i_begin;
  int* ptr = &(*i_begin);
  for (int i = 0; i < N; ++i) {
    i_func(*(ptr + i), i);
  }
}

//-*****************************************************************************
struct Base {
  Base()
    : a(0.0f) {}
  float a;

  float A() const { return a; }
};

//-*****************************************************************************
template <typename T>
struct Derived : public Base {
  Derived()
    : Base()
    , b(T(0)) {}

  T b;
  // using Base::A;
  T B() const { return b; }
  T C() const { return b + A(); }
};

//-*****************************************************************************
template <typename T>
struct Derived2 : public Derived<T> {
  Derived2()
    : Derived<T>()
    , c(T(0)) {}

  T c;

  using Derived<T>::C;

  T C2() const { return c + C(); }
};

//-*****************************************************************************
int main(int argc, char* argv[]) {
  std::vector<int> v(17);

  applyToAll(v.begin(), v.end(), [](int& i, int j) { i = j; });
  std::for_each(v.begin(), v.end(), [](int i) { std::cout << i << std::endl; });

  testFancyCmathStuff();

  Derived2<double> D;
  std::cout << "A = " << D.A() << std::endl << "C = " << D.C() << std::endl;

  return 0;
}
