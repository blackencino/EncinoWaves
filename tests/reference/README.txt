original_scalar_fields.txt was produced on Apple silicon, clang 17 / libc++, by
compiling the actual 2015 headers. No ocean equation is reimplemented in the
oracle; the adapters replace only the obsolete dependency plumbing.

Regenerate (developer check only; users do not need a C++ compiler):
    c++ -std=c++17 -O2 tests/reference/original_oracle.cpp -o /tmp/encino_oracle
    /tmp/encino_oracle > tests/reference/original_scalar_fields.txt

First line: the original seeded JONSWAP gamma.
Remaining lines: h_positive.real, h_positive.imag, h_negative.real,
h_negative.imag, omega, kx, ky, domega_dk, spectrum, direction_positive,
direction_negative; row-major, a 16 x 9 real-FFT half-plane.

The tests independently check the scalar equations, then replay this original
random realization through the new GPU propagation and compare displacement,
crest and normals with a double-precision FFTW reference. This does not assert
that the new portable RNG reproduces a platform-specific std distribution.
