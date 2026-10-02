"""Compile the maintained scalar headers without the obsolete viewer dependencies."""
from dataclasses import replace
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import math
import numpy as np
import pytest
from scipy.special import expit
from encino_waves.model import Wave_parameters, dispersion_at, spectrum_at, spreading_at, peak_omega


@pytest.fixture(scope="module")
def cpp_values(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("cpp_numerics") / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root / "src"),
                    str(root / "tests/reference/current_scalar_oracle.cpp"),
                    "-o", str(executable)], check=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    values = np.loadtxt(StringIO(result.stdout))
    assert np.isfinite(values).all()
    return values


def test_cpp_dispersion_and_spectra_match_reference(cpp_values):
    base = Wave_parameters(depth=20)
    for bits, kind, model, a, b, _, x, y in cpp_values[cpp_values[:, 1] < 2]:
        tolerance = 3e-5 if bits == 32 else 2e-12
        if kind == 0:
            p = replace(base, dispersion=("deep", "finite", "capillary")[int(model)])
            expected = dispersion_at(p, a)
            np.testing.assert_allclose((x, y), expected, rtol=tolerance, atol=1e-12)
        else:
            p = replace(base, spectrum=("pm", "jonswap", "tma")[int(model)], gamma=b)
            np.testing.assert_allclose(x, spectrum_at(p, a), rtol=tolerance, atol=1e-12)
            if model == 2:
                np.testing.assert_allclose(y, expit(3.6*(a*math.sqrt(p.depth/p.gravity)-1.125)), rtol=tolerance)


def test_cpp_spreading_matches_paper_and_is_nonnegative(cpp_values):
    models = ("hasselmann", "mitsuyasu", "donelan_banner", "cosine_squared")
    for bits, kind, model, swell, ratio, angle, value, _ in cpp_values[cpp_values[:, 1] == 2]:
        assert value >= 0
        p = Wave_parameters(spreading=models[int(model)], swell=float(swell))
        expected = spreading_at(p, peak_omega(p)*ratio, angle*math.pi)
        # The exact DC limit has shape=0, whereas the array reference floors
        # omega at 1e-12. Their 0^0 endpoints differ on a zero-measure set.
        if ratio == 0:
            continue
        np.testing.assert_allclose(value, expected, atol=4e-6, rtol=5e-4 if bits == 32 else 1e-10)


def test_cpp_angular_energy_is_one(cpp_values):
    integral = cpp_values[cpp_values[:, 1] == 3, 6]
    np.testing.assert_allclose(integral, 1, atol=1e-5, rtol=0)
