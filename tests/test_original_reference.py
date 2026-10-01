"""Conformance against values evaluated by the original, unmodified headers."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from encino_waves.model import (Wave_parameters, make_initial_state, evaluate,
                               dispersion_at, spectrum_at, spreading_at)


def original_values():
    path = Path(__file__).parent / "reference/original_scalar_fields.txt"
    gamma = float(path.read_text().splitlines()[0])
    values = np.loadtxt(path, skiprows=1).reshape(16, 9, 11)
    p = Wave_parameters(resolution=16, domain=100, wind_speed=17, fetch_km=300,
                        depth=20, swell=.35, spreading="hasselmann",
                        convention="legacy_2015", gamma=gamma)
    return p, values


def test_scalar_equations_against_original_headers():
    p, values = original_values()
    kx, ky = values[..., 5], values[..., 6]
    k = np.hypot(kx, ky)
    omega, derivative = dispersion_at(p, k)
    theta = np.arctan2(-ky, kx)
    opposite = np.arctan2(ky, -kx)
    nonzero = k > 0
    for actual, expected in (
        (omega, values[..., 4]), (derivative, values[..., 7]),
        (spectrum_at(p, omega), values[..., 8]),
        (spreading_at(p, omega, theta), values[..., 9]),
        (spreading_at(p, omega, opposite), values[..., 10]),
    ):
        np.testing.assert_allclose(actual[nonzero], expected[nonzero], rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_original_random_realization_against_fftw(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal not visible")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not visible")
    fftw = pytest.importorskip("pyfftw.interfaces.numpy_fft")
    p, values = original_values()
    state = make_initial_state(p, device)
    pos = (values[..., 0] + 1j*values[..., 1]).astype(np.complex64)
    neg = (values[..., 2] + 1j*values[..., 3]).astype(np.complex64)
    omega = values[..., 4].astype(np.float32)
    state = replace(state, h_positive=torch.from_numpy(pos).to(device),
                    h_negative=torch.from_numpy(neg).to(device),
                    omega=torch.from_numpy(omega).to(device))
    for time in (0.0, 10.0, 120.0):
        phase = omega.astype(np.float64)*time
        h = pos*np.exp(-1j*phase) + neg*np.exp(1j*phase)
        fields = fftw.irfft2(state.multipliers.cpu().numpy()*h[None], s=(16, 16), norm="forward")
        height, dx, dy, dxx, dyy, dxy = fields
        jxx, jyy, jxy = 1-p.pinch*dxx, 1-p.pinch*dyy, -p.pinch*dxy
        crest = -.5*(jxx+jyy-np.sqrt((jxx-jyy)**2+4*jxy**2))
        expected = np.stack((-p.pinch*dx, -p.pinch*dy, height, crest), axis=-1)
        tangent_x = np.roll(expected[..., :3], -1, 1)-np.roll(expected[..., :3], 1, 1)
        tangent_y = np.roll(expected[..., :3], -1, 0)-np.roll(expected[..., :3], 1, 0)
        tangent_x[..., 0] += 2*p.domain/p.resolution
        tangent_y[..., 1] += 2*p.domain/p.resolution
        normal = np.cross(tangent_x, tangent_y)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        frame = evaluate(state, time)
        np.testing.assert_allclose(frame.displacement.cpu().numpy(), expected, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(frame.normal[..., :3].cpu().numpy(), normal, atol=3e-5, rtol=2e-4)
