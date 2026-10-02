from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from encino_waves.model import (Wave_parameters, make_initial_state, evaluate,
    dispersion_at, spectrum_at, spreading_at, peak_omega)


@pytest.mark.parametrize("spreading",["hasselmann","mitsuyasu","donelan_banner","cosine_squared"])
@pytest.mark.parametrize("swell",[-1,0,.35,1,2])
def test_directional_energy_is_one(spreading,swell):
    p=Wave_parameters(spreading=spreading,swell=swell)
    for ratio in (.1,.6,1,1.5,3,10):
        omega=ratio*peak_omega(p)
        integral=quad(lambda theta: float(spreading_at(p,omega,theta)),-math.pi,math.pi,epsabs=1e-7,points=[-math.pi/2,math.pi/2])[0]
        assert integral == pytest.approx(1,abs=2e-5)


@pytest.mark.parametrize("mode",["deep","finite","capillary"])
def test_dispersion_derivative(mode):
    p=Wave_parameters(depth=8,dispersion=mode)
    k=np.geomspace(.001,1000,100)
    _,analytic=dispersion_at(p,k)
    lo,_=dispersion_at(p,k*.99999)
    hi,_=dispersion_at(p,k*1.00001)
    np.testing.assert_allclose(analytic,(hi-lo)/(.00002*k),rtol=1e-7)


def test_physical_limits_and_comparison():
    p=Wave_parameters()
    omega=np.geomspace(.02,100,200)
    np.testing.assert_allclose(spectrum_at(replace(p,depth=1e7),omega),spectrum_at(replace(p,spectrum="jonswap"),omega),rtol=1e-12)
    assert np.all(spectrum_at(replace(p,depth=2),omega)<=spectrum_at(p,omega))
    a=p.tessendorf()
    b=replace(p,fetch_km=1000,swell=1).tessendorf()
    np.testing.assert_array_equal(spectrum_at(a,omega),spectrum_at(b,omega))
    theta=np.linspace(-math.pi,math.pi,100)
    np.testing.assert_allclose(spreading_at(a,omega[0],theta),spreading_at(a,omega[-1],theta))


def test_resolution_preserves_shared_wave_components():
    p=Wave_parameters(resolution=32,spreading="hasselmann")
    a=make_initial_state(p,"cpu")
    b=make_initial_state(replace(p,resolution=64),"cpu")
    for name in ("h_positive","h_negative","omega"):
        low=getattr(a,name).numpy()
        high=getattr(b,name).numpy()
        np.testing.assert_array_equal(low[:16,:16],high[:16,:16])
        np.testing.assert_array_equal(low[-15:,:16],high[-15:,:16])


def test_functional_evaluation_has_no_history_or_mutation():
    state=make_initial_state(Wave_parameters(resolution=32),"cpu")
    initial=state.h_positive.clone()
    a=evaluate(state,5)
    evaluate(state,900)
    b=evaluate(state,5)
    torch.testing.assert_close(a.displacement,b.displacement,rtol=0,atol=0)
    torch.testing.assert_close(initial,state.h_positive,rtol=0,atol=0)
    assert torch.isfinite(a.normal).all()


@pytest.mark.parametrize("device",["cpu","mps","cuda"])
def test_all_inverse_transforms_against_fftw(device):
    if device=="mps" and not torch.backends.mps.is_available(): pytest.skip("Metal not visible")
    if device=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA not visible")
    fftw=pytest.importorskip("pyfftw.interfaces.numpy_fft")
    p=Wave_parameters(resolution=128,spreading="hasselmann",depth=13,trough_damping=0)
    state=make_initial_state(p,device)
    pos,neg,w=(t.cpu().numpy() for t in (state.h_positive,state.h_negative,state.omega))
    for time in (0,1.25,120):
        # Independent CPU phase evaluation, double precision FFTW reference.
        phase=w.astype(np.float64)*time
        h=pos*np.exp(-1j*phase)+neg*np.exp(1j*phase)
        reference=fftw.irfft2(state.multipliers.cpu().numpy()*h[None],s=(128,128),norm="forward")
        frame=evaluate(state,time)
        height,dx,dy,dxx,dyy,dxy=reference
        jxx,jyy,jxy=1-p.pinch*dxx,1-p.pinch*dyy,-p.pinch*dxy
        crest=-(jxx+jyy-np.sqrt((jxx-jyy)**2+4*jxy**2))/2
        expected=np.stack((-p.pinch*dx,-p.pinch*dy,height,crest),axis=-1)
        np.testing.assert_allclose(frame.displacement.cpu().numpy(),expected,atol=6e-4,rtol=3e-4)


@pytest.mark.parametrize("kwargs",[{"resolution":100},{"wind_speed":0},{"depth":-1},{"fetch_km":float('nan')},{"swell":3}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError): Wave_parameters(**kwargs)
