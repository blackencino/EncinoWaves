"""The historical wave comparison must not inherit modern foam or scattering."""
from dataclasses import replace

import numpy as np
import pytest

from test_water_rendering import CAMERA, LOOK, device, renderer_factory, linear_hdr


def historical_frame(renderer):
    frame = renderer.frame
    renderer.upload(replace(frame,parameters=replace(frame.parameters,spectrum="phillips")))


@pytest.mark.parametrize("samples", [1, 4])
def test_historical_water_ignores_loaded_foam_and_modern_look(renderer_factory, monkeypatch, samples):
    renderer = renderer_factory(samples=samples,foam_density=(2.,4.,8.))
    historical_frame(renderer)
    # Crest values and retained foam history deliberately make every modern
    # mechanism eligible. None may affect the historical water or process foam.
    displacement = renderer.frame.displacement.clone()
    displacement[...,3] = -.1
    renderer.upload(replace(renderer.frame,displacement=displacement))
    history = renderer.foam_state
    def unexpected_foam_update(*args, **kwargs):
        pytest.fail("Historical shading must not build the modern foam material")
    monkeypatch.setattr(renderer.foam_material,"update",unexpected_foam_update)
    baseline = renderer.render_image(112,80,CAMERA,LOOK)
    assert np.isfinite(linear_hdr(renderer)).all()
    assert linear_hdr(renderer)[-20:,:,:3].mean() < .9  # Actual water is in view.
    loaded = replace(LOOK,material="2015",foam=4.,aeration=20.,
                     crest_threshold=-20.,crest_maximum=-10.,crest_crumble=True)
    np.testing.assert_array_equal(renderer.render_image(112,80,CAMERA,loaded),baseline)
    assert renderer.active_material == "physical"
    assert loaded.material == "2015" and loaded.foam == 4. and loaded.crest_crumble
    renderer.upload_foam(None)
    np.testing.assert_array_equal(renderer.render_image(112,80,CAMERA,loaded),baseline)
    # Hiding historical foam is non-destructive; an existing modern scene can
    # still restore its history after the comparison.
    assert history.density.max() == 8.


def test_historical_water_has_no_direct_volume_scattering_and_keeps_exposure(renderer_factory):
    renderer = renderer_factory(foam_density=(0.,4.,8.))
    historical_frame(renderer)
    baseline = renderer.render_image(112,80,CAMERA,LOOK)
    renderer.lighting = replace(renderer.lighting,direct_irradiance=np.array([100.,80.,50.]))
    np.testing.assert_array_equal(renderer.render_image(112,80,CAMERA,LOOK),baseline)
    brighter = renderer.render_image(112,80,CAMERA,replace(LOOK,exposure=LOOK.exposure+1.))
    assert brighter[...,:3].mean() > baseline[...,:3].mean()
    assert np.isfinite(linear_hdr(renderer)).all()


def test_returning_to_modern_water_restores_foam_shading(renderer_factory):
    renderer = renderer_factory(foam_density=(2.,4.,8.))
    frame = renderer.frame
    look = replace(LOOK,foam=1.,aeration=1.5,crest_crumble=True)
    modern = renderer.render_image(112,80,CAMERA,look)
    historical_frame(renderer)
    historical = renderer.render_image(112,80,CAMERA,look)
    assert np.count_nonzero(historical != modern) > 100
    renderer.upload(frame)
    np.testing.assert_array_equal(renderer.render_image(112,80,CAMERA,look),modern)
