"""Continuous state/history, full-frame comparison and Academy CLI integration."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import wgpu

from encino_waves import export
from encino_waves.camera import Camera
from encino_waves.cli import main
from encino_waves.foam import Foam_parameters
from encino_waves.model import Wave_parameters, phase_at
from encino_waves.presentation import Presentation, Presentation_state, Cue, sample_presentation


def small_presentation():
    initial=Presentation_state(Wave_parameters(resolution=16,domain=1000.,depth=1000.),Camera())
    current=initial
    cues=[]
    def add(duration=.5,**changes):
        nonlocal current
        if "comparison" in changes or "camera" in changes:
            current=replace(current,**changes)
        else:
            current=replace(current,parameters=replace(current.parameters,**changes))
        cues.append(Cue(duration,"A continuous sea","One control at a time",current))
    add()
    add(wind_speed=5.)
    add(comparison=1.)
    add()
    add(comparison=0.)
    add(camera=replace(initial.camera,yaw=initial.camera.yaw+30))
    add(1.,depth=1.5)
    add(wind_speed=32.)
    add(swell=-.7)
    add()
    foam=Foam_parameters(resolution=16,emission=0,diffusion=0,exchange=0,
        surface_half_life=1e15,shallow_half_life=1e15,deep_half_life=1e15)
    return Presentation(initial,tuple(cues),foam=foam,foam_preroll=0)


def test_one_basis_one_main_foam_history_and_continuous_phase():
    presentation=small_presentation()
    with patch.object(export,"make_wave_basis",wraps=export.make_wave_basis) as basis_setup, \
         patch.object(export,"prepare_foam",wraps=export.prepare_foam) as foam_setup, \
         patch.object(export,"edit_state",wraps=export.edit_state) as edits:
        ocean=export._Presentation_ocean(presentation,"cpu")
        basis=ocean.basis
        foam_basis=ocean.foam.basis
        ocean.foam=replace(ocean.foam,density=torch.ones_like(ocean.foam.density))
        changed_frames=0
        comparisons=0
        for i in range(round(presentation.duration*8)):
            elapsed=i/8
            sample=sample_presentation(presentation,elapsed)
            previous=ocean.state
            before=phase_at(previous,presentation.start_time+elapsed)
            changed_frames+=sample.state.parameters!=previous.parameters
            frame,foam,other,_=ocean.advance(sample,elapsed)
            after=phase_at(ocean.state,frame.time)
            for trig in (torch.sin,torch.cos):
                torch.testing.assert_close(trig(after),trig(before),atol=4e-5,rtol=2e-5)
            assert ocean.basis is basis and ocean.state.multipliers is basis.multipliers
            assert foam.basis is foam_basis
            torch.testing.assert_close(foam.density,torch.ones_like(foam.density),atol=1e-6,rtol=0)
            assert foam.time==frame.time
            if sample.state.parameters==previous.parameters:
                assert ocean.state is previous
            if other is not None:
                comparisons+=1
                assert other.parameters==frame.parameters.tessendorf()
                assert other.time==frame.time
                assert ocean.comparison_state.phase is ocean.state.phase
                assert ocean.comparison_state.phase_steps is ocean.state.phase_steps
        assert basis_setup.call_count==1
        assert foam_setup.call_count==2  # Main once, comparison once when it appears.
        assert edits.call_count==changed_frames
        assert comparisons>1
        assert len(ocean.state.phase_steps)>1  # Depth changed over multiple frames.


class _Target:
    def __init__(self,size,format,**kwargs):
        self.size,self.format=size,format
        self.pixels=np.zeros((size[1],size[0],4),np.uint8)
        self.destroyed=False
    def create_view(self): return self
    def destroy(self): self.destroyed=True


@pytest.fixture
def mocked_output(monkeypatch):
    targets=[]
    def texture(**kwargs):
        result=_Target(**kwargs)
        targets.append(result)
        return result
    graphics=SimpleNamespace(adapter=SimpleNamespace(info={"device":"Mock GPU"}),create_texture=texture)
    renderers=[]
    class Renderer:
        format="rgba8unorm"
        sample_count=4
        sky_name="Controlled test sky"
        def __init__(self,device,sky,mesh_resolution):
            self.mesh_resolution=mesh_resolution
            self.draws=[]
            renderers.append(self)
        def upload(self,frame): self.frame=frame
        def upload_foam(self,foam): self.foam=foam
        def draw(self,target,width,height,camera,look,**kwargs):
            assert not kwargs  # No viewport/aspect change is allowed.
            self.draws.append((self.frame.time,width,height,camera,look))
            target.pixels[...,:3]=210 if self.frame.parameters.spectrum=="pm" else 10
            target.pixels[...,3]=255
    overlays=[]
    class Overlay:
        def __init__(self,device,current,earlier,format):
            self.current,self.earlier=current,earlier
            self.target=texture(size=current.size,format=format)
            self.opacities=[]
            overlays.append(self)
        def compose(self,opacity):
            self.opacities.append(opacity)
            self.target.pixels[:]=self.current.pixels
            half=self.current.size[0]//2
            self.target.pixels[:,:half]=np.rint((1-opacity)*self.current.pixels[:,:half]+opacity*self.earlier.pixels[:,:half])
            return self.target
        def destroy(self): self.target.destroy()
    writers=[]
    class Writer:
        def __init__(self,*args):
            self.frames=[]
            self.committed=None
            writers.append(self)
        def write(self,pixels): self.frames.append(pixels.copy())
        def close(self,commit=True): self.committed=commit
    monkeypatch.setattr(export,"make_device",lambda:graphics)
    monkeypatch.setattr(export,"Ocean_renderer",Renderer)
    monkeypatch.setattr(export,"_Comparison_overlay",Overlay)
    monkeypatch.setattr(export,"Movie_writer",Writer)
    monkeypatch.setattr(export,"read_rgba",lambda device,target,width,height:target.pixels.copy())
    return SimpleNamespace(targets=targets,renderers=renderers,overlays=overlays,writers=writers)


def test_movie_uses_matched_full_frames_and_records_continuous_cues(mocked_output,tmp_path):
    presentation=small_presentation()
    output=tmp_path/"presentation.mp4"
    result=export.render_presentation(presentation,output,device="cpu",width=32,height=16,fps=8,
                                      captions=False,progress=lambda *args,**kwargs:None)
    writer=mocked_output.writers[0]
    assert writer.committed and len(writer.frames)==round(presentation.duration*8)
    main,other=mocked_output.renderers
    main_draws={draw[0]:draw for draw in main.draws}
    assert all(draw==main_draws[draw[0]] for draw in other.draws)
    for i,pixels in enumerate(writer.frames):
        opacity=sample_presentation(presentation,i/8).state.comparison
        np.testing.assert_array_equal(pixels[:,16:],10)
        np.testing.assert_allclose(pixels[:,:16],10+200*opacity,atol=.5)
    assert any(0<opacity<1 for opacity in mocked_output.overlays[0].opacities)
    assert all(target.destroyed for target in mocked_output.targets)
    saved=json.loads(output.with_name(output.name+".json").read_text())
    assert saved["type"]=="continuous_presentation"
    assert saved["initial"]["parameters"]["domain"]==1000
    assert saved["wave_resolution"]==saved["foam_resolution"]==16
    assert saved["mesh_resolution"]==[960,576] and saved["antialiasing_samples"]==4
    assert saved["cues"][0]["start_seconds"]==0
    assert saved["cues"][-1]["end_seconds"]==presentation.duration
    assert {cue["control"] for cue in saved["cues"]}=={None,"wind_speed","comparison","camera","depth","swell"}
    assert result["final_phase_steps"]


def test_uhd_export_retains_screen_space_mesh_density(mocked_output,tmp_path):
    original=small_presentation()
    presentation=replace(original,cues=(replace(original.cues[0],duration=.25),))
    export.render_presentation(presentation,tmp_path/"uhd.mp4",device="cpu",width=3840,height=2160,
                               fps=4,captions=False,progress=lambda *args,**kwargs:None)
    assert mocked_output.renderers[0].mesh_resolution==(1920,1152)


@pytest.mark.parametrize("preview,duration",[(False,300),(True,60)])
def test_demo_cli_selects_continuous_timeline_and_defaults(monkeypatch,tmp_path,preview,duration):
    received=[]
    def render(presentation,output,**kwargs):
        received.append((presentation,kwargs))
        return {"duration_seconds":presentation.duration,"frames":round(presentation.duration*kwargs["fps"])}
    monkeypatch.setattr(export,"render_presentation",render)
    with patch.object(export,"render_shots",side_effect=AssertionError("Independent-shot demo")):
        main(["demo",str(tmp_path/"demo.mp4"),"--no-foam","--foam-preroll","0"]+(["--preview"] if preview else []))
    presentation,kwargs=received[0]
    assert presentation.duration==pytest.approx(duration)
    assert presentation.initial.parameters.domain==1000
    assert presentation.initial.parameters.resolution==4096
    assert presentation.foam.resolution==1024 and not presentation.foam.enabled
    assert presentation.foam_preroll==0


def test_saved_shot_cli_still_uses_independent_shot_export(monkeypatch,tmp_path):
    captured=[]
    def render(shots,output,**kwargs):
        captured.extend(shots)
        return {"duration_seconds":shots[0].duration,"frames":4}
    monkeypatch.setattr(export,"render_shots",render)
    with patch.object(export,"render_presentation",side_effect=AssertionError("Changed saved-shot path")):
        main(["render",str(tmp_path/"shot.mp4"),"--seconds","1","--resolution","16","--fps","4"])
    assert len(captured)==1 and isinstance(captured[0],export.Shot)
    assert captured[0].parameters.resolution==16


def test_caption_dissolve_keeps_live_controls_and_fades_comparison():
    rgba=np.full((216,384,4),30,np.uint8)
    rgba[...,3]=255
    parameters=Wave_parameters(wind_speed=17.123,fetch_km=1234.987,depth=1.567,swell=.349)
    assert export._physical_caption(parameters)=="Wind 17.1 m/s   Fetch 1235 km   Depth 1.6 m   Swell 0.35"
    faded=export.caption_image(rgba,"Title","Caption",parameters,opacity=0,rounded_controls=True)
    shown=export.caption_image(rgba,"Title","Caption",parameters,opacity=1,rounded_controls=True)
    np.testing.assert_array_equal(faded[:60],shown[:60])
    assert np.any(faded[:60]!=30)  # Actual physical controls remain visible.
    np.testing.assert_array_equal(faded[100:],30)
    assert np.any(shown[170:]!=30)
    partial=export.caption_image(rgba,"","",parameters,.25,opacity=0)
    full=export.caption_image(rgba,"","",parameters,1,opacity=0)
    assert 30<partial[108,192,0]<full[108,192,0]


def test_gpu_comparison_overlay_preserves_right_half_and_pixel_projection():
    try:
        graphics=export.make_device()
    except RuntimeError:
        pytest.skip("Hardware graphics adapter unavailable")
    width,height=32,16
    y,x=np.mgrid[:height,:width]
    main=np.stack((x*4+10,y*8+5,x+y,np.full_like(x,255)),axis=-1).astype(np.uint8)
    earlier=main.copy()
    earlier[...,:3]=255-main[...,:3]
    textures=[]
    for pixels in (main,earlier):
        texture=graphics.create_texture(size=(width,height,1),format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
        graphics.queue.write_texture({"texture":texture},pixels,
            {"bytes_per_row":width*4,"rows_per_image":height},(width,height,1))
        textures.append(texture)
    overlay=export._Comparison_overlay(graphics,*textures,"rgba8unorm")
    for opacity in (0.,.25,.75,1.):
        result=export.read_rgba(graphics,overlay.compose(opacity),width,height)
        expected=main.astype(np.float32)
        expected[:,:width//2]=(1-opacity)*main[:,:width//2]+opacity*earlier[:,:width//2]
        np.testing.assert_allclose(result,expected,atol=1,rtol=0)
        np.testing.assert_array_equal(result[:,width//2:],main[:,width//2:])
    overlay.destroy()
    for texture in textures: texture.destroy()
