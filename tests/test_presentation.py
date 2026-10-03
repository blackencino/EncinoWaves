# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
from dataclasses import fields, replace
import numpy as np
import pytest

from encino_waves.camera import interpolate_camera
from encino_waves.model import Wave_parameters
from encino_waves.presets import presentation_views
from encino_waves.presentation import (Presentation, Cue, school_presentation,
                                      changed_control, sample_presentation)


def test_school_stays_on_one_deep_seeded_patch_and_changes_one_control():
    movie=school_presentation()
    assert movie.duration==300
    assert movie.initial.parameters.depth==1000
    assert movie.foam.resolution==1024
    initial=movie.initial.parameters
    last=movie.initial
    conditions=[]
    for cue in movie.cues:
        p=cue.target.parameters
        assert (p.resolution,p.domain,p.seed)==(4096,1000,initial.seed)
        changed_control(last,cue.target)
        if cue.title in ("Chaos","Lawful evil","Shallow chop"):
            conditions.append((cue.title,p))
        last=cue.target
    chaos=next(p for title,p in conditions if title=="Chaos")
    orderly=next(p for title,p in conditions if title=="Lawful evil")
    shallow=next(p for title,p in conditions if title=="Shallow chop")
    assert chaos.wind_speed==orderly.wind_speed==shallow.wind_speed==32
    assert chaos.fetch_km==20 and chaos.swell<0
    assert orderly.fetch_km==1250 and orderly.swell>=1
    assert shallow.depth<=2 and shallow.swell<0


def test_camera_and_physical_values_are_continuous_at_every_cue_boundary():
    movie=school_presentation()
    time=0.
    for cue in movie.cues[:-1]:
        time+=cue.duration
        before=sample_presentation(movie,time-1e-4).state
        after=sample_presentation(movie,time+1e-4).state
        for f in fields(Wave_parameters):
            a,b=getattr(before.parameters,f.name),getattr(after.parameters,f.name)
            if isinstance(a,(float,int)): assert a==pytest.approx(b,abs=1e-6)
            else: assert a==b
        np.testing.assert_allclose(before.camera.eye,after.camera.eye,atol=1e-6)
        np.testing.assert_allclose(before.camera.basis(),after.camera.basis(),atol=1e-7)
        assert before.comparison==pytest.approx(after.comparison,abs=1e-8)


def test_sunset_uses_supplied_maya_fields_and_short_camera_arc():
    views=presentation_views(1000)
    camera=views[3].camera
    assert views[3].name=="Sunset"
    assert (camera.height,camera.pitch,camera.yaw)==(92.5,-15.4,-487.6)
    np.testing.assert_allclose(camera.pivot,0,atol=1e-12)
    previous=views[2].camera
    middle=interpolate_camera(previous,camera,.5)
    assert abs(middle.yaw-previous.yaw)<20
    np.testing.assert_allclose(middle.pivot,0,atol=1e-12)
    assert interpolate_camera(previous,camera,1)==camera


def test_timeline_rejects_cuts_and_multiple_physical_edits():
    movie=school_presentation(32,32)
    initial=movie.initial
    for changes in ({"domain":1200},{"seed":999},{"wind_speed":25,"swell":1}):
        target=replace(initial,parameters=replace(initial.parameters,**changes))
        with pytest.raises(ValueError):
            Presentation(initial,(Cue(10,"Invalid","",target),))
    target=replace(initial,parameters=replace(initial.parameters,depth=3),
                   camera=presentation_views()[0].camera)
    with pytest.raises(ValueError,match="only one"):
        Presentation(initial,(Cue(10,"Invalid","",target),))


def test_preview_keeps_the_same_endpoints_and_reaches_full_caption_opacity():
    full=school_presentation()
    preview=school_presentation(preview=True)
    assert preview.duration==pytest.approx(60)
    elapsed=0
    for a,b in zip(full.cues,preview.cues):
        assert a.target==b.target
        assert a.duration==pytest.approx(5*b.duration)
        assert sample_presentation(preview,elapsed+.5*b.duration).caption_opacity==1
        elapsed+=b.duration
