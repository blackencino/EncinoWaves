# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""One continuous ocean: authored physical edits and Maya camera moves.

The timeline contains target values, not independent seeded shots. Each cue
changes one physical control, the camera, or the comparison overlay. The seed,
patch, lighting, propagating phase and foam history remain continuous.
"""
from dataclasses import dataclass, fields, replace
import math

from .camera import Camera, interpolate_camera
from .foam import Foam_parameters
from .model import Wave_parameters
from .presets import presentation_views
from .render import Look


@dataclass(frozen=True)
class Presentation_state:
    parameters: Wave_parameters
    camera: Camera
    comparison: float = 0.0


@dataclass(frozen=True)
class Cue:
    duration: float
    title: str
    caption: str
    target: Presentation_state


def changed_control(start,end):
    changed=[f.name for f in fields(Wave_parameters)
             if getattr(start.parameters,f.name)!=getattr(end.parameters,f.name)]
    if start.camera!=end.camera: changed.append("camera")
    if start.comparison!=end.comparison: changed.append("comparison")
    if len(changed)>1:
        raise ValueError(f"A presentation cue must change only one control: {changed}")
    if changed and changed[0] not in ("wind_speed","fetch_km","swell","depth","camera","comparison"):
        raise ValueError(f"A continuous presentation cannot change {changed[0]}")
    return changed[0] if changed else None


@dataclass(frozen=True)
class Presentation:
    initial: Presentation_state
    cues: tuple[Cue,...]
    look: Look = Look()
    foam: Foam_parameters = Foam_parameters(resolution=1024)
    start_time: float = 10.0
    foam_preroll: float = 6.0

    def __post_init__(self):
        if not self.cues: raise ValueError("A presentation needs at least one cue")
        previous=self.initial
        for cue in self.cues:
            if not math.isfinite(cue.duration) or cue.duration<=0:
                raise ValueError("Cue durations must be finite and positive")
            changed_control(previous,cue.target)
            if not 0<=cue.target.comparison<=1:
                raise ValueError("Comparison opacity must be between zero and one")
            previous=cue.target

    @property
    def duration(self):
        return sum(cue.duration for cue in self.cues)


@dataclass(frozen=True)
class Presentation_sample:
    state: Presentation_state
    cue_index: int
    control: str | None
    caption_opacity: float


def smooth_step(value):
    value=max(0.,min(1.,value))
    return value**3*(value*(6*value-15)+10)


def sample_presentation(presentation,time):
    """Pure timeline lookup; a quintic ease has zero endpoint speed/acceleration."""
    if not math.isfinite(time): raise ValueError("Presentation time must be finite")
    elapsed=0.
    previous=presentation.initial
    for index,cue in enumerate(presentation.cues):
        if time<elapsed+cue.duration or index==len(presentation.cues)-1:
            local=max(0.,min(cue.duration,time-elapsed))
            amount=smooth_step(local/cue.duration)
            control=changed_control(previous,cue.target)
            state=previous
            if control=="camera":
                state=replace(previous,camera=interpolate_camera(previous.camera,cue.target.camera,amount))
            elif control=="comparison":
                state=replace(previous,comparison=(1-amount)*previous.comparison+amount*cue.target.comparison)
            elif control:
                a=getattr(previous.parameters,control)
                b=getattr(cue.target.parameters,control)
                value=math.exp((1-amount)*math.log(a)+amount*math.log(b)) if control in ("fetch_km","depth") else (1-amount)*a+amount*b
                if amount==0: value=a
                if amount==1: value=b
                state=replace(previous,parameters=replace(previous.parameters,**{control:value}))
            # Lower thirds dissolve while the running physical values remain.
            fade=min(.6,cue.duration*.2)
            opacity=min(smooth_step(local/fade),smooth_step((cue.duration-local)/fade))
            return Presentation_sample(state,index,control,opacity)
        elapsed+=cue.duration
        previous=cue.target
    raise AssertionError("Unreachable presentation time")


def academy_presentation(resolution=4096,foam_resolution=1024,*,preview=False):
    """Five minutes, one kilometre, no cuts, reseeds or simultaneous controls.

    Preview compresses the authored timings to one minute. Propagation still
    advances at real time; it is an editing/layout preview, not a sped-up movie.
    """
    views=presentation_views(1000.)
    p=Wave_parameters(resolution=resolution,domain=1000.,depth=1000.)
    initial=Presentation_state(p,views[1].camera)
    state=initial
    cues=[]
    def cue(seconds,title,caption,*,camera=None,comparison=None,**physical):
        nonlocal state
        if physical: state=replace(state,parameters=replace(state.parameters,**physical))
        if camera is not None: state=replace(state,camera=camera)
        if comparison is not None: state=replace(state,comparison=comparison)
        cues.append(Cue(seconds/(5 if preview else 1),title,caption,state))

    cue(12,"Encino Waves","Christopher J. Horvath | An ocean shaped by real conditions")
    cue(10,"Deep ocean","One kilometre of ocean. One continuous take.",camera=views[0].camera)
    cue(3,"The starting point","Earlier model / Encino Waves | The same wind, camera and lighting",comparison=1.)
    cue(14,"Wave directions","Different sizes of waves travel in different directions")
    cue(3,"Encino Waves","Returning to the same ocean",comparison=0.)
    cue(12,"Wind speed","Ease the wind. Everything else stays the same.",wind_speed=5.)
    cue(6,"A quieter sea","Wind: 5 metres per second")
    cue(12,"Fetch","Give the wind less room to build waves: 300 to 20 kilometres",fetch_km=20.)
    cue(10,"Swell","Reduce the swell. Let the directions spread.",swell=-.6)
    cue(18,"Wind speed","Keep the short fetch and low swell. Bring in a strong wind.",wind_speed=32.)
    cue(12,"Chaos","Strong wind. Short fetch. Low swell.")
    cue(18,"Chaos","The same conditions, seen across the crests",camera=views[2].camera)
    cue(18,"Fetch","The wind stays at 32 m/s. Give it 1,250 kilometres to work.",fetch_km=1250.)
    cue(6,"Room to grow","Strong wind and long fetch; swell is still low")
    cue(18,"Swell","Organize the wave trains. Change only the swell.",swell=1.2)
    cue(12,"Lawful evil","The same strong wind, with long fetch and organized swell")
    cue(18,"Lawful evil","Moving toward the sunset view",camera=views[3].camera)
    cue(12,"A different direction for the shot","Powerful, ordered waves from the same set of physical controls")
    cue(18,"Swell","Spread the directions again before changing the depth",swell=-.7)
    cue(18,"Ocean depth","From deep ocean to just 1.5 metres of water",depth=1.5)
    cue(12,"Shallow chop","Very shallow water. Negative swell. The strong wind remains.")
    cue(16,"Ocean depth","Return to deep water. The waves keep moving.",depth=1000.)
    cue(16,"Swell","Bring back the long, ordered wave trains",swell=1.2)
    cue(6,"Encino Waves","Empirical directional wave spectra for computer graphics | 2015")
    return Presentation(initial,tuple(cues),foam=Foam_parameters(resolution=foam_resolution))
