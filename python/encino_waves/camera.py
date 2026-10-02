# Copyright 2015-2026 Christopher Jon Horvath. Apache-2.0.
"""Value-based port of SimpleSimViewer/GLCamera.cpp's Z-up Maya camera.

Orbit and dolly preserve the center of interest. Track moves eye and pivot
in the camera plane. Pixel deltas use the original 400 / 5 sensitivities.
"""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class Camera:
    x: float = 319.0
    y: float = -319.0
    height: float = 221.7
    yaw: float = -45.0
    pitch: float = -26.17
    fov: float = 45.0
    center_of_interest: float = 502.82

    def __post_init__(self):
        if not all(math.isfinite(getattr(self,name)) for name in
                   ("x","y","height","yaw","pitch","fov","center_of_interest")):
            raise ValueError("Camera values must be finite")
        if not 0 < self.fov < 179 or self.center_of_interest <= 0:
            raise ValueError("Camera requires a positive center of interest and 0 < fov < 179")

    @property
    def eye(self):
        return np.array((self.x,self.y,self.height),dtype=np.float64)

    def basis(self):
        yaw,pitch=math.radians(self.yaw),math.radians(self.pitch)
        forward=np.array((math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),math.sin(pitch)))
        right=np.array((math.cos(yaw),-math.sin(yaw),0.0))
        return forward,right,np.cross(right,forward)

    @property
    def pivot(self):
        return self.eye+self.center_of_interest*self.basis()[0]

    def orbit(self,dx,dy,width,height):
        changed=replace(self,yaw=self.yaw+400*dx/width,pitch=self.pitch-400*dy/height)
        eye=self.pivot-self.center_of_interest*changed.basis()[0]
        return replace(changed,x=float(eye[0]),y=float(eye[1]),height=float(eye[2]))

    def track(self,dx,dy,width,height):
        _,right,up=self.basis()
        scale=2*self.center_of_interest*math.tan(math.radians(self.fov)/2)
        eye=self.eye-scale*dx/width*right+scale*dy/height*up
        return replace(self,x=float(eye[0]),y=float(eye[1]),height=float(eye[2]))

    def dolly(self,dx,width):
        distance=max(.1,self.center_of_interest*math.exp(-5*dx/width))
        eye=self.pivot-distance*self.basis()[0]
        return replace(self,x=float(eye[0]),y=float(eye[1]),height=float(eye[2]),center_of_interest=distance)


def look_at(eye,target=(0,0,0),fov=45.0):
    eye=np.asarray(eye,dtype=np.float64)
    delta=np.asarray(target,dtype=np.float64)-eye
    distance=float(np.linalg.norm(delta))
    if distance<.1: raise ValueError("Camera eye and target must be distinct")
    return Camera(float(eye[0]),float(eye[1]),float(eye[2]),
                  math.degrees(math.atan2(delta[0],delta[1])),
                  math.degrees(math.atan2(delta[2],np.hypot(delta[0],delta[1]))),fov,distance)


def frame_domain(domain,fov=45.0):
    # ViewScene::getBounds before mesh creation: +/- 0.125 * domain.
    radius=.5*math.sqrt(3)*.25*domain
    g=1.1*radius/math.sin(math.radians(fov)/2)
    return look_at((g,-g,2*radius),fov=fov)


def interpolate_camera(start,end,amount):
    """Interpolate Maya orbit, pivot and distance, taking the shorter yaw arc."""
    if amount<=0 or start==end: return start
    if amount>=1: return end
    yaw_delta=(end.yaw-start.yaw+180)%360-180
    distance=math.exp((1-amount)*math.log(start.center_of_interest)
                      +amount*math.log(end.center_of_interest))
    result=replace(start,yaw=start.yaw+amount*yaw_delta,
        pitch=(1-amount)*start.pitch+amount*end.pitch,
        fov=(1-amount)*start.fov+amount*end.fov,center_of_interest=distance)
    pivot=(1-amount)*start.pivot+amount*end.pivot
    eye=pivot-distance*result.basis()[0]
    return replace(result,x=float(eye[0]),y=float(eye[1]),height=float(eye[2]))
