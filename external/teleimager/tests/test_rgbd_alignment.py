"""Verify metric RGBD projection against SDK geometry without robot motion."""
import numpy as np
import pytest

from teleimager.rgbd_stream import NativeDepthAligner

rs = pytest.importorskip("pyrealsense2")


def sdk_reference(c,raw):
    intr=[]
    for name in ('depth','color'):
        i=rs.intrinsics()
        for k,v in c[name].items():setattr(i,k,v)
        i.model=rs.distortion.none;intr.append(i)
    e=rs.extrinsics();e.rotation=c['rotation'];e.translation=c['translation']
    out=np.zeros((intr[1].height,intr[1].width),np.uint16)
    for y,x in zip(*np.nonzero(raw)):
        pixels=[]
        for offset in (-.5,.5):
            point=rs.rs2_deproject_pixel_to_point(intr[0],[x+offset,y+offset],float(raw[y,x]*c['scale']))
            point=rs.rs2_transform_point_to_point(e,point)
            pixels.append(rs.rs2_project_point_to_pixel(intr[1],point))
        (x0,y0),(x1,y1)=([int(z+.5) for z in pixel] for pixel in pixels)
        if x0<0 or y0<0 or x1>=intr[1].width or y1>=intr[1].height:continue
        for py in range(y0,y1+1):
            for px in range(x0,x1+1):out[py,px]=min(out[py,px],raw[y,x]) if out[py,px] else raw[y,x]
    return out
def test_factory_projection_matches_sdk_with_rotation_translation_and_occlusions():
    d=dict(width=32,height=24,fx=20.,fy=20.,ppx=16.,ppy=12.,coeffs=[0]*5)
    c=dict(width=32,height=24,fx=30.,fy=30.,ppx=17.,ppy=11.,coeffs=[0]*5)
    theta=.02;rotation=[np.cos(theta),0,-np.sin(theta),0,1,0,np.sin(theta),0,np.cos(theta)]
    calib=dict(depth=d,color=c,scale=.001,rotation=rotation,translation=[.015,.002,0])
    raw=np.random.default_rng(42).integers(500,3000,size=(24,32),dtype=np.uint16);raw[5:8,9:15]=0
    expected=sdk_reference(calib,raw);actual=NativeDepthAligner(calib).compute(raw)
    np.testing.assert_array_equal(actual,expected)
