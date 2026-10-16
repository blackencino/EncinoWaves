import test from 'node:test';
import assert from 'node:assert/strict';
import { SchoolControls, authoredCamera, basis, cameraGesture, interpolateCamera,
  controls, position, valueAt, scenes } from '../controls.js';

const near = (a,b,tolerance=1e-9) => assert.ok(Math.abs(a-b) <= tolerance, `${a} != ${b}`);
const pivot = c => basis(c).forward.map((v,i) => [c.x,c.y,c.height][i]+v*c.center_of_interest);

test('authored School compositions retain the native origin and lens', () => {
  assert.equal(scenes.length,5);
  for(const scene of scenes) {
    for(const value of pivot(scene.camera)) near(value,0,1e-8);
    near(scene.camera.fov,40.735,.001);
    assert.equal(scene.parameters.domain,1024);
  }
  near(scenes[1].parameters.wind_speed,50*1852/3600);
  near(scenes[3].parameters.wind_speed,4*1852/3600);
});

test('each performance control reaches its exact range endpoints and round-trips', () => {
  for(const [name,range] of Object.entries(controls)) {
    near(valueAt(name,-1),range.low);
    near(valueAt(name,2),range.high);
    for(const p of [0,.1,.5,.9,1]) near(position(name,valueAt(name,p)),p);
  }
});

test('arrow acceleration and release produce the same motion at different frame rates', () => {
  function run(fps) {
    const c=new SchoolControls();
    c.selectMode('wind_speed',0); c.held.right=true;
    for(let frame=1;frame<=fps;frame++)c.update(1/fps,frame/fps);
    const released=c.parameters.wind_speed;
    c.held.right=false;
    for(let frame=1;frame<=fps;frame++)c.update(1/fps,1+frame/fps);
    assert.ok(c.parameters.wind_speed>released,'Release must retain the native inertial glide');
    return c;
  }
  const slow=run(15),fast=run(120);
  near(slow.parameters.wind_speed,fast.parameters.wind_speed,1e-8);
  near(slow.velocity.wind_speed,fast.velocity.wind_speed,1e-9);
});

test('scene transitions preserve their starting state and reach the authored endpoint', () => {
  const c=new SchoolControls(),before=structuredClone(c.parameters);
  c.setScene(2,10); c.update(0,10);
  for(const name of Object.keys(controls))near(c.parameters[name],before[name]);
  c.update(1.5,11.5);
  assert.ok(c.parameters.wind_speed>scenes[2].parameters.wind_speed && c.parameters.wind_speed<before.wind_speed);
  c.update(1.5,13);
  for(const name of Object.keys(controls))near(c.parameters[name],scenes[2].parameters[name]);
  for(const key of Object.keys(c.camera))near(c.camera[key],scenes[2].camera[key],1e-7);
  assert.equal(c.transition,null);
});

test('Maya orbit and dolly preserve the center of interest, and orbit clamps pitch', () => {
  const start=authoredCamera(145.2,-18.1,-139.9),expected=pivot(start);
  for(const mode of ['orbit','dolly']) {
    const changed=cameraGesture(start,mode,140,30,1440,810);
    assert.notDeepEqual(changed,start);
    pivot(changed).forEach((value,index)=>near(value,expected[index],1e-7));
  }
  const changed=cameraGesture(start,'orbit',0,100000,1440,810);
  assert.ok(changed.pitch>=-89 && changed.pitch<=-.5);
});

test('camera interpolation crosses the nearest yaw boundary smoothly', () => {
  const a=authoredCamera(40,-10,179),b=authoredCamera(40,-10,-179);
  const middle=interpolateCamera(a,b,.5);
  near(Math.abs(middle.yaw),180);
  pivot(middle).forEach(value=>near(value,0,1e-8));
});


test('Tessendorf controls retain manual gain between sea compositions and suppress modern-only controls', () => {
  const c=new SchoolControls();
  c.setWaveModel(true,0);
  c.setValue('phillips_amplitude',.5,0);
  const gain=c.parameters.phillips_amplitude;
  c.selectMode('foam_amount',1);
  assert.equal(c.mode,'phillips_amplitude');
  c.setScene(2,1);c.update(3,4);
  assert.equal(c.parameters.spectrum,'phillips');
  near(c.parameters.phillips_amplitude,gain);
  c.setWaveModel(false,5);
  assert.equal(c.parameters.spectrum,'tma');
  c.selectMode('phillips_amplitude',5);
  assert.equal(c.mode,null);
  c.selectMode('wind_speed',5);
  assert.equal(c.mode,'wind_speed');
});
