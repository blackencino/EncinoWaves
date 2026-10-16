import test from 'node:test';
import assert from 'node:assert/strict';
import {initialQuality,IPhoneQualityGuard} from '../quality.js';

const iphone={userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1',platform:'iPhone',maxTouchPoints:5,width:402,height:874};
const automatic={quality:'high',source:'iphone-auto',playing:true};
const advance=(guard,fps,seconds,state=automatic)=>{let change=null;for(let i=0;i<fps*seconds;i++)change=guard.sample(1/fps,state)||change;return change;};

test('iPhones start at 1024 without relying on OS version, screen size, or a model whitelist',()=>{
  for(const screen of [[402,874],[440,956],[390,844],[500,1100]])
    assert.deepEqual(initialQuality({...iphone,width:screen[0],height:screen[1]}),{quality:'high',source:'iphone-auto'});
  assert.equal(initialQuality({...iphone,userAgent:'Future browser',platform:'iPhone'}).quality,'high');
  assert.equal(initialQuality({...iphone,platform:'',userAgent:iphone.userAgent.replace('Version/26.0','CriOS/145.0')}).quality,'high');
});

test('explicit quality overrides are authoritative and invalid values are ignored',()=>{
  for(const quality of ['low','medium','high'])assert.deepEqual(initialQuality({...iphone,requested:quality}),{quality,source:'url'});
  for(const requested of ['__proto__','constructor','',null])assert.equal(initialQuality({...iphone,requested}).quality,'high');
});

test('Android and laptop defaults are preserved',()=>{
  assert.equal(initialQuality({...iphone,userAgent:'Mozilla/5.0 (Linux; Android 16)',platform:'Linux armv8l'}).quality,'low');
  assert.equal(initialQuality({userAgent:'Macintosh',maxTouchPoints:0,width:1440,height:900}).quality,'medium');
});

test('capable iPhones retain 1024 at 30, 60, and 120 fps',()=>{
  for(const fps of [30,60,120])assert.equal(advance(new IPhoneQualityGuard(),fps,60),null);
});

test('only sustained slow automatic playback lowers the grid, after startup grace',()=>{
  const guard=new IPhoneQualityGuard();
  assert.equal(advance(guard,15,12),null);
  assert.equal(advance(guard,15,5),'medium');
  assert.equal(advance(guard,15,17,{...automatic,quality:'medium'}),'low');
});

test('manual choices, URL overrides, pauses, and background gaps do not lower the grid',()=>{
  for(const state of [{...automatic,source:'manual'},{...automatic,source:'url'},{...automatic,playing:false}])
    assert.equal(advance(new IPhoneQualityGuard(),15,40,state),null);
  const guard=new IPhoneQualityGuard();advance(guard,15,14);
  assert.equal(guard.sample(3,automatic),null);
  assert.equal(advance(guard,15,12),null);
  guard.reset();assert.equal(advance(guard,60,40),null);
});
