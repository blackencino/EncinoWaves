// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
import {SchoolControls,controls,scenes,position,formatValue,cameraGesture,clamp} from './controls.js';
import {WaveSimulation} from './simulation.js';
import {OceanRenderer} from './renderer.js';
import {initialQuality,IPhoneQualityGuard} from './quality.js';

const $=id=>document.getElementById(id),canvas=$('ocean'),state=new SchoolControls();
const profiles={low:{resolution:256,pixels:1000000,ratio:1,mesh:[240,144]},medium:{resolution:512,pixels:1800000,ratio:1.5,mesh:[400,240]},high:{resolution:1024,pixels:3200000,ratio:2,mesh:[640,384]}};
const mobile=navigator.maxTouchPoints>0&&Math.min(screen.width,screen.height)<850;
const requested=new URLSearchParams(location.search).get('quality');
const initial=initialQuality({requested,userAgent:navigator.userAgent,platform:navigator.platform,maxTouchPoints:navigator.maxTouchPoints,width:screen.width,height:screen.height});
let quality=initial.quality,qualitySource=initial.source;
const qualityGuard=new IPhoneQualityGuard();
let gl,simulation,renderer,ready=false,playing=true,error=null,frames=0,previous=0,lastUi=0,now=0,fps=0,frameTime=0;
let changingQuality=false,firstReady=0,lastPointer=0,sceneNotice=-10,pointer=null,contextLost=false,slowTime=0,pixelScale=1;
const look={exposure:.55,sky_rotation:25,sky_gain:1,haze:.9,foam:1,crest_threshold:.5,crest_maximum:1.1,aeration:1.25,crest_crumble:true,crest_crumble_strength:.5};
const nextPaint=()=>new Promise(resolve=>requestAnimationFrame(resolve));

function fail(reason){error=String(reason?.message||reason);ready=false;document.body.dataset.ready='false';$('loading').hidden=true;$('error-message').textContent=error;$('error').hidden=false;console.error(reason);}
function resize(){if(!gl)return;const p=profiles[quality],r=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,p.ratio)*pixelScale;
  const scale=Math.min(ratio,Math.sqrt(p.pixels/(r.width*r.height))),w=Math.max(1,Math.round(r.width*scale)),h=Math.max(1,Math.round(r.height*scale));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
}
function setPlaying(value){playing=value;$('play').textContent=playing?'Pause':'Play';$('play').setAttribute('aria-label',playing?'Pause ocean':'Play ocean');$('paused').hidden=playing;}
function setScene(index,immediate=false){state.setScene(index,now,immediate);$('scene').value=String(index);sceneNotice=now;refreshUi();}
function clearHeld(){state.held.left=false;state.held.right=false;pointer=null;}
function togglePanel(force){$('settings').hidden=force===undefined?!$('settings').hidden:!force;$('settings-toggle').setAttribute('aria-expanded',String(!$('settings').hidden));}
function toggleInfo(){const el=$('telemetry');el.hidden=!el.hidden;$('info').setAttribute('aria-pressed',String(!el.hidden));refreshUi();}
function cleanMode(){document.body.classList.toggle('clean');$('settings').hidden=true;$('settings-toggle').setAttribute('aria-expanded','false');}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else if($('stage').requestFullscreen)await $('stage').requestFullscreen();}catch(e){$('status').textContent='Fullscreen unavailable';}}

for(const [name,c] of Object.entries(controls)){
  const row=document.createElement('div');row.className='control';
  const heading=document.createElement('div');heading.className='slider-heading';
  const label=document.createElement('label');label.htmlFor=name;label.textContent=c.label+' ';
  const key=document.createElement('kbd');key.textContent=c.key.toUpperCase();label.append(key);
  const output=document.createElement('output');output.id=name+'-value';output.htmlFor=name;heading.append(label,output);
  const input=document.createElement('input');input.type='range';input.id=name;input.min='0';input.max='1000';input.step='1';input.value=String(position(name,state.parameters[name])*1000);
  input.addEventListener('input',()=>{state.setValue(name,Number(input.value)/1000,now);refreshUi();});row.append(heading,input);$('sliders').append(row);
}
$('play').onclick=()=>setPlaying(!playing);$('next-scene').onclick=()=>setScene(state.scene%4+1);$('scene').onchange=e=>setScene(Number(e.target.value));
$('reset').onclick=()=>{setScene(0);setPlaying(true);};$('settings-toggle').onclick=()=>togglePanel();$('close-settings').onclick=()=>togglePanel(false);
$('info').onclick=toggleInfo;$('fullscreen').onclick=fullscreen;$('quality').value=quality;$('quality').onchange=e=>setQuality(e.target.value);
$('retry').onclick=()=>{const url=new URL(location.href);url.searchParams.set('quality','low');location.href=url.href;};
if(!document.documentElement.requestFullscreen)$('fullscreen').hidden=true;

document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const tag=e.target.tagName;if(['INPUT','SELECT','TEXTAREA'].includes(tag))return;
  const key=e.key.toLowerCase();
  if(key==='tab'){e.preventDefault();if(!e.repeat)togglePanel();return;}
  if(key==='escape'){togglePanel(false);document.body.classList.remove('clean');clearHeld();return;}
  if(key==='arrowleft'||key==='arrowright'){e.preventDefault();state.held[key==='arrowleft'?'left':'right']=true;state.lastInput=now;state.transition=null;return;}
  if(key===' '&&tag!=='BUTTON'){e.preventDefault();if(!e.repeat)setPlaying(!playing);return;}
  if(e.repeat)return;
  const mode=Object.keys(controls).find(name=>controls[name].key===key);
  if(mode){state.selectMode(mode,now);refreshUi();}
  else if(key==='e')setScene(state.scene%4+1);
  else if(key==='i')toggleInfo();
  else if(key==='c'){$('camera-telemetry').hidden=!$('camera-telemetry').hidden;refreshUi();}
  else if(key==='h')cleanMode();
});
document.addEventListener('keyup',e=>{if(e.key==='ArrowLeft')state.held.left=false;if(e.key==='ArrowRight')state.held.right=false;});
window.addEventListener('blur',clearHeld);document.addEventListener('visibilitychange',()=>{clearHeld();previous=0;qualityGuard.reset();});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{if(!ready)return;e.preventDefault();canvas.setPointerCapture(e.pointerId);pointer={id:e.pointerId,x:e.clientX,y:e.clientY,kind:e.ctrlKey||e.button===1?'track':e.shiftKey||e.button===2?'dolly':'orbit'};state.transition=null;});
canvas.addEventListener('pointermove',e=>{lastPointer=now;document.body.classList.remove('idle');if(!pointer||pointer.id!==e.pointerId)return;const r=canvas.getBoundingClientRect();state.camera=cameraGesture(state.camera,pointer.kind,e.clientX-pointer.x,e.clientY-pointer.y,r.width,r.height);pointer.x=e.clientX;pointer.y=e.clientY;});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{pointer=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect();state.transition=null;state.camera=cameraGesture(state.camera,'dolly',-e.deltaY*(e.deltaMode===1?12:1)*.35,0,r.width,r.height);},{passive:false});
window.addEventListener('resize',resize);canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();contextLost=true;fail(new Error('The browser released its graphics context. Reload at lower quality to resume.'));});
canvas.addEventListener('webglcontextrestored',()=>location.reload());

function refreshUi(){
  for(const name of Object.keys(controls)){const text=formatValue(name,state.parameters[name]);$(name+'-value').textContent=text;$(name).value=String(Math.round(position(name,state.parameters[name])*1000));$(name).setAttribute('aria-valuetext',text);}
  const rows=Object.entries(controls).filter(([name])=>name!=='foam_amount');
  $('telemetry').replaceChildren(...rows.map(([name,c])=>{const row=document.createElement('div'),label=document.createElement('span'),value=document.createElement('b');label.textContent=c.label;value.textContent=formatValue(name,state.parameters[name]);row.append(label,value);return row;}));
  $('camera-telemetry').textContent='';
  for(const [label,value] of [['Height',state.camera.height.toFixed(1)+' meters'],['Pitch',state.camera.pitch.toFixed(1)+' degrees'],['Yaw',state.camera.yaw.toFixed(1)+' degrees'],['FOV',state.camera.fov.toFixed(1)+' degrees']]){const row=document.createElement('div'),a=document.createElement('span'),b=document.createElement('b');a.textContent=label;b.textContent=value;row.append(a,b);$('camera-telemetry').append(row);}
  const mode=state.mode,visible=mode&&now-state.lastInput<1.5;
  $('performance').style.opacity=visible?'1':'0';
  if(mode){$('control-label').textContent=controls[mode].label;$('control-value').textContent=formatValue(mode,state.parameters[mode]);const v=(state.velocity[mode]||0)/controls[mode].rate;$('left-arrow').style.opacity=String(.5+.5*clamp(-v,0,1));$('right-arrow').style.opacity=String(.5+.5*clamp(v,0,1));}
  $('scene-label').textContent=scenes[state.scene].name;$('scene-label').style.opacity=!visible&&now-sceneNotice<4?'1':'0';
  $('status').textContent=`${profiles[quality].resolution}² · ${Math.round(fps)} fps`;
}

async function warmFoam(){
  // Build the same six seconds of history as the live viewer; yield for UI.
  for(let i=0;i<48;i++){simulation.update(1/8,state.parameters);if(i%8===7)await nextPaint();}
}
async function setQuality(name,automatic=false){if(!Object.hasOwn(profiles,name)||changingQuality||!gl||contextLost)return;if(!automatic)qualitySource='manual';qualityGuard.reset();changingQuality=true;ready=false;document.body.dataset.ready='false';
  $('loading-message').textContent='Preparing '+name+' quality…';$('loading').hidden=false;$('loading').style.opacity='1';
  try{await nextPaint();simulation?.dispose();quality=name;pixelScale=1;slowTime=0;$('quality').value=name;simulation=new WaveSimulation(gl,profiles[name].resolution);renderer.setMeshResolution(...profiles[name].mesh);await warmFoam();resize();ready=true;document.body.dataset.ready='true';$('loading').hidden=true;previous=0;}catch(e){fail(e);}finally{changingQuality=false;}
}
function tick(timestamp){requestAnimationFrame(tick);if(!ready||contextLost||document.hidden){previous=0;return;}
  const seconds=timestamp*.001,frameSeconds=previous?seconds-previous:1/60,dt=Math.min(.1,frameSeconds);previous=seconds;now=seconds;
  try{state.update(dt,now);simulation.update(playing?dt:0,state.parameters);look.foam=state.parameters.foam_amount;resize();renderer.render(simulation,state.camera,look,canvas.width,canvas.height,simulation.time);frames++;frameTime=frameTime?frameTime*.95+dt*.05:dt;fps=1/frameTime;
    if(now-lastUi>.1){refreshUi();lastUi=now;}if(now-firstReady>12&&now-lastPointer>6)document.body.classList.add('idle');
    // Keep authored waves intact; reduce display pixels if sustained GPU load is high.
    if(playing&&dt>.045&&quality!=='low')slowTime+=dt;else slowTime=Math.max(0,slowTime-dt);
    if(slowTime>5&&pixelScale>.65){pixelScale=Math.max(.65,pixelScale*.85);slowTime=0;}
    const automaticQuality=qualityGuard.sample(frameSeconds,{quality,source:qualitySource,playing});
    if(automaticQuality)setQuality(automaticQuality,true);
  }catch(e){fail(e);}
}
window.__school={getState:()=>({ready,playing,parameters:{...state.parameters},camera:{...state.camera},quality,qualitySource,resolution:simulation?.resolution||0,time:simulation?.time||0,frames,scene:state.scene,renderer:'WebGL 2',error}),setScene:index=>setScene(index),setQuality};
async function start(){
  try{gl=canvas.getContext('webgl2',{alpha:false,antialias:true,depth:true,powerPreference:mobile?'default':'high-performance',preserveDrawingBuffer:false});
    if(!gl)throw new Error('This viewer needs WebGL 2. Try a current Safari, Chrome, Edge, or Firefox with graphics acceleration enabled.');
    if(!gl.getExtension('EXT_color_buffer_float')&&!gl.getExtension('EXT_color_buffer_half_float'))throw new Error('This device cannot render the floating-point ocean. Try a current browser with graphics acceleration enabled.');
    renderer=await OceanRenderer.create(gl,{gridWidth:profiles[quality].mesh[0],gridHeight:profiles[quality].mesh[1]});simulation=new WaveSimulation(gl,profiles[quality].resolution);$('loading-message').textContent='Building six seconds of whitecap history…';await warmFoam();resize();ready=true;firstReady=performance.now()*.001;now=firstReady;document.body.dataset.ready='true';refreshUi();$('loading').style.opacity='0';setTimeout(()=>{$('loading').hidden=true;},750);requestAnimationFrame(tick);
  }catch(e){fail(e);}
}
start();
