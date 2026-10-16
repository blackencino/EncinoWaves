// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// School's authored cameras and wall-clock performance controls.
export const FOV = 2 * Math.atan(14.85 / 40) * 180 / Math.PI;
const rad = Math.PI / 180;
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function basis(c) {
  const y=c.yaw*rad, p=c.pitch*rad;
  const forward=[Math.sin(y)*Math.cos(p),Math.cos(y)*Math.cos(p),Math.sin(p)];
  const right=[Math.cos(y),-Math.sin(y),0];
  const up=[right[1]*forward[2],-right[0]*forward[2],right[0]*forward[1]-right[1]*forward[0]];
  return {forward,right,up};
}
export function authoredCamera(height,pitch,yaw) {
  const c={height,pitch,yaw,fov:FOV,center_of_interest:-height/Math.sin(pitch*rad)};
  const f=basis(c).forward;
  return {...c,x:-c.center_of_interest*f[0],y:-c.center_of_interest*f[1]};
}
export const defaults={wind_speed:17,depth:100,fetch_km:300,swell:.35,foam_amount:1,
  domain:1024,pinch:.75,seed:54321,gravity:9.81,surface_tension:.074,density:1000,gamma:3.3,
  amplitude_gain:1,trough_damping:1};
export const scenes=[
  {name:'Opening ocean',parameters:{...defaults},camera:authoredCamera(145.2,-18.1,-139.9)},
  {name:'NOAA severe sea state',parameters:{...defaults,wind_speed:50*1852/3600,fetch_km:2000,depth:1000,swell:0},camera:authoredCamera(85,-13,-52)},
  {name:'Shallow choppy bay',parameters:{...defaults,wind_speed:9,fetch_km:15,depth:3,swell:.05},camera:authoredCamera(24,-12,-122)},
  {name:'Calm deep seas',parameters:{...defaults,wind_speed:4*1852/3600,fetch_km:300,depth:1000,swell:0},camera:authoredCamera(45.1,-4.7,55.9)},
  {name:'Choppy shallow water',parameters:{...defaults,wind_speed:40*1852/3600,fetch_km:100,depth:5,swell:-.9},camera:authoredCamera(35,-11,-155)},
];
export const controls={
  wind_speed:{key:'w',label:'Wind Speed',low:1,high:500,log:true,rate:.16},
  depth:{key:'d',label:'Ocean Depth',low:.25,high:1000,log:true,rate:.17},
  fetch_km:{key:'f',label:'Fetch',low:1,high:5000,log:true,rate:.17},
  swell:{key:'s',label:'Swell',low:-1,high:2,log:false,rate:.22},
  foam_amount:{key:'m',label:'Foam',low:0,high:2,log:false,rate:.15},
};
export function position(name,value) { const c=controls[name]; return clamp(c.log?Math.log(value/c.low)/Math.log(c.high/c.low):(value-c.low)/(c.high-c.low),0,1); }
export function valueAt(name,value) { const c=controls[name],p=clamp(value,0,1); return c.log?c.low*Math.exp(p*Math.log(c.high/c.low)):c.low+p*(c.high-c.low); }
export function formatValue(name,value) {
  if(name==='wind_speed') return `${(value*3600/1852).toFixed(1)} knots`;
  if(name==='depth') return `${value.toFixed(1)} meters`;
  if(name==='fetch_km') return `${value.toFixed(0)} kilometers`;
  if(name==='foam_amount') return `${(100*value).toFixed(0)}%`;
  return value.toFixed(2);
}
function pivot(c) {const f=basis(c).forward;return [c.x+f[0]*c.center_of_interest,c.y+f[1]*c.center_of_interest,c.height+f[2]*c.center_of_interest];}
function fromPivot(c,p) {const f=basis(c).forward;return {...c,x:p[0]-f[0]*c.center_of_interest,y:p[1]-f[1]*c.center_of_interest,height:p[2]-f[2]*c.center_of_interest};}
export function cameraGesture(camera,kind,dx,dy,width,height) {
  const c={...camera},p=pivot(c);
  if(kind==='orbit') {c.yaw+=400*dx/width;c.pitch=clamp(c.pitch-400*dy/height,-89,-.5);return fromPivot(c,p);}
  if(kind==='dolly') {c.center_of_interest=clamp(c.center_of_interest*Math.exp(-5*dx/width),8,15000);return fromPivot(c,p);}
  const {right,up}=basis(c),s=2*c.center_of_interest*Math.tan(c.fov*rad/2);
  const delta=right.map((r,i)=>-s*dx/width*r+s*dy/height*up[i]);
  c.x+=delta[0];c.y+=delta[1];c.height=Math.max(2,c.height+delta[2]);return c;
}
export function interpolateCamera(a,b,t) {
  if(t<=0)return {...a};
  if(t>=1)return {...b};
  const pa=pivot(a),pb=pivot(b),yawDelta=((b.yaw-a.yaw+180)%360+360)%360-180;
  const c={...a,yaw:a.yaw+t*yawDelta,pitch:a.pitch+t*(b.pitch-a.pitch),fov:a.fov+t*(b.fov-a.fov),
    center_of_interest:Math.exp((1-t)*Math.log(a.center_of_interest)+t*Math.log(b.center_of_interest))};
  return fromPivot(c,pa.map((v,i)=>v+t*(pb[i]-v)));
}
export class SchoolControls {
  constructor(){this.parameters={...defaults};this.camera={...scenes[0].camera};this.mode=null;this.velocity={};this.held={left:false,right:false};this.scene=0;this.transition=null;this.lastInput=0;}
  selectMode(name,now){this.mode=this.mode===name?null:name;this.lastInput=now;this.transition=null;}
  setValue(name,value,now){this.parameters[name]=valueAt(name,value);this.velocity[name]=0;this.mode=name;this.lastInput=now;this.transition=null;}
  setScene(index,now,immediate=false){const s=scenes[index];if(!s)return;this.scene=index;this.mode=null;this.velocity={};this.lastInput=now;
    if(immediate){this.parameters={...s.parameters};this.camera={...s.camera};this.transition=null;}
    else this.transition={at:now,parameters:{...this.parameters},camera:{...this.camera},target:s};
  }
  update(dt,now){
    if(this.transition){const a=this.transition,t=clamp((now-a.at)/3,0,1),q=t*t*t*(10+t*(-15+6*t));
      for(const name of Object.keys(controls))this.parameters[name]=valueAt(name,(1-q)*position(name,a.parameters[name])+q*position(name,a.target.parameters[name]));
      this.camera=interpolateCamera(a.camera,a.target.camera,q);if(t===1){this.parameters={...a.target.parameters};this.transition=null;}return;
    }
    const direction=Number(this.held.right)-Number(this.held.left);
    for(const [name,c] of Object.entries(controls)){
      const target=name===this.mode?direction*c.rate:0,old=this.velocity[name]||0;
      if(!target&&Math.abs(old)<1e-8){this.velocity[name]=0;continue;}
      const tau=target?.3:.45,decay=Math.exp(-dt/tau);
      let p=position(name,this.parameters[name])+target*dt+(old-target)*tau*(1-decay);
      this.velocity[name]=(p<0||p>1)?0:target+(old-target)*decay;
      this.parameters[name]=valueAt(name,p);
    }
    if(direction&&this.mode)this.lastInput=now;
  }
}
