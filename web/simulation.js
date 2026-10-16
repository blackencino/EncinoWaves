// Copyright 2015–2026 Christopher Jon Horvath. Apache-2.0.
// The School spectral ocean, evaluated locally on WebGL 2. The six real
// unnormalised inverse FFTs are packed into three complex transforms.
export const DEFAULT_PARAMETERS = Object.freeze({
  domain: 1024, wind_speed: 17, fetch_km: 300, depth: 100, swell: .35,
  spreading: 'hasselmann', spectrum: 'tma', dispersion: 'capillary',
  gravity: 9.81, surface_tension: .074, density: 1000, pinch: .75,
  amplitude_gain: 1, phillips_amplitude: 100, seed: 54321, gamma: 3.3, wind_direction: 0,
  trough_damping: .5, trough_smoothing_length: .1,
});

const VERTEX = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const float PI = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;
vec2 multiply(vec2 a, vec2 b) { return vec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
`;

// Float64 JS seed addressing deliberately matches model.random_variates.
// Hashing rounded float32 wavenumbers in a shader changes the realization.
export function randomVariates(kx, ky, seed) {
  let s = (Math.imul(Math.trunc(kx*10000),73856093) ^
    Math.imul(Math.trunc(ky*10000),19349663) ^ Math.imul(seed,83492791)) >>> 0;
  const values = [];
  for (let i=0;i<4;i++) {
    s = (Math.imul(s,747796405)+2891336453) >>> 0;
    let w = Math.imul((s >>> ((s >>> 28)+4)) ^ s,277803737) >>> 0;
    w = ((w >>> 22) ^ w) >>> 0;
    values.push((w+.5)/4294967296);
  }
  const radius = Math.sqrt(-2*Math.log(values[0]));
  return [radius*Math.cos(2*Math.PI*values[1]),radius*Math.sin(2*Math.PI*values[1]),
    2*Math.PI*values[2],2*Math.PI*values[3]];
}

// The early Tweak implementation used a sequential POSIX drand48 stream and
// float32 polar Box–Muller draws (including its cached-pair return order).
// A split 24-bit state keeps the 48-bit integer multiply exact in JavaScript.
export function legacyGaussianGenerator(seed) {
  let hi=(seed>>>8),lo=((seed&255)*65536)+0x330e,cached=null;
  const f=Math.fround,base=16777216,a=0xece66d,b=0x5de;
  function uniform() {
    const product=lo*a+11,carry=Math.floor(product/base);
    hi=(hi*a+lo*b+carry)%base;lo=product%base;
    return f((hi*base+lo)/281474976710656);
  }
  return () => {
    if(cached!==null){const result=cached;cached=null;return result;}
    let x,y,r;
    do {x=f(-1+f(2*uniform()));y=f(-1+f(2*uniform()));r=f(f(x*x)+f(y*y));} while(r>1||r===0);
    const scale=f(Math.sqrt(f(f(-2*f(Math.log(r)))/r)));
    cached=f(x*scale);return f(y*scale);
  };
}

export function legacyNoiseData(resolution,seed) {
  const width=resolution/2+1,data=new Float32Array(resolution*width*4),gaussian=legacyGaussianGenerator(seed);
  for(let y=0;y<resolution;y++)for(let x=0;x<width;x++) {
    if(x===0&&y===0)continue;
    const i=(y*width+x)*4,re=gaussian(),im=gaussian(),negativeRe=gaussian(),negativeIm=gaussian();
    // Our propagation is A exp(-iwt)+B exp(+iwt): A=conj(h0(-k)), B=h0(k).
    data.set([negativeRe,-negativeIm,re,im],i);
  }
  return data;
}

const INITIAL = HEAD + `
uniform sampler2D uNoise;
uniform int uN, uSpectrum, uSpreading, uDispersion;
uniform float uDomain, uWind, uFetch, uDepth, uSwell, uGravity, uTension, uDensity, uGamma, uPhillipsAmplitude;
layout(location=0) out vec4 coefficients;
layout(location=1) out vec4 frequency;
// Lanczos log-gamma, evaluated on positive arguments only.
float logGamma(float z) {
  z -= 1.0;
  float x = .99999999999980993;
  x += 676.5203681218851/(z+1.0);
  x -= 1259.1392167224028/(z+2.0);
  x += 771.32342877765313/(z+3.0);
  x -= 176.61502916214059/(z+4.0);
  x += 12.507343278686905/(z+5.0);
  x -= .13857109526572012/(z+6.0);
  x += .000009984369578019572/(z+7.0);
  x += .00000015056327351493116/(z+8.0);
  float t = z+7.5;
  return .9189385332046727+(z+.5)*log(t)-t+log(x);
}
float product(float angle,float beta,float bias) {
  float base;
  if(uSpreading==3) {
    float e=exp(-2.0*abs(beta*angle)); base=4.0*e/((1.0+e)*(1.0+e));
  } else { base=abs(angle)<PI*.5?pow(cos(angle),2.0):0.0; }
  return base*pow(max(0.0,sin((PI-abs(angle))*.5)),2.0*bias);
}
float directional(float angle,float ratio,float peak) {
  float bias=16.0*tanh(1.0/ratio)*pow(max(0.0,uSwell),2.0);
  float value;
  if(uSpreading<2) {
    float uc=uWind*peak/uGravity;
    float shape=uSpreading==0 ? (ratio<=1.0?6.97*pow(ratio,4.06):9.77*pow(ratio,-2.33-1.45*(uc-1.17)))
      : 11.5*pow(uc,-2.5)*(ratio<=1.0?pow(ratio,5.0):pow(ratio,-2.5));
    shape+=bias;
    float q=exp(logGamma(shape+1.0)-logGamma(shape+.5)-log(2.0*sqrt(PI)));
    value=q*pow(max(0.0,sin((PI-abs(angle))*.5)),2.0*shape);
  } else {
    float beta=ratio<.95?2.61*pow(ratio,1.3):(ratio<1.6?2.28*pow(ratio,-1.3):
      pow(10.0,-.4+.8393*exp(-.567*log(pow(max(ratio,1.6),2.0)))));
    float extent=uSpreading==3?PI:PI*.5;
    float denom=0.0;
    // Midpoint quadrature for optional non-School spreading models.
    for(int i=0;i<64;i++) denom+=product((float(i)+.5)*extent/64.0,beta,bias);
    denom*=2.0*extent/64.0;
    value=product(angle,beta,bias)/max(1e-30,denom);
  }
  return uSwell<0.0?(1.0+uSwell)*value-uSwell/TAU:value;
}
void main() {
  ivec2 pos=ivec2(gl_FragCoord.xy);
  vec2 wave=vec2(float(pos.x),float(pos.y<=uN/2?pos.y:pos.y-uN))*(TAU/uDomain);
  float k=length(wave);
  if(k==0.0) { coefficients=vec4(0); frequency=vec4(0); return; }
  if(uSpectrum==3) {
    // Direct wavenumber Phillips spectrum: no omega Jacobian, depth envelope,
    // fetch, short-wave damping or modern normalized directional spreading.
    float lengthScale=uWind*uWind/uGravity;
    float directionalPower=wave.x*wave.x/(k*k);
    float power=directionalPower*exp(-1.0/pow(k*lengthScale,2.0))/pow(k,4.0);
    float coefficient=uPhillipsAmplitude/(uDomain*uDomain)*sqrt(max(0.0,power*.5));
    coefficients=texelFetch(uNoise,pos,0)*coefficient;
    frequency=vec4(sqrt(uGravity*k*tanh(k*uDepth)),0,0,0);
    return;
  }
  float th=uDispersion==0?1.0:tanh(k*uDepth);
  float tension=uDispersion==2?k*k*uTension/uDensity:0.0;
  float omega=sqrt(k*(uGravity+tension)*th);
  float derivative=((uGravity+3.0*tension)*th+k*uDepth*(uGravity+tension)*(1.0-th*th))/(2.0*omega);
  float fetch=uGravity*uFetch*1000.0/(uWind*uWind);
  float peak=TAU*3.5*uGravity/uWind*pow(fetch,-.33);
  float w=max(omega,1e-12), spectrum;
  if(uSpectrum==0) {
    float wm=.87*uGravity/uWind;
    spectrum=exp(log(.0081*uGravity*uGravity)-5.0*log(w)-1.291*pow(wm/w,4.0));
  } else {
    float sigma=w<=peak?.07:.09;
    float sharpen=log(uGamma)*exp(-.5*pow((w-peak)/(sigma*peak),2.0));
    spectrum=exp(log(.076*pow(fetch,-.22)*uGravity*uGravity)-5.0*log(w)-1.25*pow(peak/w,4.0)+sharpen);
    if(uSpectrum==2) spectrum/=1.0+exp(-3.6*(w*sqrt(uDepth/uGravity)-1.125));
  }
  float energy=spectrum*pow(TAU/uDomain,2.0)*derivative/k;
  float theta=atan(-wave.y,wave.x);
  float opposite=theta<0.0?theta+PI:theta-PI;
  vec4 noise=texelFetch(uNoise,pos,0);
  coefficients=vec4(noise.xy*sqrt(max(0.0,2.0*energy*directional(theta,w/peak,peak))),
                    noise.zw*sqrt(max(0.0,2.0*energy*directional(opposite,w/peak,peak))));
  frequency=vec4(omega,0,0,0);
}`;

const PHASE = HEAD + `
uniform sampler2D uPhase,uFrequency;
uniform float uDt;
out vec4 result;
void main() {
  ivec2 pos=ivec2(gl_FragCoord.xy);
  float phase=texelFetch(uPhase,pos,0).r+texelFetch(uFrequency,pos,0).r*uDt;
  result=vec4(mod(phase,TAU),0,0,0);
}`;

const SPECTRAL = HEAD + `
uniform sampler2D uInitial,uPhase;
uniform int uN;
uniform float uDomain;
layout(location=0) out vec4 fieldsA;
layout(location=1) out vec4 fieldsB;
void fields(ivec2 p,out vec2 h,out vec2 dx,out vec2 dy,out vec2 dxx,out vec2 dyy,out vec2 dxy) {
  vec4 initial=texelFetch(uInitial,p,0);
  float phase=texelFetch(uPhase,p,0).r;
  vec2 rotation=vec2(cos(phase),sin(phase));
  h=multiply(initial.xy,rotation*vec2(1,-1))+multiply(initial.zw,rotation);
  vec2 k=vec2(float(p.x),float(p.y<=uN/2?p.y:p.y-uN))*(TAU/uDomain);
  float norm=max(length(k),1e-12);
  dx=vec2(h.y,-h.x)*(k.x/norm);
  dy=vec2(h.y,-h.x)*(k.y/norm);
  dxx=h*(k.x*k.x/norm); dyy=h*(k.y*k.y/norm); dxy=h*(k.x*k.y/norm);
}
vec2 pack(vec2 a,vec2 b) { return a+vec2(-b.y,b.x); }
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  bool mirrored=p.x>uN/2;
  if(mirrored) p=ivec2(uN-p.x,(uN-p.y)%uN);
  vec2 h,dx,dy,dxx,dyy,dxy;
  fields(p,h,dx,dy,dxx,dyy,dxy);
  // irfft2 discards the anti-Hermitian part of both boundary columns.
  if(p.x==0||p.x==uN/2) {
    vec2 h2,dx2,dy2,dxx2,dyy2,dxy2;
    fields(ivec2(p.x,(uN-p.y)%uN),h2,dx2,dy2,dxx2,dyy2,dxy2);
    vec2 conjugate=vec2(1,-1);
    h=.5*(h+h2*conjugate); dx=.5*(dx+dx2*conjugate); dy=.5*(dy+dy2*conjugate);
    dxx=.5*(dxx+dxx2*conjugate); dyy=.5*(dyy+dyy2*conjugate); dxy=.5*(dxy+dxy2*conjugate);
  }
  if(mirrored) { h.y=-h.y;dx.y=-dx.y;dy.y=-dy.y;dxx.y=-dxx.y;dyy.y=-dyy.y;dxy.y=-dxy.y; }
  fieldsA=vec4(pack(h,dx),pack(dy,dxx)); fieldsB=vec4(pack(dyy,dxy),0,0);
}`;

const FFT = HEAD + `
uniform sampler2D uA,uB,uButterfly;
uniform int uStage,uAxis;
layout(location=0) out vec4 resultA;
layout(location=1) out vec4 resultB;
vec4 butterfly(sampler2D source,ivec2 a,ivec2 b,vec2 twiddle) {
  vec4 even=texelFetch(source,a,0),odd=texelFetch(source,b,0);
  return even+vec4(multiply(odd.xy,twiddle),multiply(odd.zw,twiddle));
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  int index=uAxis==0?p.x:p.y;
  vec4 lookup=texelFetch(uButterfly,ivec2(index,uStage),0);
  ivec2 a=uAxis==0?ivec2(int(lookup.z),p.y):ivec2(p.x,int(lookup.z));
  ivec2 b=uAxis==0?ivec2(int(lookup.w),p.y):ivec2(p.x,int(lookup.w));
  resultA=butterfly(uA,a,b,lookup.xy); resultB=butterfly(uB,a,b,lookup.xy);
}`;

const DISPLACEMENT = HEAD + `
uniform sampler2D uA,uB;
uniform float uPinch,uGain;
out vec4 result;
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec4 a=texelFetch(uA,p,0),b=texelFetch(uB,p,0);
  float xx=1.0-uPinch*a.w,yy=1.0-uPinch*b.x,xy=-uPinch*b.y;
  float crest=-.5*(xx+yy-sqrt((xx-yy)*(xx-yy)+4.0*xy*xy));
  result=vec4(-uPinch*a.y,-uPinch*a.z,uGain*a.x,crest);
}`;

const NORMAL = HEAD + `
uniform sampler2D uA;
uniform int uN;
uniform float uSpacing,uPinch,uGain;
out vec4 result;
vec3 point(ivec2 p) {
  vec4 a=texelFetch(uA,(p+uN)%uN,0);
  return vec3(-uPinch*a.y,-uPinch*a.z,uGain*a.x);
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec3 tx=point(p+ivec2(1,0))-point(p-ivec2(1,0))+vec3(2.0*uSpacing,0,0);
  vec3 ty=point(p+ivec2(0,1))-point(p-ivec2(0,1))+vec3(0,2.0*uSpacing,0);
  result=vec4(normalize(cross(tx,ty)),0);
}`;

const FOAM = HEAD + `
uniform sampler2D uFoam,uDisplacement;
uniform int uN,uSeed;
uniform float uDt,uTime,uDomain,uEmission,uBreakup,uThreshold,uWidth;
out vec4 result;
float hash(ivec2 p,int salt) {
  uint s=uint(p.x)*73856093u^uint(p.y)*19349663u^uint(uSeed+salt)*83492791u;
  s=s*747796405u+2891336453u;
  uint word=((s>>((s>>28u)+4u))^s)*277803737u;
  return float((word>>22u)^word)/4294967296.0*2.0-1.0;
}
float noise(vec2 uv,int cells,int salt) {
  vec2 p=uv*float(cells),f=fract(p);
  ivec2 lo=ivec2(floor(p)),hi=(lo+1)%cells; lo=lo%cells;
  vec2 w=f*f*f*(f*(f*6.0-15.0)+10.0);
  return mix(mix(hash(lo,salt),hash(ivec2(hi.x,lo.y),salt),w.x),
    mix(hash(ivec2(lo.x,hi.y),salt),hash(hi,salt),w.x),w.y);
}
vec3 density(ivec2 p) { return texelFetch(uFoam,(p+uN)%uN,0).rgb; }
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec2 uv=vec2(p)/float(uN);
  vec3 center=density(p);
  vec3 neighbors=density(p+ivec2(1,0))+density(p-ivec2(1,0))+density(p+ivec2(0,1))+density(p-ivec2(0,1));
  // Stable discrete diffusion approximation at preview spacing, preserving mass.
  vec3 weight=.25*(1.0-exp(-4.0*uDt*.56*vec3(1,3,7)/pow(uDomain/float(uN),2.0)));
  vec3 aged=max(vec3(0),center+weight*(neighbors-4.0*center));
  float b=log(2.0)/6.0+.29,c=log(2.0)/12.0;
  float eb=exp(-b*uDt),ec=exp(-c*uDt);
  aged=vec3(aged.r*exp(-log(2.0)*uDt/1.5),aged.g*eb,
    aged.b*ec+aged.g*.29*(ec-eb)/(b-c));
  int base=max(1,int(round(uDomain/24.0)));
  float value=0.0,grain=0.0,amplitude=1.0/1.875;
  for(int octave=0;octave<4;octave++) {
    int cells=min(uN/2,base*(1<<octave));
    float rate=octave==0?1.0:(octave==1?1.37:(octave==2?1.91:2.63));
    float cosine=noise(uv,cells,octave*2+11),sine=noise(uv,cells,octave*2+12);
    value+=amplitude*(cosine*cos(uTime*.12*rate)+sine*sin(uTime*.12*rate));
    if(octave==3) grain=cosine;
    amplitude*=.5;
  }
  float broken=smoothstep(0.0,.35,value+.05);
  float stretch=-texelFetch(uDisplacement,p,0).w;
  float source=smoothstep(0.0,uWidth,uThreshold-stretch);
  float emission=source*mix(1.0,broken,uBreakup)*uEmission*uDt;
  result=vec4(aged+vec3(emission*.8,emission*.2,0),clamp(.5+.5*grain,0.0,1.0));
}`;

const REDUCE = HEAD + `
uniform sampler2D uSource;
uniform int uFirst;
out vec4 result;
vec4 value(ivec2 p) {
  vec4 v=texelFetch(uSource,p,0);
  return uFirst==1?vec4(v.z*v.z,v.w,v.w*v.w,abs(v.z)):v;
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy)*2;
  vec4 a=value(p),b=value(p+ivec2(1,0)),c=value(p+ivec2(0,1)),d=value(p+ivec2(1,1));
  result=vec4((a.xyz+b.xyz+c.xyz+d.xyz)*.25,max(max(a.w,b.w),max(c.w,d.w)));
}`;

function program(gl, fragment) {
  const shaders = [gl.VERTEX_SHADER,gl.FRAGMENT_SHADER].map((kind,index)=>{
    const shader=gl.createShader(kind);
    gl.shaderSource(shader,index?fragment:VERTEX); gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) {
      const message=gl.getShaderInfoLog(shader);gl.deleteShader(shader);
      throw new Error('Ocean GPU shader: '+message);
    }
    return shader;
  });
  const p=gl.createProgram();shaders.forEach(s=>gl.attachShader(p,s));gl.linkProgram(p);
  shaders.forEach(s=>gl.deleteShader(s));
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error('Ocean GPU program: '+gl.getProgramInfoLog(p));
  const uniforms={};
  for(let i=0;i<gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);i++) {
    const info=gl.getActiveUniform(p,i);uniforms[info.name]={location:gl.getUniformLocation(p,info.name),type:info.type};
  }
  return {program:p,uniforms};
}

export class WaveSimulation {
  constructor(gl,resolution=512) {
    if(resolution<16 || resolution>1024 || (resolution&(resolution-1))) throw new Error('Wave resolution must be a power of two from 16 to 1024.');
    this.gl=gl;this.resolution=resolution;this.domain=1024;this.time=0;
    this.parameters={...DEFAULT_PARAMETERS};this.textures=[];this.framebuffers=[];
    const full=gl.getExtension('EXT_color_buffer_float');
    const half=full||gl.getExtension('EXT_color_buffer_half_float');
    if(!half) throw new Error('This browser needs WebGL 2 floating-point render targets for the ocean.');
    this.computeFormat=full?gl.RGBA32F:gl.RGBA16F;
    this.fullPrecision=Boolean(full);
    this.vao=gl.createVertexArray();this.programs={};
    for(const [key,source] of Object.entries({initial:INITIAL,phase:PHASE,spectral:SPECTRAL,fft:FFT,displacement:DISPLACEMENT,normal:NORMAL,foam:FOAM,reduce:REDUCE})) this.programs[key]=program(gl,source);
    const n=resolution,w=n/2+1;
    // The same tiny lookup row is reused by every FFT row/column. Computing
    // trigonometry once also avoids millions of repeated mobile GPU sin/cos.
    const butterflyData=new Float32Array(n*Math.log2(n)*4);
    for(let stage=0,size=2;size<=n;stage++,size*=2) for(let i=0;i<n;i++) {
      const offset=(stage*n+i)*4,angle=2*Math.PI*(i%size)/size;
      const even=Math.floor(i/size)*(size/2)+i%(size/2);
      butterflyData.set([Math.cos(angle),Math.sin(angle),even,even+n/2],offset);
    }
    this.butterfly=gl.createTexture();this.textures.push(this.butterfly);gl.bindTexture(gl.TEXTURE_2D,this.butterfly);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,n,Math.log2(n),0,gl.RGBA,gl.FLOAT,butterflyData);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    this.initial=this.target(w,n,2,this.computeFormat);
    this.phase=[this.target(w,n,1,this.computeFormat),this.target(w,n,1,this.computeFormat)];this.phaseIndex=0;
    this.fft=[this.target(n,n,2,this.computeFormat),this.target(n,n,2,this.computeFormat)];
    this.displacementTarget=this.target(n,n,1,gl.RGBA16F,true);
    this.normalTarget=this.target(n,n,1,gl.RGBA16F,true);
    this.foamTargets=[this.target(n,n,1,gl.RGBA16F,true),this.target(n,n,1,gl.RGBA16F,true)];this.foamIndex=0;
    this.displacement=this.displacementTarget.textures[0];this.normal=this.normalTarget.textures[0];this.foam=this.foamTargets[0].textures[0];
    this.reductions=[];
    for(let size=n/2;size>=1;size/=2) this.reductions.push(this.target(size,size,1,this.computeFormat));
    this.heightStd=1;this.crestStd=.25;this.crestMean=-.75;
    this.bigHeight=10;this.crestGain=2;this.crestBias=1.5;
    this.shadingStatistics={big_height:10,crest_gain:2,crest_bias:1.5};
    this.seedKey='';this.spectrumKey='';this.statisticsKey='';
    this.noise=null;this.disposed=false;this.update(0);
  }

  target(width,height,count,format,mips=false) {
    const gl=this.gl,framebuffer=gl.createFramebuffer(),textures=[];
    this.framebuffers.push(framebuffer);gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
    for(let i=0;i<count;i++) {
      const texture=gl.createTexture();this.textures.push(texture);textures.push(texture);
      gl.bindTexture(gl.TEXTURE_2D,texture);
      gl.texStorage2D(gl.TEXTURE_2D,mips?1+Math.floor(Math.log2(Math.max(width,height))):1,format,width,height);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,mips?gl.LINEAR_MIPMAP_LINEAR:gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,mips?gl.LINEAR:gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.REPEAT);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.TEXTURE_2D,texture,0);
    }
    gl.drawBuffers(textures.map((_,i)=>gl.COLOR_ATTACHMENT0+i));
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error('The browser could not allocate the floating-point ocean buffers.');
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    return {framebuffer,textures,width,height};
  }

  draw(name,target,uniforms={},textures={}) {
    const gl=this.gl,p=this.programs[name];
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.framebuffer);gl.viewport(0,0,target.width,target.height);
    gl.useProgram(p.program);gl.bindVertexArray(this.vao);
    let unit=0;
    for(const [key,value] of Object.entries(textures)) {
      const u=p.uniforms[key];if(!u) continue;
      gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,value);gl.uniform1i(u.location,unit++);
    }
    for(const [key,value] of Object.entries(uniforms)) {
      const u=p.uniforms[key];if(!u) continue;
      if(u.type===gl.INT||u.type===gl.BOOL) gl.uniform1i(u.location,value);else gl.uniform1f(u.location,value);
    }
    gl.drawArrays(gl.TRIANGLES,0,3);
  }

  makeNoise() {
    const gl=this.gl,n=this.resolution,width=n/2+1,legacy=this.parameters.spectrum==='phillips';
    const data=legacy?legacyNoiseData(n,this.parameters.seed):new Float32Array(width*n*4),dk=2*Math.PI/this.domain;
    if(!legacy)for(let y=0;y<n;y++) for(let x=0;x<width;x++) {
      const [a,b,pa,pb]=randomVariates(x*dk,(y<=n/2?y:y-n)*dk,this.parameters.seed),i=(y*width+x)*4;
      data[i]=a*Math.cos(pa);data[i+1]=-a*Math.sin(pa);data[i+2]=b*Math.cos(pb);data[i+3]=-b*Math.sin(pb);
    }
    if(this.noise) {gl.deleteTexture(this.noise);this.textures=this.textures.filter(t=>t!==this.noise);}
    this.noise=gl.createTexture();this.textures.push(this.noise);gl.bindTexture(gl.TEXTURE_2D,this.noise);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,width,n,0,gl.RGBA,gl.FLOAT,data);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    for(const target of [...this.phase,...this.foamTargets]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER,target.framebuffer);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    }
  }

  update(dt,parameters={}) {
    if(this.disposed) return;
    if(!Number.isFinite(dt)||dt<0) throw new Error('Ocean time step must be finite and nonnegative.');
    const gl=this.gl,n=this.resolution;
    this.parameters={...this.parameters,...parameters};const p=this.parameters;this.domain=p.domain;
    gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);gl.disable(gl.SCISSOR_TEST);gl.colorMask(true,true,true,true);
    const seedKey=[p.domain,p.seed,p.spectrum==='phillips'].join('/');
    if(seedKey!==this.seedKey) {this.makeNoise();this.seedKey=seedKey;this.spectrumKey='';this.statisticsKey='';}
    const spectralKeys=['wind_speed','fetch_km','depth','swell','spectrum','spreading','dispersion','gravity','surface_tension','density','gamma','phillips_amplitude'];
    const key=spectralKeys.map(k=>p[k]).join('/');
    if(key!==this.spectrumKey) {
      this.draw('initial',this.initial,{uN:n,uDomain:p.domain,uWind:p.wind_speed,uFetch:p.fetch_km,uDepth:p.depth,uSwell:p.swell,
        uGravity:p.gravity,uTension:p.surface_tension,uDensity:p.density,uGamma:p.gamma,uPhillipsAmplitude:p.phillips_amplitude,
        uSpectrum:Math.max(0,['pm','jonswap','tma','phillips'].indexOf(p.spectrum)),
        uSpreading:Math.max(0,['hasselmann','mitsuyasu','cosine_squared','donelan_banner'].indexOf(p.spreading)),
        uDispersion:Math.max(0,['deep','finite','capillary'].indexOf(p.dispersion))},{uNoise:this.noise});
      this.spectrumKey=key;
    }
    if(dt>0) {
      const next=1-this.phaseIndex;
      this.draw('phase',this.phase[next],{uDt:dt},{uPhase:this.phase[this.phaseIndex].textures[0],uFrequency:this.initial.textures[1]});
      this.phaseIndex=next;this.time+=dt;
    }
    this.draw('spectral',this.fft[0],{uN:n,uDomain:p.domain},{uInitial:this.initial.textures[0],uPhase:this.phase[this.phaseIndex].textures[0]});
    let current=0;
    for(let axis=0;axis<2;axis++) for(let size=2,stage=0;size<=n;size*=2,stage++) {
      this.draw('fft',this.fft[1-current],{uStage:stage,uAxis:axis},
        {uA:this.fft[current].textures[0],uB:this.fft[current].textures[1],uButterfly:this.butterfly});current=1-current;
    }
    this.draw('displacement',this.displacementTarget,{uPinch:p.pinch,uGain:p.amplitude_gain},{uA:this.fft[current].textures[0],uB:this.fft[current].textures[1]});
    this.draw('normal',this.normalTarget,{uN:n,uSpacing:p.domain/n,uPinch:p.pinch,uGain:p.amplitude_gain},{uA:this.fft[current].textures[0]});
    if(p.spectrum!=='phillips') {
      const foamNext=1-this.foamIndex;
      this.draw('foam',this.foamTargets[foamNext],{uN:n,uSeed:p.seed|0,uDomain:p.domain,uDt:Math.min(dt,.25),uTime:this.time-dt*.5,
        uEmission:p.foam_enabled===false?0:(p.foam_emission??1.2),uBreakup:p.foam_breakup??1,
        uThreshold:p.compression_threshold??.75,uWidth:p.compression_width??.15},
        {uFoam:this.foam,uDisplacement:this.displacement});
      this.foamIndex=foamNext;this.foam=this.foamTargets[foamNext].textures[0];
    }
    const statisticsKey=key+'/'+p.pinch+'/'+p.amplitude_gain;
    if(statisticsKey!==this.statisticsKey) {this.updateStatistics();this.statisticsKey=statisticsKey;}
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    for(const texture of [this.displacement,this.normal,this.foam]) {gl.bindTexture(gl.TEXTURE_2D,texture);gl.generateMipmap(gl.TEXTURE_2D);}
    gl.bindVertexArray(null);
  }

  updateStatistics() {
    const gl=this.gl;let source=this.displacement,first=1;
    for(const target of this.reductions) {this.draw('reduce',target,{uFirst:first},{uSource:source});source=target.textures[0];first=0;}
    if(!this.fullPrecision) return;
    const values=new Float32Array(4);gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,values);
    this.heightStd=Math.sqrt(Math.max(0,values[0]));this.crestMean=values[1];
    this.crestStd=Math.sqrt(Math.max(0,values[2]-values[1]*values[1]));
    this.bigHeight=Math.max(.001,1.5*values[3]);this.crestGain=1/Math.max(1e-8,2*this.crestStd);this.crestBias=-this.crestMean*this.crestGain;
    this.shadingStatistics={big_height:this.bigHeight,crest_gain:this.crestGain,crest_bias:this.crestBias};
  }

  resetFoam() {
    const gl=this.gl;for(const target of this.foamTargets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER,target.framebuffer);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }

  dispose() {
    if(this.disposed) return;this.disposed=true;const gl=this.gl;
    this.textures.forEach(t=>gl.deleteTexture(t));this.framebuffers.forEach(f=>gl.deleteFramebuffer(f));
    Object.values(this.programs).forEach(p=>gl.deleteProgram(p.program));gl.deleteVertexArray(this.vao);
  }
}
