// Copyright 2015–2026 Christopher Jon Horvath. Apache-2.0.
// The School physical ocean material and projected grid, ported to WebGL 2.
// The separately licensed sky photograph is described in assets/SKY-LICENSE.txt.

const DEG = Math.PI / 180;
export const DEFAULT_LOOK = Object.freeze({exposure:.55, sky_rotation:25, sky_gain:1, haze:.9,
  foam:1, aeration:1.25, crest_crumble:true, crest_crumble_strength:.5});

export function cameraBasis(camera) {
  const yaw=camera.yaw*DEG, pitch=camera.pitch*DEG;
  const forward=[Math.sin(yaw)*Math.cos(pitch),Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch)];
  const right=[Math.cos(yaw),-Math.sin(yaw),0];
  const up=[right[1]*forward[2],-right[0]*forward[2],right[0]*forward[1]-right[1]*forward[0]];
  return {forward,right,up};
}

const COMMON = `
precision highp float;
precision highp int;
const float pi=3.141592653589793;
uniform vec3 eye,forward,right,up;
uniform float tanFov,aspect,exposure,skyRotation,skyGain,haze,oceanRotation,domain,waveSize;
uniform float bigHeight,foamStrength,aeration,crestStrength,skyScale,reflectionScale,reflectionLevels;
uniform vec2 grid;
uniform vec3 diffuseSH[9],ambientSH[9],sunDirection,sunIrradiance;
uniform sampler2D skyTexture,reflectionTexture,displacements,normals,foamTexture,foamMaterial,foamDetail;
vec2 rotateXY(vec2 v,float angle){float c=cos(angle),s=sin(angle);return vec2(c*v.x-s*v.y,s*v.x+c*v.y);}
vec3 oceanToWorld(vec3 v){return vec3(rotateXY(v.xy,oceanRotation),v.z);}
vec3 environment(vec3 direction,float level){
  vec3 d=normalize(direction);
  vec2 uv=vec2((atan(d.y,d.x)+skyRotation)/(2.*pi)+.5,acos(clamp(d.z,-1.,1.))/pi);
  vec3 encoded=textureLod(skyTexture,uv,level).rgb;
  return encoded*encoded*skyScale*skyGain;
}
vec3 filteredReflection(vec3 direction,float roughness){
  vec3 d=normalize(direction);
  vec2 uv=vec2((atan(d.y,d.x)+skyRotation)/(2.*pi)+.5,acos(clamp(d.z,-1.,1.))/pi);
  vec3 encoded=textureLod(reflectionTexture,uv,roughness*reflectionLevels).rgb;
  return encoded*encoded*reflectionScale*skyGain;
}
vec3 horizonRadiance(vec3 direction){
  vec2 horizontal=direction.xy/max(length(direction.xy),1e-6);
  return environment(vec3(horizontal,1e-6),5.);
}
vec3 displayColor(vec3 linear){
  vec3 color=max(linear*exp2(exposure),vec3(0));
  float floorValue=min(color.r,min(color.g,color.b));
  float offset=floorValue<.08?floorValue-6.25*floorValue*floorValue:.04;
  color=max(color-vec3(offset),vec3(0));
  float peak=max(color.r,max(color.g,color.b));
  if(peak>.76){
    float shoulder=1.-.0576/(peak-.52);
    color*=shoulder/peak;
    color=mix(color,vec3(shoulder),1.-1./(1.+.15*(peak-shoulder)));
  }
  return mix(12.92*color,1.055*pow(color,vec3(1./2.4))-.055,greaterThan(color,vec3(.0031308)));
}
void skyBasis(vec3 direction,out float basis[9]){
  vec3 p=vec3(rotateXY(direction.xy,skyRotation),direction.z);
  basis[0]=.2820947918;basis[1]=.4886025119*p.y;basis[2]=.4886025119*p.z;basis[3]=.4886025119*p.x;
  basis[4]=1.0925484306*p.x*p.y;basis[5]=1.0925484306*p.y*p.z;basis[6]=.3153915653*(3.*p.z*p.z-1.);
  basis[7]=1.0925484306*p.x*p.z;basis[8]=.5462742153*(p.x*p.x-p.y*p.y);
}
vec3 skyLight(vec3 normal,bool ambient){
  float basis[9];skyBasis(normal,basis);vec3 result=vec3(0);
  for(int i=0;i<9;i++)result+=(ambient?ambientSH[i]:diffuseSH[i])*basis[i];
  return max(result,vec3(0))*skyGain;
}
vec3 fogRadiance(vec3 direction){
  float basis[9];skyBasis(direction,basis);vec3 result=vec3(0);
  for(int i=0;i<9;i++)result+=diffuseSH[i]*basis[i]*((i>0&&i<4)?.75:1.);
  return max(result,vec3(0))*skyGain;
}
float fresnelDielectric(float cosine){
  float ci=clamp(cosine,0.,1.),eta=1./1.333;
  float ct=sqrt(max(0.,1.-eta*eta*(1.-ci*ci)));
  float rs=(ci-1.333*ct)/max(ci+1.333*ct,1e-6);
  float rp=(ct-1.333*ci)/max(ct+1.333*ci,1e-6);
  return clamp(.5*(rs*rs+rp*rp),0.,1.);
}
float environmentBRDF(float cosine,float roughness){
  float nv=clamp(cosine,0.,1.);
  vec4 r=roughness*vec4(-1.,-.0275,-.572,.022)+vec4(1.,.0425,1.04,-.04);
  float q=min(r.x*r.x,exp2(-9.28*nv))*r.x+r.y;
  vec2 ab=vec2(-1.04,1.04)*q+r.zw;
  float fitted=clamp(.0203732*ab.x+ab.y,0.,1.);
  return mix(fresnelDielectric(nv),fitted,smoothstep(0.,.3,roughness));
}
float phaseHG(float cosine,float g){
  float d=max(1e-4,1.+g*g-2.*g*clamp(cosine,-1.,1.));
  return (1.-g*g)/(4.*pi*d*sqrt(d));
}
vec3 atmosphere(vec3 color,vec3 world,vec3 incident){
  float distanceToEye=length(world-eye);
  float opticalDepth=3.912*distanceToEye*haze/20000.;
  float transmittance=exp(-opticalDepth);
  vec3 airlight=fogRadiance(incident);
  float horizonWeight=(1.-smoothstep(0.,.0872,abs(incident.z)))*smoothstep(.5,2.,opticalDepth);
  vec3 inscatter=mix(airlight,horizonRadiance(incident),horizonWeight);
  return mix(inscatter,color,transmittance);
}
vec2 foamSurface(vec3 density,float grain,float strength){
  vec3 d=max(density,vec3(0));
  float olderAir=max(d.y+d.z-.25*d.x,0.);
  float fresh=d.x/max(d.x+4.*olderAir,1e-8);
  float detail=clamp(grain,0.,1.);
  float modulation=mix(.35+1.3*detail,.85+.3*detail,fresh);
  float coverage=smoothstep(.07,.5,d.x*max(strength,0.)*modulation);
  return vec2(coverage,coverage*fresh);
}
float crestSurface(float minimumStretch,float coarse,float fine,float strength){
  const float threshold=.75,width=.15,onset=.79;
  float compression=clamp((onset-minimumStretch)/(onset-threshold+width),0.,1.);
  float coarseGrain=clamp(coarse,0.,1.),fineGrain=clamp(fine,0.,1.);
  float boundary=.30+1.1*(1.-coarseGrain)+.6*(.5-fineGrain);
  float fragments=smoothstep(.18,.68,fineGrain);
  return .60*smoothstep(max(0.,boundary-.22),boundary+.22,compression)*fragments*clamp(strength,0.,1.);
}
vec3 foamDetailNormal(vec3 baseNormal,vec4 detail){
  vec3 normal=normalize(baseNormal),oceanX=vec3(cos(oceanRotation),sin(oceanRotation),0.);
  vec3 tangent=oceanX-dot(oceanX,normal)*normal;
  if(dot(tangent,tangent)<1e-6){vec3 oceanY=vec3(-sin(oceanRotation),cos(oceanRotation),0.);tangent=oceanY-dot(oceanY,normal)*normal;}
  tangent=normalize(tangent);
  return normalize(normal-detail.x*tangent-detail.y*cross(normal,tangent));
}
float foamDetailVariance(vec4 detail){return max(detail.w-dot(detail.xy,detail.xy),0.);}
`;

const FULLSCREEN_VERTEX = `#version 300 es
precision highp float;
out vec2 screen;
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2))*2.-1.;screen=p;gl_Position=vec4(p,0.,1.);}
`;
const SKY_FRAGMENT = `#version 300 es
${COMMON}
in vec2 screen;out vec4 color;
void main(){
  vec3 direction=normalize(forward+screen.x*tanFov*aspect*right+screen.y*tanFov*up);
  float clear=smoothstep(0.,.035,max(direction.z,0.));
  vec3 sky=mix(horizonRadiance(direction),environment(direction,0.),clear);
  color=vec4(displayColor(sky),1.);
}`;

const OCEAN_VERTEX = `#version 300 es
${COMMON}
centroid out vec3 world;
centroid out vec2 uv;
void main(){
  int nx=int(grid.x)+1,ix=gl_VertexID%nx,iy=gl_VertexID/nx;
  float lateral=2.*float(ix)/grid.x-1.,row=float(iy)/grid.y;
  vec2 side=right.xy;
  float upright=up.z>=0.?1.:-1.;
  vec2 ahead=vec2(-side.y,side.x)*upright;
  float height=eye.z;
  vec3 bottomRay=forward-1.05*tanFov*up*upright,topRay=forward+1.05*tanFov*up*upright;
  if(bottomRay.z>=0.){gl_Position=vec4(0.,0.,2.,1.);world=eye;uv=vec2(0.);return;}
  float padding=max(bigHeight,2.*domain/waveSize);
  float bottomRatio=dot(bottomRay.xy,ahead)/(-bottomRay.z);
  float nearA=max(-30000.,min((height-padding)*bottomRatio,(height+padding)*bottomRatio)-padding);
  float farA=30000.;
  if(topRay.z<0.){float topRatio=dot(topRay.xy,ahead)/(-topRay.z);farA=min(30000.,max((height-padding)*topRatio,(height+padding)*topRatio)+padding);}
  float span=max(.5,farA-nearA),scale=max(abs(height),.5);
  float distanceToRow=nearA+scale*(pow(1.+span/scale,row)-1.);
  float planeDepth=dot(vec3(ahead*distanceToRow,-height),forward);
  float halfWidth=max(.5,planeDepth+padding)*tanFov*aspect*1.05+padding;
  vec2 base=eye.xy+distanceToRow*ahead+lateral*halfWidth*side;
  uv=rotateXY(base,-oceanRotation)/domain;
  float texelWorld=domain/waveSize,footprint=2.*halfWidth/grid.x;
  float lod=max(0.,log2(max(footprint/texelWorld,1.))-.6);
  vec3 disp=textureLod(displacements,uv,lod).xyz;
  world=vec3(base,0.)+oceanToWorld(disp);
  vec3 view=world-eye;float z=dot(view,forward);
  // WebGL's depth interval is [-w,w]; native WebGPU uses [0,w].
  gl_Position=vec4(dot(view,right)/(tanFov*aspect),dot(view,up)/tanFov,z*1.000002-.2,z);
}`;

const OCEAN_FRAGMENT = `#version 300 es
${COMMON}
centroid in vec3 world;
centroid in vec2 uv;
out vec4 fragmentColor;
void main(){
  vec3 view=normalize(eye-world);
  vec3 averagedNormal=oceanToWorld(texture(normals,uv).xyz);
  float normalLength=clamp(length(averagedNormal),.01,1.);
  vec3 normal=averagedNormal/normalLength;
  if(dot(normal,view)<0.)normal=reflect(normal,view);
  normal=normalize(normal+view*1e-5);
  float nv=clamp(dot(normal,view),0.,1.);
  vec3 reflection=reflect(-view,normal);
  float slope=sqrt(.025*.025+max(0.,2.*(1.-normalLength)/normalLength));
  float roughness=clamp(sqrt(slope),.04,.8);
  float reflectance=environmentBRDF(nv,roughness);
  vec3 reflected=filteredReflection(reflection,roughness);
  vec4 rawFoam=texture(foamTexture,uv);
  vec3 rawDensity=max(rawFoam.rgb,vec3(0)),density=rawDensity*foamStrength;
  float shallow=1.-exp(-density.g*aeration*2.),deepAir=1.-exp(-density.b*aeration);
  vec3 waterAbsorption=vec3(.145,.115,.125),waterScattering=vec3(.021,.022,.019);
  vec3 waterAlbedo=waterScattering/(waterAbsorption+waterScattering);
  vec3 aeratedAlbedo=min(4.*waterAlbedo,vec3(.85));
  float bubbles=.6*shallow+.1*deepAir;
  vec3 sigmaS=waterScattering+bubbles*aeratedAlbedo;
  vec3 sigmaA=waterAbsorption+bubbles*(vec3(1.)-aeratedAlbedo),sigmaT=sigmaA+sigmaS;
  vec3 albedo=sigmaS/sigmaT;
  vec3 ambient=skyLight(normalize(vec3(normal.xy*.25,1.)),true);
  vec3 body=.25*ambient*albedo;
  vec3 sun=normalize(sunDirection);
  float nl=clamp(dot(normal,sun),0.,1.);
  if(nl>0.){
    vec3 insideView=refract(-view,normal,1./1.333),insideLight=refract(-sun,normal,1./1.333);
    float muView=max(.01,-dot(insideView,normal)),muLight=max(.01,-dot(insideLight,normal));
    float opticalThickness=max(.25,2.5-2.*world.z/max(bigHeight,.01));
    vec3 attenuation=vec3(1.)-exp(-sigmaT*opticalThickness*(1./muView+1./muLight));
    float phase=phaseHG(dot(insideLight,-insideView),mix(.85,.35,shallow));
    float eta=1./1.333;
    body+=sunIrradiance*skyGain*(1.-fresnelDielectric(nl))*eta*eta*albedo*phase*nl/(muLight+muView)*attenuation;
  }
  vec3 color=reflectance*reflected+(1.-reflectance)*body;
  float grain=texture(foamTexture,uv*4.).a;
  vec2 closeMaterial=foamSurface(rawDensity,grain,foamStrength);
  vec4 filteredMaterial=texture(foamMaterial,uv);
  vec2 size=vec2(textureSize(foamTexture,0));
  float footprint=max(length(dFdx(uv)*size),length(dFdy(uv)*size));
  float filteredWeight=smoothstep(.6,1.5,footprint);
  vec2 material=mix(closeMaterial,filteredMaterial.rg,filteredWeight);
  float coverage=clamp(material.x,0.,1.),fresh=clamp(material.y/max(coverage,1e-6),0.,1.);
  vec4 detail=texture(foamDetail,uv*domain/4.);
  vec3 foamBaseNormal=normalize(vec3(normal.xy*.65,max(normal.z,.15)));
  vec3 foamNormal=foamDetailNormal(foamBaseNormal,detail);
  float resolvedRoughness=mix(.6,.32,fresh);
  float foamRoughness=min(.9,pow(pow(resolvedRoughness,4.)+.5*foamDetailVariance(detail),.25));
  float foamReflectance=environmentBRDF(dot(foamNormal,view),foamRoughness);
  float foamAlbedo=mix(.72,.86,fresh)*detail.b;
  vec3 foamLight=(1.-foamReflectance)*foamAlbedo*skyLight(foamNormal,false)
    +foamReflectance*filteredReflection(reflect(-view,foamNormal),foamRoughness);
  vec3 waterLight=color;
  color=mix(color,foamLight,coverage);
  if(crestStrength>0.){
    float minimumStretch=-texture(displacements,uv).w;
    float closeCrest=(1.-closeMaterial.x)*crestSurface(minimumStretch,rawFoam.a,grain,foamStrength);
    float crestFootprint=footprint*max(1.,waveSize/size.x);
    float crestFilteredWeight=smoothstep(.6,1.5,crestFootprint);
    float extra=clamp(mix(closeCrest,filteredMaterial.a,crestFilteredWeight),0.,1.-coverage)*clamp(crestStrength,0.,1.);
    vec3 breakingNormal=foamDetailNormal(normal,detail);
    if(dot(breakingNormal,view)<0.)breakingNormal=reflect(breakingNormal,view);
    breakingNormal=normalize(breakingNormal+view*1e-5);
    float breakingRoughness=min(.9,pow(pow(.32,4.)+.5*foamDetailVariance(detail),.25));
    float breakingReflectance=environmentBRDF(dot(breakingNormal,view),breakingRoughness);
    vec3 crestLight=(1.-breakingReflectance)*.86*detail.b*skyLight(breakingNormal,false)
      +breakingReflectance*filteredReflection(reflect(-view,breakingNormal),breakingRoughness);
    color+=extra*(crestLight-waterLight);
  }
  fragmentColor=vec4(displayColor(atmosphere(color,world,-view)),1.);
}`;

const MATERIAL_FRAGMENT=`#version 300 es
${COMMON}
uniform float materialSize;
out vec4 result;
void main(){
  vec2 pixel=floor(gl_FragCoord.xy);vec3 integrated=vec3(0.);float crest=0.;
  for(int y=0;y<4;y++)for(int x=0;x<4;x++){
    vec2 uv=(pixel+(vec2(float(x),float(y))+.5)*.25)/materialSize;
    vec4 raw=textureLod(foamTexture,uv,0.);
    float grain=clamp(textureLod(foamTexture,uv*4.,0.).a,0.,1.);
    vec2 material=foamSurface(raw.rgb,grain,foamStrength);
    integrated+=vec3(material,material.x*grain);
    if(crestStrength>0.)crest+=(1.-material.x)*crestSurface(-textureLod(displacements,uv,0.).w,raw.a,grain,foamStrength);
  }
  result=vec4(integrated/16.,min(crest/16.,1.-integrated.x/16.));
}`;

function makeProgram(gl,vertexSource,fragmentSource,label){
  const compile=(type,source)=>{
    const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){
      const message=gl.getShaderInfoLog(shader);gl.deleteShader(shader);throw new Error(`${label} shader: ${message}`);
    }
    return shader;
  };
  const vertex=compile(gl.VERTEX_SHADER,vertexSource),fragment=compile(gl.FRAGMENT_SHADER,fragmentSource);
  const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);
  gl.deleteShader(vertex);gl.deleteShader(fragment);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(`${label} link: ${gl.getProgramInfoLog(program)}`);
  const uniforms={};
  for(let i=0;i<gl.getProgramParameter(program,gl.ACTIVE_UNIFORMS);i++){
    const name=gl.getActiveUniform(program,i).name;uniforms[name.replace(/\[0\]$/,'')]=gl.getUniformLocation(program,name);
  }
  return {program,uniforms};
}

async function fetchChecked(url){const result=await fetch(url);if(!result.ok)throw new Error(`Unable to load ${url}: HTTP ${result.status}`);return result;}

async function loadAtlas(gl,base,spec){
  const blob=await (await fetchChecked(new URL(spec.url,base))).blob();
  const bitmap=await createImageBitmap(blob,{premultiplyAlpha:'none',colorSpaceConversion:'none'});
  const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d',{willReadFrequently:true});
  canvas.width=bitmap.width;canvas.height=bitmap.height;ctx.drawImage(bitmap,0,0);bitmap.close();
  const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);
  for(let level=0;level<spec.levels.length;level++){
    const mip=spec.levels[level],pixels=ctx.getImageData(0,mip.y,mip.width,mip.height).data;
    gl.texImage2D(gl.TEXTURE_2D,level,gl.RGBA8,mip.width,mip.height,0,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
  }
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAX_LEVEL,spec.levels.length-1);
  return texture;
}

async function loadFoamDetail(gl,base,spec){
  const data=await (await fetchChecked(new URL(spec.url,base))).arrayBuffer();
  const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);let offset=0;
  for(let level=0;level<spec.levels;level++){
    const n=Math.max(1,spec.resolution>>level),count=n*n*4;
    gl.texImage2D(gl.TEXTURE_2D,level,gl.RGBA16F,n,n,0,gl.RGBA,gl.HALF_FLOAT,new Uint16Array(data,offset,count));
    offset+=count*2;
  }
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.REPEAT);
  return texture;
}

export class OceanRenderer {
  static async create(gl,options={}){
    const base=new URL('./assets/',import.meta.url);
    const metadata=await (await fetchChecked(new URL('sky.json',base))).json();
    const renderer=new OceanRenderer(gl,metadata,options);
    // Upload sequentially: each loader mutates the currently bound GL texture.
    renderer.sky=await loadAtlas(gl,base,metadata.sky);
    renderer.reflection=await loadAtlas(gl,base,metadata.reflection);
    renderer.detail=await loadFoamDetail(gl,base,metadata.foamDetail);
    return renderer;
  }

  constructor(gl,metadata,options={}){
    this.gl=gl;this.metadata=metadata;
    this.skyProgram=makeProgram(gl,FULLSCREEN_VERTEX,SKY_FRAGMENT,'Sky');
    this.oceanProgram=makeProgram(gl,OCEAN_VERTEX,OCEAN_FRAGMENT,'Ocean');
    this.materialProgram=makeProgram(gl,FULLSCREEN_VERTEX,MATERIAL_FRAGMENT,'Foam coverage');
    this.emptyVAO=gl.createVertexArray();this.meshVAO=gl.createVertexArray();this.indices=gl.createBuffer();
    this.setMeshResolution(options.gridWidth??320,options.gridHeight??192);
    this.materialSize=options.materialSize??256;
    this.material=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.material);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,this.materialSize,this.materialSize,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.REPEAT);
    gl.generateMipmap(gl.TEXTURE_2D);
    this.materialFramebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.materialFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.material,0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('Foam material framebuffer is unavailable.');
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindVertexArray(null);
  }

  setMeshResolution(nx,ny){
    const gl=this.gl;this.grid=[nx,ny];const indices=new Uint32Array(nx*ny*6);let i=0;
    for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
      const a=y*(nx+1)+x;indices.set([a,a+1,a+nx+1,a+1,a+nx+2,a+nx+1],i);i+=6;
    }
    gl.bindVertexArray(this.meshVAO);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);this.indexCount=indices.length;gl.bindVertexArray(null);
  }

  bindUniforms(program,sim,camera,look,width,height){
    const gl=this.gl,u=program.uniforms;gl.useProgram(program.program);
    const scalar=(name,value)=>{if(u[name]!==undefined)gl.uniform1f(u[name],value);};
    const vector=(name,value)=>{if(u[name]!==undefined)gl.uniform3fv(u[name],value);};
    const basis=cameraBasis(camera),statistics=sim.shadingStatistics??{};
    vector('eye',[camera.x,camera.y,camera.height]);vector('forward',basis.forward);vector('right',basis.right);vector('up',basis.up);
    scalar('tanFov',Math.tan(camera.fov*DEG/2));scalar('aspect',width/height);
    scalar('exposure',look.exposure);scalar('skyRotation',look.sky_rotation*DEG);scalar('skyGain',look.sky_gain);
    scalar('haze',look.haze);scalar('oceanRotation',(look.wind_direction??sim.parameters?.wind_direction??0)*DEG);
    scalar('domain',sim.domain??1024);scalar('waveSize',sim.resolution);
    scalar('bigHeight',statistics.big_height??sim.bigHeight??10);
    scalar('foamStrength',look.foam);scalar('aeration',look.aeration);
    scalar('crestStrength',look.crest_crumble?look.crest_crumble_strength:0);
    scalar('skyScale',this.metadata.sky.scale);scalar('reflectionScale',this.metadata.reflection.scale);
    scalar('reflectionLevels',this.metadata.reflection.levels.length-1);scalar('materialSize',this.materialSize);
    if(u.grid!==undefined)gl.uniform2fv(u.grid,this.grid);
    if(u.diffuseSH!==undefined)gl.uniform3fv(u.diffuseSH,this.metadata.diffuseSH.flat());
    if(u.ambientSH!==undefined)gl.uniform3fv(u.ambientSH,this.metadata.ambientSH.flat());
    const sun=this.metadata.sunDirection,rotation=look.sky_rotation*DEG,c=Math.cos(rotation),s=Math.sin(rotation);
    vector('sunDirection',[sun[0]*c+sun[1]*s,sun[1]*c-sun[0]*s,sun[2]]);vector('sunIrradiance',this.metadata.sunIrradiance);
    const textures=[['skyTexture',this.sky],['reflectionTexture',this.reflection],['displacements',sim.displacement],
      ['normals',sim.normal],['foamTexture',sim.foam],['foamMaterial',this.material],['foamDetail',this.detail]];
    textures.forEach(([name,texture],i)=>{if(u[name]!==undefined){gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,texture);gl.uniform1i(u[name],i);}});
  }

  render(sim,camera,appearance,width,height,time=0){
    const gl=this.gl,look={...DEFAULT_LOOK,...appearance};
    gl.disable(gl.SCISSOR_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);gl.disable(gl.DEPTH_TEST);
    gl.colorMask(true,true,true,true);gl.depthMask(true);
    // Area coverage is filtered after the nonlinear material, retaining distant whitecaps.
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.materialFramebuffer);gl.viewport(0,0,this.materialSize,this.materialSize);
    gl.bindVertexArray(this.emptyVAO);this.bindUniforms(this.materialProgram,sim,camera,look,width,height);gl.drawArrays(gl.TRIANGLES,0,3);
    gl.bindTexture(gl.TEXTURE_2D,this.material);gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,width,height);
    gl.clearDepth(1);gl.clear(gl.DEPTH_BUFFER_BIT);
    this.bindUniforms(this.skyProgram,sim,camera,look,width,height);gl.drawArrays(gl.TRIANGLES,0,3);
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
    this.bindUniforms(this.oceanProgram,sim,camera,look,width,height);gl.bindVertexArray(this.meshVAO);
    gl.drawElements(gl.TRIANGLES,this.indexCount,gl.UNSIGNED_INT,0);gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }

  destroy(){
    const gl=this.gl;for(const program of [this.skyProgram,this.oceanProgram,this.materialProgram])gl.deleteProgram(program.program);
    for(const texture of [this.sky,this.reflection,this.detail,this.material])gl.deleteTexture(texture);
    gl.deleteFramebuffer(this.materialFramebuffer);gl.deleteBuffer(this.indices);gl.deleteVertexArray(this.meshVAO);gl.deleteVertexArray(this.emptyVAO);
  }
}
