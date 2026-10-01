// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
struct Uniforms {
    eye_time: vec4f,
    forward_fov: vec4f,
    right_aspect: vec4f,
    up_exposure: vec4f,
    ocean: vec4f,        // domain, resolution, crest threshold, foam amount
    environment: vec4f,  // sky rotation, haze, sky gain, ocean rotation
    sun: vec4f,
    sun_color: vec4f,
    grid: vec4f,         // x segments, y segments, near distance, far distance
    statistics: vec4f,   // big height, crest gain, crest bias, max crest
    moon_color: vec4f,
    aeration: vec4f,      // use foam history, subsurface strength, fresh crests, crest breakup
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var wave_sampler: sampler;
@group(0) @binding(2) var displacements: texture_2d<f32>;
@group(0) @binding(3) var normals: texture_2d<f32>;
@group(0) @binding(4) var sky_sampler: sampler;
@group(0) @binding(5) var sky_texture: texture_2d<f32>;
@group(0) @binding(6) var foam_texture: texture_2d<f32>;
const pi: f32 = 3.141592653589793;

fn rotate_xy(v: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(c*v.x-s*v.y,s*v.x+c*v.y);
}

fn ocean_to_world(v: vec3f) -> vec3f {
    return vec3f(rotate_xy(v.xy,u.environment.w),v.z);
}

fn environment(direction: vec3f, level: f32) -> vec3f {
    let d = normalize(direction);
    let uv = vec2f((atan2(d.y,d.x)+u.environment.x)/(2.0*pi)+0.5,
                   acos(clamp(d.z,-1.0,1.0))/pi);
    return textureSampleLevel(sky_texture,sky_sampler,uv,level).rgb*u.environment.z;
}

// Original OceanTestShaders.cpp gammaCorrect(..., 2.2), with an explicit
// photographic exposure control at zero by default. Target is UNORM, not sRGB.
fn display_color(linear: vec3f) -> vec3f {
    return pow(max(linear*exp2(u.up_exposure.w),vec3f(0.0)),vec3f(1.0/2.2));
}
struct Sky_vertex { @builtin(position) position: vec4f, @location(0) ray: vec3f };
@vertex fn sky_vertex(@builtin(vertex_index) index: u32) -> Sky_vertex {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    let p = vec2f(x,y)*2.0-1.0;
    var out: Sky_vertex;
    out.position = vec4f(p,0.999999,1.0);
    out.ray = u.forward_fov.xyz + p.x*u.forward_fov.w*u.right_aspect.w*u.right_aspect.xyz
              + p.y*u.forward_fov.w*u.up_exposure.xyz;
    return out;
}
@fragment fn sky_fragment(in: Sky_vertex) -> @location(0) vec4f {
    return vec4f(display_color(environment(in.ray,0.0)),1.0);
}

struct Ocean_vertex {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
    @location(1) uv: vec2f,
};
@vertex fn ocean_vertex(@builtin(vertex_index) index: u32) -> Ocean_vertex {
    let nx = u32(u.grid.x)+1u;
    let ix = index % nx;
    let iy = index / nx;
    let lateral = 2.0*f32(ix)/u.grid.x-1.0;
    let row = f32(iy)/u.grid.y;
    let distance = u.grid.z*pow(u.grid.w/u.grid.z,row);
    let ahead = normalize(vec2f(u.forward_fov.x,u.forward_fov.y));
    let side = u.right_aspect.xy;
    let base = u.eye_time.xy + distance*(ahead + lateral*side*u.forward_fov.w*u.right_aspect.w*1.8);
    // The view-adaptive grid is in world space. Its inverse-transformed base
    // samples the unchanged +X ocean; rotate displaced points back to the world.
    let uv = rotate_xy(base,-u.environment.w)/u.ocean.x;
    let texel_world = u.ocean.x/u.ocean.y;
    let footprint = distance*(2.0*u.forward_fov.w*u.right_aspect.w/u.grid.x);
    let lod = max(0.0,log2(max(footprint/texel_world,1.0))-.6);
    let disp = textureSampleLevel(displacements,wave_sampler,uv,lod).xyz;
    let world = vec3f(base,0.0)+ocean_to_world(disp);
    let view = world-u.eye_time.xyz;
    let z = dot(view,u.forward_fov.xyz);
    let clip = vec4f(dot(view,u.right_aspect.xyz)/(u.forward_fov.w*u.right_aspect.w),
                     dot(view,u.up_exposure.xyz)/u.forward_fov.w,
                     z*1.000001-0.1,z);
    var out: Ocean_vertex;
    out.position = clip;
    out.world = world;
    out.uv = uv;
    return out;
}

// Direct WGSL port of g_fragmentShaderTextureSkyBase in
// src/EncinoWaves/Tests/OceanTestShaders.cpp (2015), including its artistic
// constants. Changes are confined to guarding singular denominators and the
// display / texture coordinate conventions required by WebGPU.
fn linstep(low: f32, high: f32, value: f32) -> f32 {
    return clamp((value-low)/max(high-low,1e-8),0.0,1.0);
}
struct Refraction { reflection: vec3f, transmission: vec3f, kr: f32, kt: f32 };
fn my_refract(incident: vec3f, normal: vec3f) -> Refraction {
    let ni=1.0;
    let nt=1.3;
    let eta=ni/nt;
    let view=-incident;
    let ci=dot(view,normal);
    let a=view-ci*normal;
    let b=-eta*a;
    let si=sqrt(clamp(1.0-ci*ci,0.0,1.0));
    let st=eta*si;
    let ct=sqrt(clamp(1.0-st*st,0.0,1.0));
    let rs=(ni*ci-nt*ct)/(ni*ci+nt*ct);
    let rp=(ni*ct-nt*ci)/(ni*ct+nt*ci);
    let kr=(rp*rp+rs*rs)/2.0;
    return Refraction(normalize(view-2.0*a),normalize(b-ct*normal),kr,(1.0-kr)/(eta*eta));
}
fn hg_phase(g: vec3f, theta: f32) -> vec3f {
    return (vec3f(1.0)-g*g)/(4.0*3.141592*pow(vec3f(1.0)+g*g-2.0*g*cos(theta),vec3f(1.5)));
}
fn full_extinction_color(light_color: vec3f, extinction: vec3f,
                         scattering: vec3f, phase_g: vec3f,
                         to_light: vec3f, incident: vec3f,
                         big_height: f32, world: vec3f) -> vec3f {
    let ln=normalize(to_light);
    let start_depth=max(big_height-world.z,.01)/u.ocean.x;
    let theta=acos(clamp(dot(-incident,-ln),-1.0,1.0));
    let hgp=hg_phase(phase_g,theta);
    let iz=.1*abs(incident.z);
    let lz=max(.1*abs(ln.z),1e-7);
    let extincted=exp(-start_depth*extinction/lz)*lz/((iz+lz)*extinction);
    let gain=linstep(-2.8*big_height,big_height,world.z);
    return 10.0*5.25*gain*light_color*hgp*scattering*extincted;
}
fn layered_fog(color: vec3f, distance: f32, incident: vec3f) -> vec3f {
    let beta=.1;
    let iz=incident.z;
    let bd=beta*iz*distance;
    // Limit at the horizon is distance, avoiding the original 0/0.
    let eye_height=max(0.0,u.eye_time.z+.4*u.statistics.x);
    let start=exp(-beta*eye_height);
    var k=distance*start;
    if abs(bd)>1e-5 {
        // Algebraically identical, without 0 * infinity from a high camera.
        let end=exp(min(80.0,-beta*eye_height-bd));
        k=max(0.0,(start-end)/(beta*iz));
    }
    let density=1.5*vec3f(.0005,.0004,.00045)*u.environment.y;
    let transmittance=exp(-k*density);
    let fog=pow(vec3f(.75),vec3f(2.2));
    return mix(fog,color,transmittance);
}
@fragment fn ocean_fragment(in: Ocean_vertex) -> @location(0) vec4f {
    let incident=normalize(in.world-u.eye_time.xyz);
    let normal=normalize(ocean_to_world(textureSample(normals,wave_sampler,in.uv).xyz));
    let refracted=my_refract(incident,normal);
    var reflection=refracted.reflection;
    reflection.z=abs(reflection.z);
    let sun=normalize(u.sun.xyz);
    let moon=vec3f(-sun.x,-sun.y,sun.z);
    let sun_color=u.sun_color.rgb;
    let moon_color=u.moon_color.rgb;
    let diffuse=.95*sun_color*max(dot(sun,normal),0.0)
                +.35*moon_color*max(dot(moon,normal),0.0);
    let sky=environment(reflection,0.0);
    // RGB history: surface coverage, shallow aeration, deeper aeration. These
    // densities move with the same ocean-space mesh and use its original media.
    let density=max(textureSample(foam_texture,wave_sampler,in.uv).rgb,vec3f(0.0))*u.aeration.x*u.ocean.w;
    let shallow=1.0-exp(-density.g*u.aeration.y*2.0);
    let deep_air=1.0-exp(-density.b*u.aeration.y);
    let sigma=vec3f(3.3645,3.158,3.2428)+shallow*vec3f(1.8)+deep_air*vec3f(.8);
    let scattering=vec3f(.1800,.1834,.2281)+shallow*vec3f(1.35,1.50,1.55)+deep_air*vec3f(.25,.65,.80);
    let phase_g=mix(vec3f(.902,.825,.914),vec3f(.65),clamp(shallow+.5*deep_air,0.0,1.0));
    let big_height=u.statistics.x;
    let deep2=full_extinction_color(sun_color,sigma,scattering,phase_g,sun,incident,big_height,in.world)
             +full_extinction_color(moon_color,sigma,scattering,phase_g,moon,incident,big_height,in.world);
    let deep=.25*(1.0-.75*abs(refracted.transmission.z))
             *linstep(-1.8*big_height,big_height,in.world.z)*sun_color*scattering*hg_phase(phase_g,.1*3.141592);
    let half_vector=normalize(sun-incident);
    let nh=dot(normal,half_vector);
    let specular=sun_color*refracted.kr*.01*pow(nh*nh,400.0);
    var color=refracted.kr*sky+specular+refracted.kt*deep2+.375*refracted.kt*deep+.001*diffuse;
    if u.aeration.x>0.5 {
        // Very dilute remnants are transparent; connected patches become an
        // opaque foam layer instead of washing every ripple with white.
        // Mip-filtered bubble-scale breakup is anchored to the ocean. Its
        // average tends to one in the distance, without crawling pixel noise.
        let grain=textureSample(foam_texture,wave_sampler,in.uv*4.0).a;
        let old_coverage=smoothstep(.07,.5,density.r*(.35+1.3*grain));
        // A light immediate crest layer fills gaps before foam accumulates.
        // Its breakup is independent of emission and never cuts holes as
        // strongly by default. Use the same softer lighting as aged foam.
        var fresh_coverage=0.0;
        if u.ocean.z<u.statistics.w {
            let crest=textureSample(displacements,wave_sampler,in.uv).w*u.statistics.y+u.statistics.z;
            let breakup=mix(1.0,grain,clamp(u.aeration.w,0.0,1.0));
            fresh_coverage=clamp(smoothstep(u.ocean.z,u.statistics.w,crest)*u.aeration.z*breakup*u.ocean.w,0.0,1.0);
        }
        // Coverage union: already opaque foam does not get a second bright coat.
        let coverage=old_coverage+(1.0-old_coverage)*fresh_coverage;
        let foam_light=.35*diffuse+.45*environment(normalize(vec3f(normal.xy*.3,1.0)),4.0);
        color=mix(color,foam_light,coverage);
    } else if u.ocean.z<u.statistics.w {
        let crest=textureSample(displacements,wave_sampler,in.uv).w*u.statistics.y+u.statistics.z;
        color=mix(color,diffuse,smoothstep(u.ocean.z,u.statistics.w,crest)*u.ocean.w);
    }
    let distance=1000.0*pow(length(in.world-u.eye_time.xyz)/1000.0,1.125);
    color=layered_fog(color,distance,incident);
    return vec4f(display_color(color),1.0);
}
