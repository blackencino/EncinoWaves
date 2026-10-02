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
    aeration: vec4f,      // use foam history, subsurface strength, unused, unused
    optics: vec4f,        // unresolved RMS slope, max environment mip, visibility metres, unused
    diffuse_sh: array<vec4f,9>, // cosine-convolved radiance, already divided by pi
    ambient_sh: array<vec4f,9>, // same, with the compact direct source removed
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var wave_sampler: sampler;
@group(0) @binding(2) var displacements: texture_2d<f32>;
@group(0) @binding(3) var normals: texture_2d<f32>;
@group(0) @binding(4) var sky_sampler: sampler;
@group(0) @binding(5) var sky_texture: texture_2d<f32>;
@group(0) @binding(6) var foam_texture: texture_2d<f32>;
@group(0) @binding(7) var reflection_texture: texture_2d<f32>;
@group(0) @binding(8) var foam_material: texture_2d<f32>;
@group(0) @binding(9) var foam_detail_texture: texture_2d<f32>;
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

fn filtered_reflection(direction: vec3f, roughness: f32) -> vec3f {
    let d=normalize(direction);
    let uv=vec2f((atan2(d.y,d.x)+u.environment.x)/(2.0*pi)+0.5,
                 acos(clamp(d.z,-1.0,1.0))/pi);
    return textureSampleLevel(reflection_texture,sky_sampler,uv,roughness*u.optics.y).rgb*u.environment.z;
}

fn horizon_radiance(direction: vec3f) -> vec3f {
    return environment(normalize(vec3f(direction.xy,0.0)),5.0);
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
    let direction=normalize(in.ray);
    // Sky and sea meet the same maritime haze limit. The photographic sky is
    // blended only over the lowest two degrees, rather than ending in a seam.
    let clear=smoothstep(0.0,.035,max(direction.z,0.0));
    let sky=mix(horizon_radiance(direction),environment(direction,0.0),clear);
    return vec4f(sky,1.0);
}

struct Ocean_vertex {
    @builtin(position) position: vec4f,
    // Tiny horizon triangles may cover MSAA samples without covering the pixel
    // center. Centroid sampling prevents extrapolation behind the camera and
    // the resulting flashes of the opposite horizon's fog/reflection color.
    @location(0) @interpolate(perspective, centroid) world: vec3f,
    @location(1) @interpolate(perspective, centroid) uv: vec2f,
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

// Bounded dielectric surface with environment-lit foam and a homogeneous
// water-medium approximation. Wave displacement, normals and history are inputs;
// material changes never alter the synthesized wave field.
fn sky_light(normal: vec3f,ambient: bool) -> vec3f {
    let p=vec3f(rotate_xy(normal.xy,u.environment.x),normal.z);
    let basis=array<f32,9>(.2820947918,.4886025119*p.y,.4886025119*p.z,.4886025119*p.x,
        1.0925484306*p.x*p.y,1.0925484306*p.y*p.z,.3153915653*(3.0*p.z*p.z-1.0),
        1.0925484306*p.x*p.z,.5462742153*(p.x*p.x-p.y*p.y));
    var result=vec3f(0.0);
    for(var i=0u;i<9u;i++) {
        result+=select(u.diffuse_sh[i].rgb,u.ambient_sh[i].rgb,ambient)*basis[i];
    }
    return max(result,vec3f(0.0));
}

fn sky_irradiance(normal: vec3f) -> vec3f { return sky_light(normal,false); }

fn fresnel_dielectric(cosine: f32) -> f32 {
    let ci=clamp(cosine,0.0,1.0);
    let eta=1.0/1.333;
    let ct=sqrt(max(0.0,1.0-eta*eta*(1.0-ci*ci)));
    let rs=(ci-1.333*ct)/max(ci+1.333*ct,1e-6);
    let rp=(ct-1.333*ci)/max(ct+1.333*ci,1e-6);
    return clamp(.5*(rs*rs+rp*rp),0.0,1.0);
}

fn phase_hg(cosine: f32,g: f32) -> f32 {
    let d=max(1e-4,1.0+g*g-2.0*g*clamp(cosine,-1.0,1.0));
    return (1.0-g*g)/(4.0*pi*d*sqrt(d));
}

// Karis 2014, analytical split-sum environment BRDF. Input r is perceptual
// roughness; the prefilter uses alpha=r*r. Fresnel is already included here.
// https://www.unrealengine.com/blog/physically-based-shading-on-mobile
fn environment_brdf(cosine: f32,roughness: f32) -> f32 {
    let nv=clamp(cosine,0.0,1.0);
    let r=roughness*vec4f(-1.0,-.0275,-.572,.022)+vec4f(1.0,.0425,1.04,-.04);
    let q=min(r.x*r.x,exp2(-9.28*nv))*r.x+r.y;
    let ab=vec2f(-1.04,1.04)*q+r.zw;
    let fitted=clamp(.0203732*ab.x+ab.y,0.0,1.0);
    // Recover the exact dielectric boundary for a resolved smooth surface.
    return mix(fresnel_dielectric(nv),fitted,smoothstep(.0,.3,roughness));
}

fn atmosphere(color: vec3f,world: vec3f,incident: vec3f) -> vec3f {
    // Homogeneous maritime visibility, metres. Inscatter takes the actual
    // horizon radiance in this view direction, eliminating the old gray seam.
    let distance=length(world-u.eye_time.xyz);
    let transmittance=exp(-3.912*distance*u.environment.y/u.optics.z);
    let horizon=horizon_radiance(incident);
    return mix(horizon,color,transmittance);
}

@fragment fn ocean_fragment(in: Ocean_vertex) -> @location(0) vec4f {
    let view=normalize(u.eye_time.xyz-in.world);
    let averaged_normal=ocean_to_world(textureSample(normals,wave_sampler,in.uv).xyz);
    let normal_length=clamp(length(averaged_normal),.01,1.0);
    var normal=averaged_normal/normal_length;
    // A filtered shading normal may face away although the triangle is visible.
    // Reflect it about V (Bruneton et al. 2010) before evaluating Fresnel.
    if dot(normal,view)<0.0 { normal=reflect(normal,view); }
    normal=normalize(normal+view*1e-5);
    let nv=clamp(dot(normal,view),0.0,1.0);
    let reflection=reflect(-view,normal);
    // Averaged normals encode unresolved variance. Carry it into reflection
    // filtering instead of renormalizing away the distant wave energy.
    let slope=sqrt(u.optics.x*u.optics.x+max(0.0,2.0*(1.0-normal_length)/normal_length));
    let roughness=clamp(sqrt(slope),.04,.8);
    let reflectance=environment_brdf(nv,roughness);
    let reflected=filtered_reflection(reflection,roughness);
    let raw_density=max(textureSample(foam_texture,wave_sampler,in.uv).rgb,vec3f(0.0));
    let density=raw_density*u.aeration.x*u.ocean.w;
    let shallow=1.0-exp(-density.g*u.aeration.y*2.0);
    let deep_air=1.0-exp(-density.b*u.aeration.y);
    // Absorption/scattering are in inverse metres; added bubbles increase both
    // scattering and extinction, keeping the single-scattering albedo bounded.
    let sigma_a=vec3f(.22,.045,.018);
    let sigma_s=vec3f(.006,.010,.009)+vec3f(1.2)*shallow+vec3f(.22)*deep_air;
    let sigma_t=sigma_a+sigma_s;
    let albedo=sigma_s/sigma_t;
    let ambient=sky_light(normalize(vec3f(normal.xy*.25,1.0)),true);
    var body=.33*ambient*albedo;
    let sun=normalize(u.sun.xyz);
    let nl=clamp(dot(normal,sun),0.0,1.0);
    if nl>0.0 {
        let inside_view=refract(-view,normal,1.0/1.333);
        let inside_light=refract(-sun,normal,1.0/1.333);
        let mu_view=max(.01,-dot(inside_view,normal));
        let mu_light=max(.01,-dot(inside_light,normal));
        let optical_thickness=max(.25,2.5-2.0*in.world.z/max(u.statistics.x,.01));
        let attenuation=vec3f(1.0)-exp(-sigma_t*optical_thickness*(1.0/mu_view+1.0/mu_light));
        let phase=phase_hg(dot(inside_light,-inside_view),mix(.85,.35,shallow));
        // Entry conserves flux: E_water/E_air=(1-F)*nl/mu_light.
        // The slab integral cancels mu_light; eta^2 converts its radiance back
        // to air. Exit transmission is applied once with the body below.
        let eta=1.0/1.333;
        body+=u.sun_color.rgb*(1.0-fresnel_dielectric(nl))*eta*eta*albedo*phase
              *nl/(mu_light+mu_view)*attenuation;
    }
    var color=reflectance*reflected+(1.0-reflectance)*body;
    if u.aeration.x>.5 {
        let grain=textureSample(foam_texture,wave_sampler,in.uv*4.0).a;
        let close_material=foam_surface(raw_density,grain,u.ocean.w);
        let filtered_material=textureSample(foam_material,wave_sampler,in.uv).rg;
        let size=vec2f(textureDimensions(foam_texture));
        let footprint=max(length(dpdx(in.uv)*size),length(dpdy(in.uv)*size));
        // Coverage is integrated BEFORE filtering. A distant whitecap keeps its
        // projected area instead of vanishing at a nonlinear threshold.
        let material=mix(close_material,filtered_material,smoothstep(.6,1.5,footprint));
        let coverage=clamp(material.x,0.0,1.0);
        let fresh=clamp(material.y/max(coverage,1e-6),0.0,1.0);
        // Foam is a diffuse scattering layer, illuminated by the whole sky.
        // A sun glint may legitimately be brighter; never clamp foam to water.
        let detail=textureSample(foam_detail_texture,wave_sampler,in.uv*u.ocean.x/4.0);
        let foam_base_normal=normalize(vec3f(normal.xy*.65,max(normal.z,.15)));
        let foam_normal=foam_detail_normal(foam_base_normal,detail,u.environment.w);
        let resolved_roughness=mix(.6,.32,fresh);
        let foam_roughness=min(.9,pow(pow(resolved_roughness,4.0)+.5*foam_detail_variance(detail),.25));
        let foam_reflectance=environment_brdf(dot(foam_normal,view),foam_roughness);
        let foam_albedo=mix(.58,.68,fresh)*detail.b;
        let foam_light=(1.0-foam_reflectance)*foam_albedo*sky_irradiance(foam_normal)
            +foam_reflectance*filtered_reflection(reflect(-view,foam_normal),foam_roughness);
        color=mix(color,foam_light,coverage);
    } else if u.ocean.z<u.statistics.w {
        let crest=textureSample(displacements,wave_sampler,in.uv).w*u.statistics.y+u.statistics.z;
        let coverage=smoothstep(u.ocean.z,u.statistics.w,crest)*u.ocean.w;
        color=mix(color,.65*sky_irradiance(normal),coverage);
    }
    return vec4f(atmosphere(color,in.world,-view),1.0);
}
