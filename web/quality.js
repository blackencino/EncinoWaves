// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Safari identifies iPhone, but does not expose a dependable hardware model.
// Give current/future capable phones the full ocean; observe actual performance
// instead of guessing the chip from screen dimensions or the iOS version.
export function initialQuality({requested,userAgent='',platform='',maxTouchPoints=0,width=0,height=0}) {
  if(['low','medium','high'].includes(requested))return {quality:requested,source:'url'};
  if(/iPhone/i.test(userAgent)||/iPhone/i.test(platform))return {quality:'high',source:'iphone-auto'};
  const smallTouch=maxTouchPoints>0&&Math.min(width,height)<850;
  return {quality:smallTouch?'low':'medium',source:'default'};
}

export class IPhoneQualityGuard {
  constructor(){this.reset();}
  reset(){this.elapsed=0;this.slow=0;}
  sample(seconds,{quality,source,playing}){
    // Hidden/background gaps, pauses, and explicit choices are not benchmarks.
    if(source!=='iphone-auto'||!playing||!Number.isFinite(seconds)||seconds<=0||seconds>.25){this.reset();return null;}
    this.elapsed+=seconds;
    // Shader compilation, loading, and a brief frame hitch must not lower quality.
    if(this.elapsed<10)return null;
    this.slow=seconds>1/24?this.slow+seconds:Math.max(0,this.slow-2*seconds);
    if(this.slow<6)return null;
    this.reset();
    return quality==='high'?'medium':quality==='medium'?'low':null;
  }
}
