Local lighting
==============
Use the viewer's Open HDR sky command, --sky /absolute/path/to/sky.exr,
or ENCINO_WAVES_SKY=/absolute/path/to/sky.hdr.

Alternatively put licensed Dutch Skies latitude/longitude HDR/EXR panoramas
inside assets/local/. That directory is ignored by Git. The program does not
download commercial assets or redistribute them with the source.

This Mac now has DS360_Autumn_Pack_01_03a installed in assets/local/, copied from
the supplied Downloads pack. Its 4000 x 2000 _Ref.hdr is the default panorama;
the 360 x 180 _Env.hdr is the pack's blurred lighting map. Automatic discovery
prefers _Ref panoramas. Both the native viewer and offline renderer use that
default unless a command-line, environment-variable, or saved-scene sky overrides it.
Copy assets/local/ separately when moving this checkout to another workstation.

Dutch Skies 360: Bob Groothuis, Autumn Pack 01, 03a.
The source .ibl descriptor, HDRs and JPEG previews remain together locally.

When no panorama is available the UI explicitly identifies its procedural
fallback. That fallback is not a Dutch Skies photograph.
