# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from imgui_bundle import imgui, portable_file_dialogs as file_dialogs
from rendercanvas.glfw import RenderCanvas, loop
from wgpu.utils.imgui import ImguiRenderer
import wgpu
from .model import Wave_parameters, Phase_step, evaluate
from .editing import (make_wave_basis, state_from_basis, same_wave_basis,
                      follow_parameters, edit_state, restore_phase)
from .render import Ocean_renderer, make_device, Camera, Look, Shading_statistics
from .presets import SCENES, scene_with_resolution
from .camera import frame_domain
from .foam import Foam_parameters, update_foam, save_foam, load_foam


class Viewer:
    def __init__(self, resolution=1024, device="auto", sky=None, preset=2, max_frames=0, scene=None, canvas=None):
        self.canvas=canvas or RenderCanvas(size=(1440,900),title="Encino Waves",update_mode="continuous",max_fps=60)
        self.device=make_device(self.canvas)
        self.context=self.canvas.get_wgpu_context()
        self.format=self.context.get_preferred_format(self.device.adapter).removesuffix("-srgb")
        self.context.configure(device=self.device,format=self.format,usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
        self.renderer=Ocean_renderer(self.device,sky,target_format=self.format)
        self.comparison_renderer=None
        self.sky_path=sky
        self.device_name=device
        self.selected_scene=preset
        initial_scene=scene_with_resolution(preset,resolution)
        self.parameters,self.camera,self.look=initial_scene.parameters,initial_scene.camera,initial_scene.look
        self.basis=make_wave_basis(self.parameters,device)
        self.state=state_from_basis(self.basis,self.parameters)
        self.post_seed=True
        self.reset_requested=False
        self.rendered_state=None
        self.rendered_comparison_state=None
        self.saved_shading=None
        self.saved_comparison_shading=None
        self.comparison_state=None
        self.comparing=False
        self.tessendorf_only=False
        self.show_ui=True
        self.playing=True
        self.time=10.0
        self.foam_parameters=Foam_parameters()
        self.foam_state=None
        self.comparison_foam_state=None
        self.speed=1.0
        self.last_time=time.perf_counter()
        self.last_update=0.0
        self.changed=False
        self.future=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="ocean_setup")
        self.message=""
        self.frame_times=[]
        self.frames=0
        self.max_frames=max_frames
        self.save_requested=False
        self.dialog=None
        self.dialog_action=None
        self.movie_process=None
        self.movie_log=None
        self.movie_seconds=10.0
        self.movie_fps=24
        self.movie_resolution=2048
        self.movie_compact=True
        self.pointer=None
        self.pointer_buttons=()
        self.pointer_modifiers=()
        self.gui=ImguiRenderer(self.device,self.canvas,self.format)
        self.gui.set_gui(self.draw_gui)
        self._style()
        self.canvas.add_event_handler(self.on_key,"key_down")
        self.canvas.add_event_handler(self.on_pointer,"pointer_down","pointer_move","pointer_up","wheel")
        if scene: self.load_scene(scene)
        self.canvas.request_draw(self.draw)

    def _style(self):
        style=imgui.get_style()
        style.window_rounding=12
        style.frame_rounding=5
        style.grab_rounding=5
        style.window_padding=(20,18)
        style.item_spacing=(10,10)
        style.frame_padding=(10,7)
        colors={
            imgui.Col_.window_bg:(.024,.044,.055,.94),
            imgui.Col_.text:(.91,.93,.92,1),imgui.Col_.text_disabled:(.48,.61,.64,1),
            imgui.Col_.frame_bg:(.09,.15,.18,1),imgui.Col_.frame_bg_hovered:(.14,.24,.28,1),
            imgui.Col_.frame_bg_active:(.16,.29,.33,1),
            imgui.Col_.button:(.08,.17,.20,1),imgui.Col_.button_hovered:(.13,.29,.32,1),
            imgui.Col_.button_active:(.16,.40,.42,1),
            imgui.Col_.slider_grab:(.48,.80,.78,1),imgui.Col_.slider_grab_active:(.69,.94,.87,1),
            imgui.Col_.check_mark:(.6,.88,.80,1),imgui.Col_.header:(.10,.27,.29,1),
            imgui.Col_.header_hovered:(.15,.35,.37,1),imgui.Col_.border:(.19,.30,.32,.5),
        }
        for key,value in colors.items(): style.set_color_(key,value)
        io=imgui.get_io()
        io.set_ini_filename("")
        fonts=[Path("/System/Library/Fonts/Supplemental/Avenir Next.ttc"),
               Path("/System/Library/Fonts/Supplemental/Arial.ttf")]
        for font in fonts:
            if font.exists():
                self.font=io.fonts.add_font_from_file_ttf(str(font),18)
                io.font_default=self.font
                break
        else: self.font=None

    def edit_parameters(self,**kwargs):
        self.parameters=replace(self.parameters,**kwargs)
        self.last_update=time.perf_counter()
        self.changed=True

    def select_scene(self,index):
        self.selected_scene=index
        scene=scene_with_resolution(index,self.parameters.resolution)
        self.parameters,self.camera,self.look=scene.parameters,scene.camera,scene.look
        self.last_update=time.perf_counter()
        self.changed=True
        self.reset_requested=True
        self.reset_foam()

    def reset_foam(self):
        self.foam_state=None
        self.comparison_foam_state=None

    def _start_update(self):
        # Only lattice / random-basis edits need a CPU build. Physical controls
        # feed the post-seed GPU stage on every displayed frame, without debounce.
        self.future=self.executor.submit(make_wave_basis,self.parameters,self.device_name)

    def _update_parameters(self,elapsed):
        desired=self.parameters.tessendorf() if self.tessendorf_only else self.parameters
        if not same_wave_basis(self.basis.parameters,desired):
            if not self.future and time.perf_counter()-self.last_update>.2:
                self._start_update()
            return
        if self.reset_requested:
            self.state=state_from_basis(self.basis,desired)
            self.comparison_state=None
            self.post_seed=True
            self.reset_requested=False
        parameters=follow_parameters(self.state.parameters,desired,elapsed)
        if parameters!=self.state.parameters:
            self.state=edit_state(self.basis,self.state,parameters,self.time)
            self.post_seed=True
        if self.comparing:
            parameters=self.state.parameters.tessendorf()
            if self.comparison_state is None:
                self.comparison_state=state_from_basis(self.basis,parameters)
            elif self.comparison_state.parameters!=parameters:
                self.comparison_state=edit_state(self.basis,self.comparison_state,parameters,self.time)
            # Both panels use exactly the same travelling-wave phase history.
            if self.comparison_state.phase is not self.state.phase:
                self.comparison_state=replace(self.comparison_state,phase=self.state.phase,
                                              phase_steps=self.state.phase_steps)
        else:
            self.comparison_state=None
        self.changed=self.state.parameters!=desired

    def _upload_state(self,renderer,state,previous):
        # An orientation edit only changes the mesh transform. A paused ocean
        # can keep its FFT results and textures, including in comparison view.
        same_fields=previous is not None and all(
            getattr(state,name) is getattr(previous,name)
            for name in ("h_positive","h_negative","omega","multipliers","phase"))
        same_fields=same_fields and state.phase_steps==previous.phase_steps and (
            state.parameters.in_ocean_space()==previous.parameters.in_ocean_space())
        if renderer.frame is None or renderer.frame.time!=self.time or not same_fields:
            renderer.upload(evaluate(state,self.time))
        elif renderer.frame.parameters!=state.parameters:
            renderer.frame=replace(renderer.frame,parameters=state.parameters)

    def draw(self):
        now=time.perf_counter()
        delta=min(now-self.last_time,.1)
        self.last_time=now
        self._poll_files()
        # Occluded / minimized windows cancel here before doing ocean work.
        target=self.context.get_current_texture()
        width,height=self.canvas.get_physical_size()
        if width < 1 or height < 1: return
        if self.playing: self.time+=delta*self.speed
        if self.future and self.future.done():
            try:
                basis=self.future.result()
                # A newer grid edit can supersede a build in flight.
                if same_wave_basis(basis.parameters,self.parameters):
                    self.basis=basis
                    self.reset_requested=True
                self.message=""
            except Exception as error:
                self.message=str(error)
            self.future=None
        self._update_parameters(delta)
        self._upload_state(self.renderer,self.state,self.rendered_state)
        self.rendered_state=self.state
        if self.saved_shading is not None:
            self.renderer.restore_shading_statistics(self.saved_shading)
            self.saved_shading=None
        shading=self.renderer.shading_statistics
        self.foam_state=update_foam(self.foam_state,self.renderer.frame,self.foam_parameters,shading.crest_gain,shading.crest_bias)
        self.renderer.upload_foam(self.foam_state)
        view=target.create_view()
        if self.comparing and self.comparison_state is not None:
            if self.comparison_renderer is None:
                self.comparison_renderer=Ocean_renderer(self.device,self.sky_path,target_format=self.format)
            self._upload_state(self.comparison_renderer,self.comparison_state,self.rendered_comparison_state)
            self.rendered_comparison_state=self.comparison_state
            if self.saved_comparison_shading is not None:
                self.comparison_renderer.restore_shading_statistics(self.saved_comparison_shading)
                self.saved_comparison_shading=None
            shading=self.comparison_renderer.shading_statistics
            self.comparison_foam_state=update_foam(self.comparison_foam_state,self.comparison_renderer.frame,self.foam_parameters,shading.crest_gain,shading.crest_bias)
            self.comparison_renderer.upload_foam(self.comparison_foam_state)
            half=width//2
            self.comparison_renderer.draw(view,width,height,self.camera,self.look,viewport=(0,0,half,height))
            self.renderer.draw(view,width,height,self.camera,self.look,viewport=(half,0,width-half,height),clear=False)
        else:
            self.renderer.draw(view,width,height,self.camera,self.look)
        self.gui.render()
        if self.save_requested:
            self.save_requested=False
            self.save_still()
        self.frames+=1
        elapsed=time.perf_counter()-now
        self.frame_times=(self.frame_times+[elapsed])[-90:]
        if self.max_frames and self.frames>=self.max_frames:
            self.canvas.close()

    def save_still(self):
        directory=Path("renders")
        directory.mkdir(exist_ok=True)
        name=time.strftime("encino_%Y%m%d_%H%M%S")
        other=self.comparison_renderer if self.comparing else None
        image=self.renderer.render_image(1920,1080,self.camera,self.look,left_renderer=other)
        Image.fromarray(image).save(directory/f"{name}.png")
        self.save_scene(directory/f"{name}.json")
        self.message=f"Saved renders/{name}.png"

    def save_scene(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        sky=self.sky_path or self.renderer.sky_name
        metadata={"parameters":asdict(self.state.parameters),"camera":asdict(self.camera),"look":asdict(self.look),
                  "requested_parameters":asdict(self.parameters),"tessendorf_only":self.tessendorf_only,
                  "time":self.time,"sky":str(Path(sky).resolve()) if Path(sky).is_file() else None,
                  "post_seed":self.post_seed,
                  "phase_steps":[asdict(step) for step in self.state.phase_steps],
                  "comparison":self.comparing,"shading_statistics":asdict(self.renderer.shading_statistics),
                  "comparison_shading_statistics":asdict(self.comparison_renderer.shading_statistics)
                      if self.comparing and self.comparison_renderer is not None else None}
        metadata["foam"]=asdict(self.foam_parameters)
        for name,state in (("foam_state",self.foam_state),("comparison_foam_state",self.comparison_foam_state if self.comparing else None)):
            metadata[name]=None
            if state is not None:
                checkpoint=path.with_name(path.stem+"."+name+".npz")
                save_foam(checkpoint,state)
                metadata[name]=checkpoint.name
        path.write_text(json.dumps(metadata,indent=2)+"\n")

    def load_scene(self,path):
        saved=json.loads(Path(path).read_text())
        self.parameters=Wave_parameters(**saved.get("requested_parameters",saved["parameters"]))
        self.camera=Camera(**saved["camera"])
        self.look=Look.from_dict(saved["look"])
        self.time=saved.get("time",10.0)
        self.foam_parameters=Foam_parameters.from_dict(saved.get("foam",{"enabled":False}))
        self.foam_state=load_foam(Path(path).parent/saved["foam_state"],self.device_name) if saved.get("foam_state") else None
        self.comparison_foam_state=load_foam(Path(path).parent/saved["comparison_foam_state"],self.device_name) if saved.get("comparison_foam_state") else None
        self.comparing=saved.get("comparison",False)
        self.tessendorf_only=saved.get("tessendorf_only",False)
        if saved.get("sky"): self.load_sky(saved["sky"])
        parameters=self.parameters.tessendorf() if self.tessendorf_only else self.parameters
        self.basis=make_wave_basis(parameters,self.device_name)
        self.post_seed=True
        self.state=state_from_basis(self.basis,parameters)
        self.state=restore_phase(self.basis,self.state,(Phase_step(**step) for step in saved.get("phase_steps",())))
        # A queued build may finish, but must not replace an explicitly opened scene.
        self.future=None
        self.comparison_state=None
        if self.comparing:
            p=parameters.tessendorf()
            other=state_from_basis(self.basis,p)
            self.comparison_state=replace(other,phase=self.state.phase,phase_steps=self.state.phase_steps)
        self.saved_shading=Shading_statistics(**saved["shading_statistics"]) if saved.get("shading_statistics") else None
        self.saved_comparison_shading=Shading_statistics(**saved["comparison_shading_statistics"]) if saved.get("comparison_shading_statistics") else None
        self.changed=False
        self.reset_requested=False
        self.message=f"Loaded {Path(path).name}"

    def load_sky(self,path):
        renderer=Ocean_renderer(self.device,path,target_format=self.format)
        if self.renderer.frame is not None:
            self.saved_shading=self.renderer.shading_statistics
        if self.comparison_renderer is not None and self.comparison_renderer.frame is not None:
            self.saved_comparison_shading=self.comparison_renderer.shading_statistics
        self.renderer=renderer
        self.sky_path=Path(path)
        self.comparison_renderer=None

    def _open_file(self,action):
        if self.dialog: return
        self.dialog_action=action
        if action=="sky":
            self.dialog=file_dialogs.open_file("Open HDR sky",str(Path.home()),["HDR skies","*.hdr *.exr","All files","*"])
        elif action=="scene":
            self.dialog=file_dialogs.open_file("Open ocean scene",str(Path("renders").resolve()),["Ocean scenes","*.json"])
        elif action=="movie":
            self.dialog=file_dialogs.save_file("Export movie",str(Path("renders/encino.mp4").resolve()),["H.264 movie","*.mp4"])

    def _poll_files(self):
        if self.dialog and self.dialog.ready(0):
            result=self.dialog.result()
            action=self.dialog_action
            self.dialog=None
            path=result[0] if isinstance(result,list) and result else result
            if path:
                try:
                    if action=="sky": self.load_sky(path)
                    elif action=="scene": self.load_scene(path)
                    else: self.export_movie(path)
                except Exception as error: self.message=str(error)
        if self.movie_process and self.movie_process.poll() is not None:
            success=self.movie_process.returncode==0
            self.movie_log.close()
            self.movie_process=None
            self.message="Movie export complete" if success else "Movie export failed; see renders/export.log"

    def export_movie(self,path):
        if self.movie_process: return
        directory=Path("renders")
        directory.mkdir(exist_ok=True)
        scene=directory/f"export_{time.time_ns()}.json"
        self.save_scene(scene)
        command=[sys.executable,"-m","encino_waves","render",str(path),"--scene",str(scene),
                 "--resolution",str(self.movie_resolution),"--seconds",str(self.movie_seconds),
                 "--fps",str(self.movie_fps),"--device",self.device_name,"--no-captions","--overwrite"]
        if self.comparing: command.append("--compare")
        if self.movie_compact: command.extend(("--max-mbps","8"))
        self.movie_log=(directory/"export.log").open("w")
        self.movie_process=subprocess.Popen(command,stdout=self.movie_log,stderr=subprocess.STDOUT)
        self.playing=False
        self.message=f"Exporting {Path(path).name}"

    def on_key(self,event):
        if imgui.get_io().want_text_input: return
        key=event.get("key","")
        if key==" ": self.playing=not self.playing
        elif key=="Tab": self.show_ui=not self.show_ui
        elif key.lower()=="c":
            self.comparing=not self.comparing
            self.tessendorf_only=False
            self.changed=True
        elif key.lower()=="s":
            if self.changed or self.future: self.message="Wait for the parameter update before saving."
            else: self.save_requested=True
        elif key.lower()=="r": self.select_scene(self.selected_scene)
        elif key.lower()=="f": self.camera=frame_domain(self.parameters.domain)
        elif key in "123456" and len(key)==1: self.select_scene(int(key)-1)
        elif key=="Escape": self.show_ui=True
        elif key=="F11":
            import glfw
            window=self.canvas._window
            monitor=glfw.get_window_monitor(window)
            if monitor:
                glfw.set_window_monitor(window,None,80,80,1440,900,0)
            else:
                monitor=glfw.get_primary_monitor()
                mode=glfw.get_video_mode(monitor)
                glfw.set_window_monitor(window,monitor,0,0,mode.size.width,mode.size.height,mode.refresh_rate)

    def on_pointer(self,event):
        if imgui.get_io().want_capture_mouse: return
        kind=event["event_type"]
        width,height=self.canvas.get_logical_size()
        if kind=="pointer_down":
            self.pointer=(event["x"],event["y"])
            self.pointer_buttons=event.get("buttons",(event.get("button",1),))
            self.pointer_modifiers=event.get("modifiers",())
        elif kind=="pointer_up": self.pointer=None
        elif kind=="pointer_move" and self.pointer:
            dx,dy=event["x"]-self.pointer[0],event["y"]-self.pointer[1]
            buttons=event.get("buttons",self.pointer_buttons)
            modifiers=event.get("modifiers",self.pointer_modifiers)
            # Maya mapping, plus the original viewer's Mac shortcuts.
            if "Alt" in modifiers:
                if 2 in buttons or (1 in buttons and 3 in buttons):
                    self.camera=self.camera.dolly(dx,width)
                elif 1 in buttons: self.camera=self.camera.orbit(dx,dy,width,height)
                elif 3 in buttons: self.camera=self.camera.track(dx,dy,width,height)
            elif "Control" in modifiers and 1 in buttons:
                self.camera=self.camera.track(dx,dy,width,height)
            elif "Shift" in modifiers and 1 in buttons:
                self.camera=self.camera.dolly(dx,width)
            self.pointer=(event["x"],event["y"])
        elif kind=="wheel":
            self.camera=self.camera.dolly(-event["dy"]*.15,width)

    def _label(self,label,size=14,color=(.48,.72,.74,1)):
        imgui.push_font(self.font,size)
        imgui.text_colored(color,label)
        imgui.pop_font()

    def _slider(self,label,name,lo,hi,fmt,help_text,log=False):
        imgui.text(label)
        imgui.set_next_item_width(-1)
        flags=imgui.SliderFlags_.logarithmic if log else 0
        changed,value=imgui.slider_float("##"+name,getattr(self.parameters,name),lo,hi,fmt,flags)
        if changed: self.edit_parameters(**{name:value})
        if imgui.is_item_hovered(): imgui.set_tooltip(help_text)

    def draw_gui(self):
        width,height=self.canvas.get_logical_size()
        fixed=imgui.WindowFlags_.no_decoration|imgui.WindowFlags_.no_move|imgui.WindowFlags_.no_saved_settings
        if not self.show_ui:
            return
        panel_width=310
        imgui.set_next_window_pos((width-panel_width-24,24))
        imgui.set_next_window_size((panel_width,height-48))
        imgui.begin("ocean_controls",flags=fixed)
        self._label("Encino Waves",23,(.92,.94,.92,1))
        imgui.spacing()
        imgui.set_next_item_width(-1)
        changed,index=imgui.combo("##scene",self.selected_scene,[s.name for s in SCENES])
        if changed: self.select_scene(index)
        imgui.spacing()
        self._slider("Wind speed","wind_speed",1,500,"%.1f m/s","Wind speed at 10 m. Edits reshape the existing waves continuously.",True)
        self._slider("Fetch","fetch_km",1,5000,"%.0f km","How far the wind has had to build the sea.",True)
        self._slider("Ocean depth","depth",.25,1000,"%.1f m","Depth changes the wave spectrum and speed while preserving travelling-wave phase.",True)
        self._slider("Swell","swell",-1,2,"%.2f","0: empirical spreading; positive: narrower swell; -1: all directions equally.")
        self._slider("Wind direction","wind_direction",-180,180,"%.0f degrees","Rotate the same ocean. 0 degrees is +X; 90 degrees is +Y.")
        imgui.text("Directional spreading")
        models=["donelan_banner","hasselmann","mitsuyasu","cosine_squared"]
        labels=["Donelan-Banner","Hasselmann","Mitsuyasu","Cosine squared"]
        imgui.set_next_item_width(-1)
        changed,index=imgui.combo("##direction",models.index(self.parameters.spreading),labels)
        if changed: self.edit_parameters(spreading=models[index])
        imgui.spacing()
        imgui.separator()
        changed,value=imgui.checkbox("Compare with earlier model",self.comparing)
        if changed:
            self.comparing=value
            self.tessendorf_only=False
            self.changed=True
        if imgui.is_item_hovered(): imgui.set_tooltip("Pierson-Moskowitz + cosine squared. Same seed, camera, scale and shading.")
        changed,value=imgui.checkbox("Tessendorf mode",self.tessendorf_only)
        if changed:
            self.tessendorf_only=value
            self.comparing=False
            self.changed=True
        imgui.separator()
        if imgui.button("Pause" if self.playing else "Play",(116,0)): self.playing=not self.playing
        imgui.same_line()
        imgui.begin_disabled(self.changed or self.future is not None)
        if imgui.button("Save still",(116,0)): self.save_requested=True
        imgui.end_disabled()
        changed,value=imgui.slider_float("Time",self.time,0,300,"%.1f s")
        if changed:
            self.time=value
            self.reset_foam()
        if imgui.collapsing_header("Foam & aeration"):
            changed,value=imgui.checkbox("Persistent foam",self.foam_parameters.enabled)
            if changed:
                self.foam_parameters=replace(self.foam_parameters,enabled=value)
                self.reset_foam()
            if imgui.is_item_hovered(): imgui.set_tooltip("Build surface foam and underwater bubbles over time. Off restores the original crest shading.")
            if imgui.button("Reset foam"): self.reset_foam()
            for label,name,lo,hi,fmt in (
                ("Emission","emission",0,5,"%.2f /s"),
                ("Surface lifetime","surface_half_life",.3,20,"%.1f s"),
                ("Breakup","breakup",0,1,"%.2f"),
                ("Spreading","diffusion",0,1,"%.2f m2/s"),
                ("Shallow to deep","exchange",0,1,"%.2f /s")):
                changed,value=imgui.slider_float(label,getattr(self.foam_parameters,name),lo,hi,fmt)
                if changed: self.foam_parameters=replace(self.foam_parameters,**{name:value})
            changed,value=imgui.slider_float("Underwater bubbles",self.look.aeration,0,2,"%.2f")
            if changed: self.look=replace(self.look,aeration=value)
            sizes=[256,512,1024,2048]
            # Small test / API maps are valid even though the UI starts at 256.
            if self.foam_parameters.resolution not in sizes: sizes=sorted(sizes+[self.foam_parameters.resolution])
            changed,index=imgui.combo("Foam map",sizes.index(self.foam_parameters.resolution),[str(n) for n in sizes])
            if changed: self.foam_parameters=replace(self.foam_parameters,resolution=sizes[index])
            imgui.text_wrapped("Foam builds during playback. Major sea changes and time scrubbing clear its history.")
        expanded=imgui.collapsing_header("Camera & light")
        if expanded:
            for label,name,lo,hi,fmt in (("Height","height",.5,2000,"%.1f m"),("Pitch","pitch",-89,89,"%.1f deg"),("Heading","yaw",-180,180,"%.1f deg")):
                changed,value=imgui.slider_float(label,getattr(self.camera,name),lo,hi,fmt)
                if changed: self.camera=replace(self.camera,**{name:value})
            for label,name,lo,hi,fmt in (("Exposure","exposure",-4,4,"%.1f stops"),("Sky rotation","sky_rotation",-180,180,"%.0f deg"),("Foam","foam",0,1,"%.2f")):
                changed,value=imgui.slider_float(label,getattr(self.look,name),lo,hi,fmt)
                if changed: self.look=replace(self.look,**{name:value})
            if imgui.button("Open HDR sky..."): self._open_file("sky")
        expanded=imgui.collapsing_header("Resolution & model")
        if expanded:
            sizes=[16,32,64,128,256,512,1024,2048,4096]
            changed,index=imgui.combo("Resolution",sizes.index(self.parameters.resolution),[f"{n} x {n}" for n in sizes])
            if changed: self.edit_parameters(resolution=sizes[index])
            self._slider("Ocean patch","domain",50,4000,"%.0f m","Physical patch size; waves repeat beyond this distance.",True)
            spectra=["tma","jonswap","pm"]
            changed,index=imgui.combo("Spectrum",spectra.index(self.parameters.spectrum),["TMA","JONSWAP","Pierson-Moskowitz"])
            if changed: self.edit_parameters(spectrum=spectra[index])
            modes=["deep","finite","capillary"]
            changed,index=imgui.combo("Dispersion",modes.index(self.parameters.dispersion),["Deep water","Finite depth","Capillary"])
            if changed: self.edit_parameters(dispersion=modes[index])
            self._slider("Pinch","pinch",-3,3,"%.2f","Horizontal displacement; the original artist control.")
            changed,value=imgui.input_int("Seed",self.parameters.seed)
            if changed: self.edit_parameters(seed=value%2**32)
            imgui.text_wrapped(f"{self.device.adapter.info['device']} / {self.state.device}")
            imgui.text_wrapped(self.renderer.sky_name)
            if self.frame_times: imgui.text(f"Frame work: {1000*np.median(self.frame_times):.1f} ms")
        if imgui.collapsing_header("Files & export"):
            if imgui.button("Open scene..."): self._open_file("scene")
            imgui.text_wrapped("Save still also saves camera, lighting and parameters as a scene.")
            _,self.movie_seconds=imgui.slider_float("Duration",self.movie_seconds,1,300,"%.0f s")
            rates=[24,25,30,60]
            changed,index=imgui.combo("FPS",rates.index(self.movie_fps),[str(x) for x in rates])
            if changed: self.movie_fps=rates[index]
            sizes=[512,1024,2048,4096]
            changed,index=imgui.combo("Movie waves",sizes.index(self.movie_resolution),[f"{x} x {x}" for x in sizes])
            if changed: self.movie_resolution=sizes[index]
            _,self.movie_compact=imgui.checkbox("Compact review copy (8 Mbps)",self.movie_compact)
            imgui.begin_disabled(self.movie_process is not None or self.future is not None or self.changed)
            if imgui.button("Export 1080p movie..."): self._open_file("movie")
            imgui.end_disabled()
        if self.future: self._label("Building wave grid...",14,(.91,.76,.48,1))
        elif self.changed: self._label("Adjusting the sea...",14,(.91,.76,.48,1))
        if self.message: imgui.text_wrapped(self.message)
        imgui.end()
        imgui.set_next_window_pos((20,height-42))
        imgui.set_next_window_bg_alpha(.65)
        imgui.begin("navigation",flags=fixed|imgui.WindowFlags_.always_auto_resize|imgui.WindowFlags_.no_inputs)
        self._label("Alt + LMB  tumble    MMB  track    RMB  dolly    F  frame    Tab  hide UI",12,(.8,.85,.85,1))
        imgui.end()
        if self.comparing:
            for x,label in ((width*.15,"EARLIER MODEL"),(width*.60,"ENCINO WAVES")):
                imgui.set_next_window_pos((x,125))
                imgui.set_next_window_bg_alpha(.65)
                imgui.begin(label,flags=fixed|imgui.WindowFlags_.always_auto_resize|imgui.WindowFlags_.no_inputs)
                self._label(label,16,(.92,.94,.89,1))
                imgui.end()

    def run(self):
        print(f"Compute: {self.state.device}; graphics: {dict(self.device.adapter.info)}",flush=True)
        try: loop.run()
        finally: self.executor.shutdown(wait=True,cancel_futures=True)
