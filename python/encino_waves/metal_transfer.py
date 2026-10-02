# Copyright 2026 Christopher Jon Horvath. Apache-2.0.
"""Torch MPS buffers to wgpu Metal textures, without a host field copy.

Only this module knows wgpu's native handles and Torch's Metal storage ABI.
The bridge runs on the current Torch stream. Explicit host fences order the
compute and graphics queues, since wgpu-hal 29 does not expose its Metal queue.
Keep tensors alive until the GPU signals completion:
retaining an MTLBuffer alone does not prevent Torch's allocator from reusing it.
"""
from collections import deque
import ctypes
import hashlib
from pathlib import Path
import platform
import subprocess
import tempfile
import weakref
import torch


class Texture_copy(ctypes.Structure):
    _fields_ = [("texture",ctypes.c_void_p), ("source",ctypes.c_void_p),
                ("source_offset",ctypes.c_uint64), ("grain",ctypes.c_void_p),
                ("grain_offset",ctypes.c_uint64), ("width",ctypes.c_uint32),
                ("height",ctypes.c_uint32), ("foam",ctypes.c_uint32)]


def _load_bridge():
    import fcntl
    source = Path(__file__).parent/"native/metal_transfer.mm"
    torch_root = Path(torch.__file__).parent
    torch_lib = torch_root/"lib"
    identity = (source.read_bytes() + Path(__file__).read_bytes() + str(torch_root).encode() + torch.__version__.encode()
                + platform.machine().encode() + platform.mac_ver()[0].encode()
                + str((torch_lib/"libtorch_cpu.dylib").stat().st_mtime_ns).encode())
    cache = Path(tempfile.gettempdir())/"encino_waves_native"/hashlib.sha256(identity).hexdigest()[:20]
    cache.mkdir(parents=True,exist_ok=True)
    library = cache/"metal_transfer.dylib"
    with (cache/"build.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists():
            temporary = cache/"building.dylib"
            command = ["/usr/bin/xcrun","clang++","-std=c++20","-O2","-dynamiclib",
                       "-fobjc-arc","-fblocks","-framework","Foundation","-framework","Metal",
                       "-framework","MetalPerformanceShaders","-framework","MetalPerformanceShadersGraph",
                       "-I",str(torch_root/"include"),"-L",str(torch_lib),"-ltorch_cpu","-lc10",
                       f"-Wl,-rpath,{torch_lib}",str(source),"-o",str(temporary)]
            result = subprocess.run(command,capture_output=True,text=True,timeout=120)
            if result.returncode:
                raise RuntimeError("Metal bridge compilation failed. Install Apple's Command Line Tools "
                                   "with xcode-select --install.\n"+result.stderr)
            temporary.replace(library)
    bridge = ctypes.CDLL(str(library))
    signatures = {
        "create": ([ctypes.c_void_p,ctypes.c_char_p],ctypes.c_void_p),
        "upload": ([ctypes.c_void_p,ctypes.POINTER(Texture_copy),ctypes.c_uint32],ctypes.c_uint64),
        "completed": ([ctypes.c_void_p],ctypes.c_uint64),
        "wait": ([ctypes.c_void_p],ctypes.c_int),
        "destroy": ([ctypes.c_void_p],None),
        "error": ([],ctypes.c_char_p),
    }
    for name,(args,result) in signatures.items():
        function = getattr(bridge,"encino_metal_"+name)
        function.argtypes,function.restype = args,result
    return bridge


def _buffer(tensor):
    if tensor.device.type != "mps" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError("Metal texture transfer requires contiguous float32 MPS tensors")
    # Same storage representation as ATen's getMTLBufferStorage. data_ptr() on
    # the tensor itself includes its element offset and is not an ObjC handle.
    return tensor.untyped_storage().data_ptr(),tensor.storage_offset()*tensor.element_size()


def _dispose(bridge,context,pending):
    bridge.encino_metal_wait(context)
    pending.clear()
    bridge.encino_metal_destroy(context)


class Metal_transfer:
    name = "Metal GPU transfer"

    def __init__(self,device):
        from wgpu.backends.wgpu_native._ffi import ffi, lib
        self.device = device
        self.ffi,self.lib = ffi,lib
        self.bridge = _load_bridge()
        native_device = int(ffi.cast("uintptr_t",lib.wgpuDeviceGetNativeMetalDevice(device._internal)))
        shader = (Path(__file__).parent/"native/texture_transfer.metal").read_bytes()
        self.context = self.bridge.encino_metal_create(native_device,shader)
        if not self.context: self._error()
        self.pending = deque()
        self._finalizer = weakref.finalize(self,_dispose,self.bridge,self.context,self.pending)

    def _error(self):
        raise RuntimeError(self.bridge.encino_metal_error().decode())

    def _texture(self,texture):
        handle = self.lib.wgpuTextureGetNativeMetalTexture(texture._internal)
        return int(self.ffi.cast("uintptr_t",handle))

    def _submit(self,copies,owners):
        completed = self.bridge.encino_metal_completed(self.context)
        while self.pending and self.pending[0][0] <= completed: self.pending.popleft()
        # Bound retained tensor memory if a producer outruns the display. This
        # is backpressure only, not a wait at every compute/graphics handoff.
        if len(self.pending) >= 6: self.wait()
        # No field readback: wait for previous sampling before overwriting.
        self.lib.wgpuDevicePoll(self.device._internal,True,self.ffi.NULL)
        array = (Texture_copy*len(copies))(*copies)
        ticket = self.bridge.encino_metal_upload(self.context,array,len(copies))
        if not ticket: self._error()
        self.pending.append((ticket,owners))

    def upload_waves(self,frame,textures):
        fields = (frame.displacement,frame.normal)
        n = frame.parameters.resolution
        copies = []
        for tensor,texture in zip(fields,textures):
            if tensor.shape != (n,n,4): raise ValueError("Expected an N×N×4 wave field")
            pointer,offset = _buffer(tensor)
            copies.append(Texture_copy(self._texture(texture),pointer,offset,None,0,n,n,0))
        self._submit(copies,(fields,tuple(textures)))

    def upload_foam(self,state,texture):
        n = state.parameters.resolution
        density,grain = state.density,state.basis.noise_cos[-1]
        if density.shape != (3,n,n) or grain.shape != (n,n):
            raise ValueError("Expected three foam planes and a matching grain plane")
        pointer,offset = _buffer(density)
        grain_pointer,grain_offset = _buffer(grain)
        copy = Texture_copy(self._texture(texture),pointer,offset,grain_pointer,grain_offset,n,n,1)
        self._submit([copy],(density,grain,texture))

    def wait(self):
        if not self.bridge.encino_metal_wait(self.context): self._error()
        self.pending.clear()

    def before_graphics(self):
        self.wait()
