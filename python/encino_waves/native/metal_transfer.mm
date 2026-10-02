// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Small C ABI: pack textures on the GPU without CPU access to field data.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ATen/mps/MPSStream.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

struct texture_copy {
    void* texture;
    void* source;
    uint64_t source_offset;
    void* grain;
    uint64_t grain_offset;
    uint32_t width;
    uint32_t height;
    uint32_t foam;
};

struct metal_transfer {
    id<MTLDevice> device;
    id<MTLSharedEvent> event;
    id<MTLComputePipelineState> wave_pipeline;
    id<MTLComputePipelineState> foam_pipeline;
    id<MTLCommandBuffer> last_wait;
    uint64_t value = 0;
};

void require(bool const condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
}

void validate(metal_transfer const& transfer, texture_copy const& copy) {
    auto const texture = (__bridge id<MTLTexture>)copy.texture;
    auto const source = (__bridge id<MTLBuffer>)copy.source;
    require(texture && source, "Missing Metal texture or tensor buffer");
    require(texture.device == transfer.device && source.device == transfer.device,
            "Torch and wgpu must use the same Metal device");
    require(texture.pixelFormat == MTLPixelFormatRGBA16Float &&
            (texture.usage & MTLTextureUsageShaderWrite), "Expected writable RGBA16F texture");
    require(texture.width == copy.width && texture.height == copy.height,
            "Tensor and texture dimensions differ");
    uint64_t const pixels = uint64_t(copy.width) * copy.height;
    uint64_t const source_bytes = pixels * (copy.foam ? 3 : 4) * sizeof(float);
    require(copy.source_offset <= source.length &&
            source_bytes <= source.length - copy.source_offset, "Tensor buffer is too small");
    if (copy.foam) {
        auto const grain = (__bridge id<MTLBuffer>)copy.grain;
        require(grain && grain.device == transfer.device, "Missing Metal foam grain buffer");
        require(copy.grain_offset <= grain.length &&
                pixels * sizeof(float) <= grain.length - copy.grain_offset,
                "Foam grain buffer is too small");
    }
}
} // namespace

extern "C" char const* encino_metal_error() { return last_error.c_str(); }

extern "C" void* encino_metal_create(void* device_ptr, char const* shader_source) {
    @autoreleasepool {
        try {
            auto transfer = std::make_unique<metal_transfer>();
            transfer->device = (__bridge id<MTLDevice>)device_ptr;
            require(transfer->device != nil, "Metal device handle is unavailable");
            transfer->event = [transfer->device newSharedEvent];
            require(transfer->event != nil, "Cannot create Metal synchronization event");
            auto const options = [MTLCompileOptions new];
            options.fastMathEnabled = NO;
            NSError* error = nil;
            auto const library = [transfer->device
                newLibraryWithSource:[NSString stringWithUTF8String:shader_source]
                options:options error:&error];
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            transfer->wave_pipeline = [transfer->device
                newComputePipelineStateWithFunction:[library newFunctionWithName:@"wave_texture"] error:&error];
            transfer->foam_pipeline = [transfer->device
                newComputePipelineStateWithFunction:[library newFunctionWithName:@"foam_texture"] error:&error];
            require(transfer->wave_pipeline && transfer->foam_pipeline,
                    "Cannot create Metal texture packing pipelines");
            return transfer.release();
        } catch (std::exception const& error) { last_error = error.what(); return nullptr; }
    }
}

extern "C" uint64_t encino_metal_upload(void* context, texture_copy const* copies,
                                        uint32_t const count) {
    @autoreleasepool {
        try {
            auto& transfer = *static_cast<metal_transfer*>(context);
            for (uint32_t i = 0; i < count; ++i) validate(transfer, copies[i]);
            auto* const stream = at::mps::getCurrentMPSStream();
            require(stream->device() == transfer.device, "Torch stream and graphics device differ");
            uint64_t const copied = transfer.value + 1;
            // wgpu-hal 29 does not expose its Metal queue. The Python caller
            // fences prior graphics work, then waits on last_wait before drawing.
            at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
                stream->endKernelCoalescing();
                auto const commands = stream->commandBuffer();
                auto const encoder = [commands computeCommandEncoder];
                for (uint32_t i = 0; i < count; ++i) {
                    auto const& copy = copies[i];
                    auto const pipeline = copy.foam ? transfer.foam_pipeline : transfer.wave_pipeline;
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:(__bridge id<MTLBuffer>)copy.source
                                offset:copy.source_offset atIndex:0];
                    if (copy.foam) [encoder setBuffer:(__bridge id<MTLBuffer>)copy.grain
                                              offset:copy.grain_offset atIndex:1];
                    [encoder setTexture:(__bridge id<MTLTexture>)copy.texture atIndex:0];
                    [encoder dispatchThreads:MTLSizeMake(copy.width, copy.height, 1)
                       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                }
                [encoder endEncoding];
                [commands encodeSignalEvent:transfer.event value:copied];
                // Capture the underlying buffer before commitAndContinue swaps
                // it out; waiting on the MPS wrapper would wait on a new buffer.
                transfer.last_wait = commands.commandBuffer;
                stream->synchronize(at::mps::SyncType::COMMIT);
            });
            transfer.value = copied;
            return copied;
        } catch (std::exception const& error) { last_error = error.what(); return 0; }
    }
}

extern "C" uint64_t encino_metal_completed(void* context) {
    return static_cast<metal_transfer*>(context)->event.signaledValue;
}

extern "C" int encino_metal_wait(void* context) {
    @autoreleasepool {
        auto const commands = static_cast<metal_transfer*>(context)->last_wait;
        [commands waitUntilCompleted];
        if (commands.error) { last_error = commands.error.localizedDescription.UTF8String; return 0; }
        return 1;
    }
}

extern "C" void encino_metal_destroy(void* context) {
    @autoreleasepool { delete static_cast<metal_transfer*>(context); }
}
