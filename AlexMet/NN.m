//
//  NN.m
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 05.11.2024.
//

@import Foundation;
@import Metal;

#import "NN.h"
#import "Shapes.h"

@implementation NN {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> convFunctionPSO;
    id<MTLComputePipelineState> reluFunctionPSO;
    id<MTLComputePipelineState> mpFunctionPSO;
    id<MTLComputePipelineState> lrnFunctionPSO;
    id<MTLComputePipelineState> dropoutFunctionPSO;
    id<MTLComputePipelineState> matmulFunctionPSO;
    id<MTLComputePipelineState> biasFunctionPSO;

    id<MTLComputePipelineState> reluBWFunctionPSO;
    id<MTLComputePipelineState> biasBWFunctionPSO;
    id<MTLComputePipelineState> dropoutBWFunctionPSO;
    id<MTLComputePipelineState> mpBWFunctionPSO;
    id<MTLComputePipelineState> convBWFunctionPSO;
    id<MTLComputePipelineState> lrnBWFunctionPSO;

    id<MTLComputePipelineState> sgdFunctionPSO;
};

- (instancetype)initWithDevice:(id<MTLDevice>)_device :(id<MTLCommandQueue>)_commandQueue {
    self = [super init];

    device = _device;
    commandQueue = _commandQueue;

    NSError *error = NULL;

    id<MTLLibrary> library = [device newDefaultLibrary];

    id<MTLFunction> convFunction = [library newFunctionWithName:@"conv"];
    convFunctionPSO = [device newComputePipelineStateWithFunction:convFunction error:&error];

    id<MTLFunction> reluFunction = [library newFunctionWithName:@"relu"];
    reluFunctionPSO = [device newComputePipelineStateWithFunction:reluFunction error:&error];

    id<MTLFunction> mpFunction = [library newFunctionWithName:@"max_pool"];
    mpFunctionPSO = [device newComputePipelineStateWithFunction:mpFunction error:&error];

    id<MTLFunction> lrnFunction = [library newFunctionWithName:@"lrn"];
    lrnFunctionPSO = [device newComputePipelineStateWithFunction:lrnFunction error:&error];

    id<MTLFunction> dropoutFunction = [library newFunctionWithName:@"dropout"];
    dropoutFunctionPSO = [device newComputePipelineStateWithFunction:dropoutFunction error:&error];

    id<MTLFunction> matmulFunction = [library newFunctionWithName:@"matmul"];
    matmulFunctionPSO = [device newComputePipelineStateWithFunction:matmulFunction error:&error];

    id<MTLFunction> biasFunction = [library newFunctionWithName:@"bias"];
    biasFunctionPSO = [device newComputePipelineStateWithFunction:biasFunction error:&error];

    id<MTLFunction> reluBWFunction = [library newFunctionWithName:@"relu_bw"];
    reluBWFunctionPSO = [device newComputePipelineStateWithFunction:reluBWFunction error:&error];

    id<MTLFunction> biasBWFunction = [library newFunctionWithName:@"bias_bw"];
    biasBWFunctionPSO = [device newComputePipelineStateWithFunction:biasBWFunction error:&error];

    id<MTLFunction> dropoutBWFunction = [library newFunctionWithName:@"dropout_bw"];
    dropoutBWFunctionPSO = [device newComputePipelineStateWithFunction:dropoutBWFunction error:&error];

    id<MTLFunction> mpBWFunction = [library newFunctionWithName:@"max_pool_bw"];
    mpBWFunctionPSO = [device newComputePipelineStateWithFunction:mpBWFunction error:&error];

    id<MTLFunction> convBWFunction = [library newFunctionWithName:@"conv_bw"];
    convBWFunctionPSO = [device newComputePipelineStateWithFunction:convBWFunction error:&error];

    id<MTLFunction> lrnBWFunction = [library newFunctionWithName:@"lrn_bw"];
    lrnBWFunctionPSO = [device newComputePipelineStateWithFunction:lrnBWFunction error:&error];

    id<MTLFunction> sgdFunction = [library newFunctionWithName:@"sgd"];
    sgdFunctionPSO = [device newComputePipelineStateWithFunction:sgdFunction error:&error];

    return self;
}

- (void)conv
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)W
:(id<MTLBuffer>)b
:(uint)k
:(uint)p
:(uint)s
:(id<MTLBuffer>)out
:(Shape) osh
:(id<MTLBuffer>)osh_buffer 
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:convFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:insh offset:0 atIndex:1];
    [encoder setBuffer:W offset:0 atIndex:2];
    [encoder setBuffer:b offset:0 atIndex:3];
    [encoder setBytes:&k length:sizeof(uint) atIndex:4];
    [encoder setBytes:&p length:sizeof(uint) atIndex:5];
    [encoder setBytes:&s length:sizeof(uint) atIndex:6];
    [encoder setBuffer:out offset:0 atIndex:7];
    [encoder setBuffer:osh_buffer offset:0 atIndex:8];
    u_long w = convFunctionPSO.threadExecutionWidth;
    u_long h = convFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, osh.C);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)conv_bw
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)W
:(id<MTLBuffer>)b
:(uint)k
:(uint)p
:(uint)s
:(id<MTLBuffer>)dout
:(Shape) osh
:(id<MTLBuffer>)osh_buffer
:(id<MTLBuffer>)din
:(id<MTLBuffer>)dW
:(id<MTLBuffer>)db
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:convBWFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:insh offset:0 atIndex:1];
    [encoder setBuffer:W offset:0 atIndex:2];
    [encoder setBuffer:b offset:0 atIndex:3];
    [encoder setBytes:&k length:sizeof(uint) atIndex:4];
    [encoder setBytes:&p length:sizeof(uint) atIndex:5];
    [encoder setBytes:&s length:sizeof(uint) atIndex:6];
    [encoder setBuffer:dout offset:0 atIndex:7];
    [encoder setBuffer:osh_buffer offset:0 atIndex:8];
    [encoder setBuffer:din offset:0 atIndex:9];
    [encoder setBuffer:dW offset:0 atIndex:10];
    [encoder setBuffer:db offset:0 atIndex:11];
    u_long w = convBWFunctionPSO.threadExecutionWidth;
    u_long h = convBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)relu
:(id<MTLBuffer>)in
:(id<MTLBuffer>)out
:(uint)height
:(uint)width
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:reluFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBytes:&width length:sizeof(uint) atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    u_long w = reluFunctionPSO.threadExecutionWidth;
    u_long h = reluFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)relu_bw
:(id<MTLBuffer>)in
:(uint)height
:(uint)width
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)din
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:reluBWFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBytes:&width length:sizeof(uint) atIndex:1];
    [encoder setBuffer:dout offset:0 atIndex:2];
    [encoder setBuffer:din offset:0 atIndex:3];
    u_long w = reluBWFunctionPSO.threadExecutionWidth;
    u_long h = reluBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)max_pool
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)out
:(id<MTLBuffer>)osh_buffer
:(Shape)osh
:(id<MTLBuffer>)idxs
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:mpFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:insh offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBuffer:osh_buffer offset:0 atIndex:3];
    [encoder setBuffer:idxs offset:0 atIndex:4];
    u_long w = mpFunctionPSO.threadExecutionWidth;
    u_long h = mpFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, osh.C);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)max_pool_bw
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)osh_buffer
:(Shape)osh
:(id<MTLBuffer>)idxs
:(id<MTLBuffer>)din
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:mpBWFunctionPSO];
    [encoder setBuffer:dout offset:0 atIndex:0];
    [encoder setBuffer:osh_buffer offset:0 atIndex:1];
    [encoder setBuffer:idxs offset:0 atIndex:2];
    [encoder setBuffer:din offset:0 atIndex:3];
    u_long w = mpBWFunctionPSO.threadExecutionWidth;
    u_long h = mpBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)lrn
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)out
:(Shape)osh 
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:lrnFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:insh offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    u_long w = lrnFunctionPSO.threadExecutionWidth;
    u_long h = lrnFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, osh.N);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)lrn_bw
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)dout
:(Shape)osh
:(id<MTLBuffer>)din
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:lrnBWFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:insh offset:0 atIndex:1];
    [encoder setBuffer:dout offset:0 atIndex:2];
    [encoder setBuffer:din offset:0 atIndex:3];
    u_long w = lrnBWFunctionPSO.threadExecutionWidth;
    u_long h = lrnBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(osh.W, osh.H, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)dropout
:(id<MTLBuffer>)in
:(id<MTLBuffer>)out
:(uint)height
:(uint)width
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:dropoutFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBytes:&width length:sizeof(uint) atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    u_long w = dropoutFunctionPSO.threadExecutionWidth;
    u_long h = dropoutFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)dropout_bw
:(id<MTLBuffer>)out
:(uint)height
:(uint)width
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)din
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:dropoutBWFunctionPSO];
    [encoder setBuffer:out offset:0 atIndex:0];
    [encoder setBytes:&width length:sizeof(uint) atIndex:1];
    [encoder setBuffer:dout offset:0 atIndex:2];
    [encoder setBuffer:din offset:0 atIndex:3];
    u_long w = dropoutBWFunctionPSO.threadExecutionWidth;
    u_long h = dropoutBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)matmul
:(id<MTLBuffer>)in
:(id<MTLBuffer>)W
:(uint)bs
:(uint)fan_in
:(uint)fan_out
:(id<MTLBuffer>)out
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:matmulFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:W offset:0 atIndex:1];
    [encoder setBytes:&fan_in length:sizeof(uint) atIndex:2];
    [encoder setBytes:&fan_out length:sizeof(uint) atIndex:3];
    [encoder setBuffer:out offset:0 atIndex:4];
    u_long w = matmulFunctionPSO.threadExecutionWidth;
    u_long h = matmulFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(fan_out, bs, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)bias
:(id<MTLBuffer>)in
:(id<MTLBuffer>)b
:(uint)bs
:(uint)fan_out
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:biasFunctionPSO];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBytes:&fan_out length:sizeof(uint) atIndex:2];
    u_long w = biasFunctionPSO.threadExecutionWidth;
    u_long h = biasFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(fan_out, bs, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)bias_bw
:(id<MTLBuffer>)dout
:(uint)bs
:(uint)fan_out
:(id<MTLBuffer>)db
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:biasBWFunctionPSO];
    [encoder setBuffer:dout offset:0 atIndex:0];
    [encoder setBytes:&fan_out length:sizeof(uint) atIndex:1];
    [encoder setBuffer:db offset:0 atIndex:2];
    u_long w = biasBWFunctionPSO.threadExecutionWidth;
    u_long h = biasBWFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(fan_out, bs, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (float)cross_entropy
:(float*)in
:(int*)Y
:(int)bs
{
    float* in_copy = malloc(bs * 1000 * sizeof(float));
    memcpy(in_copy, in, bs * 1000 * sizeof(float));

    float mean_sum = 0.0;
    for (int i = 0; i < bs; i++) {
        sub_max(&in_copy[i * 1000]);

        float probs[1000];
        softmax(&in_copy[i * 1000], probs);

        float predicted_prob = -log(probs[Y[i]]);
        mean_sum += predicted_prob;
    }
    free(in_copy);
    return mean_sum / bs;
}

- (void)cross_entropy_bw
:(float*)in
:(int*)Y
:(int)bs
:(float*)din
{
    float* in_copy = malloc(bs * 1000 * sizeof(float));
    memcpy(in_copy, in, bs * 1000 * sizeof(float));

    for (int i = 0; i < bs; i++) {
        sub_max(&in_copy[i * 1000]);

        softmax(&in_copy[i * 1000], &din[i * 1000]);
        din[i * 1000 + Y[i]] -= 1;
    }
    for (int i = 0; i < bs * 1000; i++)
        din[i] /= bs;
    free(in_copy);
}

- (void)sgd
:(id<MTLBuffer>)weights
:(id<MTLBuffer>)dw
:(id<MTLBuffer>)v
:(uint)width
:(uint)height
:(float)lr
:(id<MTLCommandBuffer>) commandBuffer
{
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:sgdFunctionPSO];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:dw offset:0 atIndex:1];
    [encoder setBuffer:v offset:0 atIndex:2];
    [encoder setBytes:&width length:sizeof(uint) atIndex:3];
    [encoder setBytes:&lr length:sizeof(float) atIndex:4];
    u_long w = sgdFunctionPSO.threadExecutionWidth;
    u_long h = sgdFunctionPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threads];
    [encoder endEncoding];
}

void softmax(float* in, float* out) {
    float e[1000];
    float exp_sum = 0;

    for (int i = 0; i < 1000; i++) {
        e[i] = exp(in[i]);
        exp_sum += e[i];
    }
    for (int i = 0; i < 1000; i++)
        out[i] = e[i] / exp_sum;
}

void sub_max(float* in) {
    float max = in[0];
    for (int i = 1; i < 1000; i++)
        if (in[i] > max)
            max = in[i];
    for (int i = 0; i < 1000; i++)
        in[i] -= max;
}

@end
