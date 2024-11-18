//
//  NN.h
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 05.11.2024.
//

#ifndef NN_h
#define NN_h

#include "base.h"

@interface NN : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)_device :(id<MTLCommandQueue>)_commandQueue;

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
:(id<MTLBuffer>)osh_buffer;

- (void)relu
:(id<MTLBuffer>)in
:(id<MTLBuffer>)out
:(uint)height
:(uint)width;

- (void)max_pool
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)out
:(id<MTLBuffer>)osh_buffer
:(Shape)osh
:(id<MTLBuffer>)idxs;

- (void)lrn
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)out
:(Shape)osh;

- (void)dropout
:(id<MTLBuffer>)in
:(id<MTLBuffer>)out
:(uint)height
:(uint)width;

- (void)matmul
:(id<MTLBuffer>)in
:(id<MTLBuffer>)W
:(uint)bs
:(uint)fan_in
:(uint)fan_out
:(id<MTLBuffer>)out;

- (void)bias
:(id<MTLBuffer>)in
:(id<MTLBuffer>)b
:(uint)bs
:(uint)fan_out;

- (void)relu_bw
:(id<MTLBuffer>)in
:(uint)height
:(uint)width
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)din;

- (void)bias_bw
:(id<MTLBuffer>)dout
:(uint)bs
:(uint)fan_out
:(id<MTLBuffer>)db;

- (void)dropout_bw
:(id<MTLBuffer>)out
:(uint)height
:(uint)width
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)din;

- (void)max_pool_bw
:(id<MTLBuffer>)dout
:(id<MTLBuffer>)osh_buffer
:(Shape)osh
:(id<MTLBuffer>)idxs
:(id<MTLBuffer>)din;

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
:(id<MTLBuffer>)db;

- (void)lrn_bw
:(id<MTLBuffer>)in
:(id<MTLBuffer>)insh
:(id<MTLBuffer>)dout
:(Shape)osh
:(id<MTLBuffer>)din;

- (float)cross_entropy
:(float*)in
:(int*)Y
:(int)bs
:(int)classes;

- (void)cross_entropy_bw
:(float*)in
:(int*)Y
:(int)bs
:(int)classes
:(float*)din;

- (void)sgd
:(id<MTLBuffer>)weights
:(id<MTLBuffer>)dw
:(id<MTLBuffer>)v
:(uint)width
:(uint)height
:(float)lr
:(id<MTLCommandBuffer>) commandBuffer;

@end

#endif /* NN_h */
