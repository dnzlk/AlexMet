//
//  Generator.m
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 02.11.2024.
//

@import Foundation;
@import Metal;

#import "Generator.h"

@implementation Generator {
    id<MTLDevice> device;
};

- (instancetype)initWithDevice:(id<MTLDevice>)_device {
    self = [super init];
    device = _device;
    return self;
}

- (id<MTLBuffer>)buffer:(uint)n :(NSString*)label {
    id<MTLBuffer> buffer = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    [buffer setLabel:label];
    return buffer;
}

- (id<MTLBuffer>)gaussian_buffer:(uint)n :(NSString*)label {
    id<MTLBuffer> buffer = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    [buffer setLabel:label];

    float* a = buffer.contents;

    float value = -0.01;
    const float step = (float)0.02 / n;

    for (int i = 0; i < n; i++) {
        a[i] = value;
        value += step;
    }
    for (int i = 0; i < n; i++) {
        int j = arc4random_uniform((u_int32_t)(n - 1));

        float temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    return buffer;
}


- (id<MTLBuffer>)zero_buffer:(uint)n :(NSString*)label {
    float* a = calloc(n, sizeof(float));
    id<MTLBuffer> buffer = [device newBufferWithBytes:a length:n * sizeof(float) options:MTLResourceStorageModeShared];
    [buffer setLabel:label];
    free(a);
    return buffer;
}

- (id<MTLBuffer>)ones_buffer:(uint)n :(NSString*)label {
    id<MTLBuffer> buffer = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    [buffer setLabel:label];
    float* a = buffer.contents;
    for (uint i = 0; i < n; a[i++] = 1);
    return buffer;
}

- (id<MTLBuffer>)shape_buffer:(Shape)shape :(NSString*)label {
    id<MTLBuffer> buffer = [device newBufferWithBytes:&shape length:sizeof(Shape) options:MTLResourceStorageModeShared];
    [buffer setLabel:label];
    return buffer;
}

@end
