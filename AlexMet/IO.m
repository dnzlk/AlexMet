//
//  IO.m
//  AlexMet
//
//  Created by Denis Khabarov on 18.11.2024.
//

#import <Foundation/Foundation.h>
#import "Metal/Metal.h"
#import "IO.h"

@implementation IO {
    id<MTLDevice> device;
};

- (instancetype)initWithDevice:(id<MTLDevice>)_device {
    self = [super init];
    device = _device;
    return self;
}

- (void)create_random_batch:(int)n :(char *)name {
    float* a = malloc(n * sizeof(float));
    for (int i = 0; i < n; a[i++] = (float)random()/(float)RAND_MAX);
    FILE *file = fopen(name, "wb");
    fwrite(a, sizeof(float), n, file);
    fclose(file);
    free(a);
    return;
}

// Temporary random
- (void)load_batch:(int)n :(id<MTLBuffer>)x :(int*)y {
    MTLIOCommandQueueDescriptor* commandQueueDescriptor = [[MTLIOCommandQueueDescriptor alloc] init];
    commandQueueDescriptor.type = MTLIOCommandQueueTypeConcurrent;
    commandQueueDescriptor.priority = MTLIOPriorityHigh;

    NSError* ioError = NULL;
    id<MTLIOCommandQueue> ioCommandQueue = [device newIOCommandQueueWithDescriptor:commandQueueDescriptor error:&ioError];
    id<MTLIOCommandBuffer> ioCommandBuffer = [ioCommandQueue commandBuffer];

    NSString* currentDirectory = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* filePath = [currentDirectory stringByAppendingPathComponent:@"Xtr.bin"];
    NSURL* fileURL = [NSURL fileURLWithPath:filePath];

    id<MTLIOFileHandle> xtrHandle = [device newIOFileHandleWithURL:fileURL error:&ioError];
    [ioCommandBuffer loadBuffer:x
                         offset:0
                           size:n
                   sourceHandle:xtrHandle
             sourceHandleOffset:0];

    [ioCommandBuffer commit];
    [ioCommandBuffer waitUntilCompleted];

    int random_y[128] = {5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2,
        5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2,
        5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2,
        5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2, 5, 1, 2, 3, 5, 4, 3, 2};

    memcpy(y, random_y, 128);
}

@end
