//
//  Generator.h
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 02.11.2024.
//
#include "base.h"

NS_ASSUME_NONNULL_BEGIN

@interface Generator : NSObject
- (instancetype) initWithDevice:(id<MTLDevice>)_device;
- (id<MTLBuffer>) buffer: (uint)n :(NSString*)label;
- (id<MTLBuffer>) gaussian_buffer:(uint)n :(NSString*)label;
- (id<MTLBuffer>) zero_buffer:(uint)n :(NSString*)label;
- (id<MTLBuffer>) ones_buffer:(uint)n :(NSString*)label;
- (id<MTLBuffer>)shape_buffer:(Shape)shape :(NSString*)label;
@end

NS_ASSUME_NONNULL_END
