//
//  IO.h
//  AlexMet
//
//  Created by Denis Khabarov on 18.11.2024.
//

#ifndef IO_h
#define IO_h

@interface IO : NSObject

- (instancetype) initWithDevice:(id<MTLDevice>)_device;
- (void)load_batch:(int)n :(id<MTLBuffer>)x :(int*)y;
- (void) create_random_batch:(int)n :(char*)name;

@end

#endif /* IO_h */
