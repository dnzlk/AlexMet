//
//  Shapes.h
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 14.11.2024.
//

#ifndef Shapes_h
#define Shapes_h

#import "Metal/Metal.h"
#import "Generator.h"

@interface Shapes : NSObject

- (instancetype) initWithGenerator:(Generator*)generator;

@property(nonatomic, assign, readonly) uint bs;
@property(nonatomic, assign, readonly) uint classes;
@property(nonatomic, assign, readonly) float lr;

@property(nonatomic, assign, readonly) uint c1_k, c1_p, c1_s;
@property(nonatomic, assign, readonly) uint c2_k, c2_p, c2_s;
@property(nonatomic, assign, readonly) uint c345_k, c345_p, c345_s;

@property(nonatomic, assign, readonly) Shape x;
@property(nonatomic, assign, readonly) Shape c1, w1 , m1;
@property(nonatomic, assign, readonly) Shape c2, w2, m2;
@property(nonatomic, assign, readonly) Shape c3, w3;
@property(nonatomic, assign, readonly) Shape c4, w4;
@property(nonatomic, assign, readonly) Shape c5, w5 , m5;

@property(nonatomic, assign, readonly) uint fc6, w6, b6;
@property(nonatomic, assign, readonly) uint fc7, w7, b7;
@property(nonatomic, assign, readonly) uint fc8, w8, b8;

@property(nonatomic, strong, readonly) id<MTLBuffer> xBuffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> c1Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> m1Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> c2Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> m2Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> c3Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> c4Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> c5Buffer;
@property(nonatomic, strong, readonly) id<MTLBuffer> m5Buffer;

@end

#endif /* Shapes_h */
