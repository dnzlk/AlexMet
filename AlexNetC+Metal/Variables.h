//
//  Variables.h
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 14.11.2024.
//

#ifndef Variables_h
#define Variables_h

#import <Metal/Metal.h>
#import "Generator.h"
#import "Shapes.h"

@interface Variables : NSObject

- (instancetype) initWithGenerator:(Generator*)generator :(Shapes*)shapes :(id<MTLDevice>)device;

@property(nonatomic, strong) id<MTLBuffer> w1;
@property(nonatomic, strong) id<MTLBuffer> w2;
@property(nonatomic, strong) id<MTLBuffer> w3;
@property(nonatomic, strong) id<MTLBuffer> w4;
@property(nonatomic, strong) id<MTLBuffer> w5;
@property(nonatomic, strong) id<MTLBuffer> w6;
@property(nonatomic, strong) id<MTLBuffer> w7;
@property(nonatomic, strong) id<MTLBuffer> w8;

@property(nonatomic, strong) id<MTLBuffer> v_w1;
@property(nonatomic, strong) id<MTLBuffer> v_w2;
@property(nonatomic, strong) id<MTLBuffer> v_w3;
@property(nonatomic, strong) id<MTLBuffer> v_w4;
@property(nonatomic, strong) id<MTLBuffer> v_w5;
@property(nonatomic, strong) id<MTLBuffer> v_w6;
@property(nonatomic, strong) id<MTLBuffer> v_w7;
@property(nonatomic, strong) id<MTLBuffer> v_w8;

@property(nonatomic, strong) id<MTLBuffer> b1;
@property(nonatomic, strong) id<MTLBuffer> b2;
@property(nonatomic, strong) id<MTLBuffer> b3;
@property(nonatomic, strong) id<MTLBuffer> b4;
@property(nonatomic, strong) id<MTLBuffer> b5;
@property(nonatomic, strong) id<MTLBuffer> b6;
@property(nonatomic, strong) id<MTLBuffer> b7;
@property(nonatomic, strong) id<MTLBuffer> b8;

@property(nonatomic, strong) id<MTLBuffer> v_b1;
@property(nonatomic, strong) id<MTLBuffer> v_b2;
@property(nonatomic, strong) id<MTLBuffer> v_b3;
@property(nonatomic, strong) id<MTLBuffer> v_b4;
@property(nonatomic, strong) id<MTLBuffer> v_b5;
@property(nonatomic, strong) id<MTLBuffer> v_b6;
@property(nonatomic, strong) id<MTLBuffer> v_b7;
@property(nonatomic, strong) id<MTLBuffer> v_b8;

@property(nonatomic, strong) id<MTLBuffer> c1;
@property(nonatomic, strong) id<MTLBuffer> r1;
@property(nonatomic, strong) id<MTLBuffer> m1;
@property(nonatomic, strong) id<MTLBuffer> n1;

@property(nonatomic, strong) id<MTLBuffer> c2;
@property(nonatomic, strong) id<MTLBuffer> r2;
@property(nonatomic, strong) id<MTLBuffer> m2;
@property(nonatomic, strong) id<MTLBuffer> n2;

@property(nonatomic, strong) id<MTLBuffer> c3;
@property(nonatomic, strong) id<MTLBuffer> r3;

@property(nonatomic, strong) id<MTLBuffer> c4;
@property(nonatomic, strong) id<MTLBuffer> r4;

@property(nonatomic, strong) id<MTLBuffer> c5;
@property(nonatomic, strong) id<MTLBuffer> r5;
@property(nonatomic, strong) id<MTLBuffer> m5;
@property(nonatomic, strong) id<MTLBuffer> d5;

@property(nonatomic, strong) id<MTLBuffer> fc6;
@property(nonatomic, strong) id<MTLBuffer> r6;
@property(nonatomic, strong) id<MTLBuffer> d6;

@property(nonatomic, strong) id<MTLBuffer> fc7;
@property(nonatomic, strong) id<MTLBuffer> r7;

@property(nonatomic, strong) id<MTLBuffer> fc8;
@property(nonatomic, strong) id<MTLBuffer> r8;

@property(nonatomic, strong) id<MTLBuffer> dx;

@property(nonatomic, strong) id<MTLBuffer> dc1;
@property(nonatomic, strong) id<MTLBuffer> dr1;
@property(nonatomic, strong) id<MTLBuffer> dm1;
@property(nonatomic, strong) id<MTLBuffer> dn1;

@property(nonatomic, strong) id<MTLBuffer> dc2;
@property(nonatomic, strong) id<MTLBuffer> dr2;
@property(nonatomic, strong) id<MTLBuffer> dm2;
@property(nonatomic, strong) id<MTLBuffer> dn2;

@property(nonatomic, strong) id<MTLBuffer> dc3;
@property(nonatomic, strong) id<MTLBuffer> dr3;

@property(nonatomic, strong) id<MTLBuffer> dc4;
@property(nonatomic, strong) id<MTLBuffer> dr4;

@property(nonatomic, strong) id<MTLBuffer> dc5;
@property(nonatomic, strong) id<MTLBuffer> dr5;
@property(nonatomic, strong) id<MTLBuffer> dm5;
@property(nonatomic, strong) id<MTLBuffer> dd5;

@property(nonatomic, strong) id<MTLBuffer> dfc6;
@property(nonatomic, strong) id<MTLBuffer> dr6;
@property(nonatomic, strong) id<MTLBuffer> dd6;

@property(nonatomic, strong) id<MTLBuffer> dfc7;
@property(nonatomic, strong) id<MTLBuffer> dr7;

@property(nonatomic, strong) id<MTLBuffer> dfc8;
@property(nonatomic, strong) id<MTLBuffer> dr8;

@property(nonatomic, strong) id<MTLBuffer> dw1;
@property(nonatomic, strong) id<MTLBuffer> dw2;
@property(nonatomic, strong) id<MTLBuffer> dw3;
@property(nonatomic, strong) id<MTLBuffer> dw4;
@property(nonatomic, strong) id<MTLBuffer> dw5;
@property(nonatomic, strong) id<MTLBuffer> dw6;
@property(nonatomic, strong) id<MTLBuffer> dw7;
@property(nonatomic, strong) id<MTLBuffer> dw8;

@property(nonatomic, strong) id<MTLBuffer> db1;
@property(nonatomic, strong) id<MTLBuffer> db2;
@property(nonatomic, strong) id<MTLBuffer> db3;
@property(nonatomic, strong) id<MTLBuffer> db4;
@property(nonatomic, strong) id<MTLBuffer> db5;
@property(nonatomic, strong) id<MTLBuffer> db6;
@property(nonatomic, strong) id<MTLBuffer> db7;
@property(nonatomic, strong) id<MTLBuffer> db8;

@property(nonatomic, strong) id<MTLBuffer> m1_idxs;
@property(nonatomic, strong) id<MTLBuffer> m2_idxs;
@property(nonatomic, strong) id<MTLBuffer> m5_idxs;

@end

#endif /* Variables_h */
