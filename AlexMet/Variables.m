//
//  Variables.m
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 14.11.2024.
//

#import <Foundation/Foundation.h>
#import "Variables.h"

@implementation Variables

- (instancetype)initWithGenerator:(Generator *)generator :(Shapes*)shapes :(id<MTLDevice>)device {
    self = [super init];

    _w1 = [generator gaussian_buffer:n(shapes.w1) :@"w1"];
    _w2 = [generator gaussian_buffer:n(shapes.w2) :@"w2"];
    _w3 = [generator gaussian_buffer:n(shapes.w3) :@"w3"];
    _w4 = [generator gaussian_buffer:n(shapes.w4) :@"w4"];
    _w5 = [generator gaussian_buffer:n(shapes.w5) :@"w5"];
    _w6 = [generator gaussian_buffer:shapes.w6    :@"w6"];
    _w7 = [generator gaussian_buffer:shapes.w7    :@"w7"];
    _w8 = [generator gaussian_buffer:shapes.w8    :@"w8"];

    _v_w1 = [generator zero_buffer:n(shapes.w1) :@"v_w1"];
    _v_w2 = [generator zero_buffer:n(shapes.w2) :@"v_w2"];
    _v_w3 = [generator zero_buffer:n(shapes.w3) :@"v_w3"];
    _v_w4 = [generator zero_buffer:n(shapes.w4) :@"v_w4"];
    _v_w5 = [generator zero_buffer:n(shapes.w5) :@"v_w5"];
    _v_w6 = [generator zero_buffer:shapes.w6    :@"v_w5"];
    _v_w7 = [generator zero_buffer:shapes.w7    :@"v_w6"];
    _v_w8 = [generator zero_buffer:shapes.w8    :@"v_w7"];

    _b1 = [generator zero_buffer:shapes.c1.C :@"b1"];
    _b2 = [generator ones_buffer:shapes.c2.C :@"b2"];
    _b3 = [generator zero_buffer:shapes.c3.C :@"b3"];
    _b4 = [generator ones_buffer:shapes.c4.C :@"b4"];
    _b5 = [generator ones_buffer:shapes.c5.C :@"b5"];
    _b6 = [generator ones_buffer:shapes.b6   :@"b6"];
    _b7 = [generator ones_buffer:shapes.b7   :@"b7"];
    _b8 = [generator ones_buffer:shapes.b8   :@"b8"];

    _v_b1 = [generator zero_buffer:shapes.c1.C :@"v_b1"];
    _v_b2 = [generator zero_buffer:shapes.c2.C :@"v_b2"];
    _v_b3 = [generator zero_buffer:shapes.c3.C :@"v_b3"];
    _v_b4 = [generator zero_buffer:shapes.c4.C :@"v_b4"];
    _v_b5 = [generator zero_buffer:shapes.c5.C :@"v_b5"];
    _v_b6 = [generator zero_buffer:shapes.b6   :@"v_b6"];
    _v_b7 = [generator zero_buffer:shapes.b7   :@"v_b7"];
    _v_b8 = [generator zero_buffer:shapes.b8   :@"v_b8"];

    _c1 = [generator zero_buffer:n(shapes.c1) :@"c1"];
    _r1 = [generator buffer:n(shapes.c1)      :@"r1"];
    _m1 = [generator buffer:n(shapes.m1)      :@"m1"];
    _n1 = [generator buffer:n(shapes.m1)      :@"n1"];

    _c2 = [generator zero_buffer:n(shapes.c2) :@"c2"];
    _r2 = [generator buffer:n(shapes.c2)      :@"r2"];
    _m2 = [generator buffer:n(shapes.m2)      :@"m2"];
    _n2 = [generator buffer:n(shapes.m2)      :@"n2"];

    _c3 = [generator zero_buffer:n(shapes.c3) :@"c3"];
    _r3 = [generator buffer:n(shapes.c3)      :@"r3"];

    _c4 = [generator zero_buffer:n(shapes.c4) :@"c4"];
    _r4 = [generator buffer:n(shapes.c4)      :@"r4"];

    _c5 = [generator zero_buffer:n(shapes.c5) :@"c5"];
    _r5 = [generator buffer:n(shapes.c5)      :@"r5"];
    _m5 = [generator buffer:n(shapes.m5)      :@"m5"];
    _d5 = [generator buffer:n(shapes.m5)      :@"d5"];

    _fc6 = [generator buffer:shapes.fc6 :@"fc6"];
    _r6 = [generator buffer:shapes.fc6  :@"r6"];
    _d6 = [generator buffer:shapes.fc6  :@"d6"];

    _fc7 = [generator buffer:shapes.fc7 :@"fc7"];
    _r7 = [generator buffer:shapes.fc7  :@"r7"];

    _fc8 = [generator buffer:shapes.fc8 :@"fc8"];
    _r8 = [generator buffer:shapes.fc8  :@"r8"];

    _dx  = [generator zero_buffer:n(shapes.x)  :@"dx"];

    _dc1  = [generator buffer:n(shapes.c1)     :@"dc1"];
    _dr1 = [generator zero_buffer:n(shapes.c1) :@"dr1"];
    _dm1 = [generator buffer:n(shapes.m1)      :@"dm1"];
    _dn1 = [generator zero_buffer:n(shapes.m1) :@"dn1"];

    _dc2  = [generator buffer:n(shapes.c2)     :@"dc2"];
    _dr2 = [generator zero_buffer:n(shapes.c2) :@"dr2"];
    _dm2 = [generator buffer:n(shapes.m2)      :@"dm2"];
    _dn2 = [generator zero_buffer:n(shapes.m2) :@"dn2"];

    _dc3  = [generator buffer:n(shapes.c3)     :@"dc3"];
    _dr3 = [generator zero_buffer:n(shapes.c3) :@"dr3"];

    _dc4  = [generator buffer:n(shapes.c4)     :@"dc4"];
    _dr4 = [generator zero_buffer:n(shapes.c4) :@"dr4"];

    _dc5  = [generator buffer:n(shapes.c5)     :@"dc5"];
    _dr5 = [generator zero_buffer:n(shapes.c5) :@"dr5"];
    _dm5 = [generator buffer:n(shapes.m5)      :@"dm5"];
    _dd5 = [generator buffer:n(shapes.m5)      :@"dd5"];

    _dfc6 = [generator buffer:shapes.fc6 :@"dfc6"];
    _dr6 = [generator buffer:shapes.fc6  :@"dr6"];
    _dd6 = [generator buffer:shapes.fc6  :@"dd6"];

    _dfc7 = [generator buffer:shapes.fc7 :@"dfc7"];
    _dr7 = [generator buffer:shapes.fc7  :@"dr7"];

    _dfc8 = [generator buffer:shapes.fc8 :@"dfc8"];
    _dr8 = [generator buffer:shapes.fc8  :@"dr8"];

    _dw1 = [generator zero_buffer:n(shapes.w1) :@"dw1"];
    _dw2 = [generator zero_buffer:n(shapes.w2) :@"dw2"];
    _dw3 = [generator zero_buffer:n(shapes.w3) :@"dw3"];
    _dw4 = [generator zero_buffer:n(shapes.w4) :@"dw4"];
    _dw5 = [generator zero_buffer:n(shapes.w5) :@"dw5"];
    _dw6 = [generator buffer:shapes.w6         :@"dw6"];
    _dw7 = [generator buffer:shapes.w7         :@"dw7"];
    _dw8 = [generator buffer:shapes.w8         :@"dw8"];

    _db1 = [generator zero_buffer:shapes.c1.C :@"db1"];
    _db2 = [generator zero_buffer:shapes.c2.C :@"db2"];
    _db3 = [generator zero_buffer:shapes.c3.C :@"db3"];
    _db4 = [generator zero_buffer:shapes.c4.C :@"db4"];
    _db5 = [generator zero_buffer:shapes.c5.C :@"db5"];
    _db6 = [generator buffer:shapes.b6        :@"db6"];
    _db7 = [generator buffer:shapes.b7        :@"db7"];
    _db8 = [generator zero_buffer:shapes.b8   :@"db8"];

    _m1_idxs = [device newBufferWithLength:n(shapes.m1) * sizeof(uint) options:MTLResourceStorageModeShared];
    _m2_idxs = [device newBufferWithLength:n(shapes.m2) * sizeof(uint) options:MTLResourceStorageModeShared];
    _m5_idxs = [device newBufferWithLength:n(shapes.m5) * sizeof(uint) options:MTLResourceStorageModeShared];

    return self;
}

@end
