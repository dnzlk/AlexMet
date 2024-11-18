//
//  Shapes.m
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 14.11.2024.
//

#import <Foundation/Foundation.h>
#import "Shapes.h"
#import "Generator.h"

int n(Shape shape) {
    return shape.N * shape.C * shape.H * shape.W;
}

@implementation Shapes

- (instancetype)initWithGenerator:(Generator*)generator {
    self = [super init];

    _bs = 1;
    _classes = 1000;

    _lr = 0.0001;

    _c1_k = 11; _c1_p = 3; _c1_s = 4;
    _c2_k = 5; _c2_p = 2; _c2_s = 1;
    _c345_k = 3; _c345_p = 1; _c345_s = 1;

    _x = [self create:_bs :3 :227 :227];

    uint c1H = [self s_conv:227 :_c1_k :_c1_p :_c1_s];
    uint c1W = [self s_conv:227 :_c1_k :_c1_p :_c1_s];
    uint m1H = [self s_max_pool:c1H];
    uint m1W = [self s_max_pool:c1W];
    _c1 = [self create:_bs :96 :c1H :c1W];
    _w1 = [self create:3 :96 :_c1_k :_c1_k];
    _m1 = [self create:_bs :96 :m1H :m1W];

    uint c2H = [self s_conv:m1H :_c2_k :_c2_p :_c2_s];
    uint c2W = [self s_conv:m1W :_c2_k :_c2_p :_c2_s];
    uint m2H = [self s_max_pool:c2H];
    uint m2W = [self s_max_pool:c2W];
    _c2 = [self create:_bs :256 :c2H :c2W];
    _w2 = [self create:96 :256 :_c2_k :_c2_k];
    _m2 = [self create:_bs :256 :m2H :m2W];

    uint c3H = [self s_conv:m2H :_c345_k :_c345_p :_c345_s];
    uint c3W = [self s_conv:m2W :_c345_k :_c345_p :_c345_s];
    _c3 = [self create:_bs :384 :c3H :c3W];
    _w3 = [self create:256 :384 :_c345_k :_c345_k];
    _c4 = [self create:_bs :384 :c3H :c3W];
    _w4 = [self create:384 :384 :_c345_k :_c345_k];

    uint c5H = [self s_conv:c3H :_c345_k :_c345_p :_c345_s];
    uint c5W = [self s_conv:c3W :_c345_k :_c345_p :_c345_s];
    uint m5H = [self s_max_pool:c5H];
    uint m5W = [self s_max_pool:c5W];
    _c5 = [self create:_bs :256 :c5H :c5W];
    _w5 = [self create:384 :256 :_c345_k :_c345_k];
    _m5 = [self create:_bs :256 :m5H :m5W];

    _fc6 = _bs * 4096;
    _w6 = _m5.C * _m5.H * _m5.W * 4096;
    _b6 = 4096;

    _fc7 = _bs * 4096;
    _w7 = 4096 * 4096;
    _b7 = 4096;

    _fc8 = _bs * _classes;
    _w8 = 4096 * 1000;
    _b8 = 1000;

    _xBuffer =  [generator shape_buffer:_x   :@"xshBuffer"];
    _c1Buffer = [generator shape_buffer:_c1 :@"c1shBuffer"];
    _m1Buffer = [generator shape_buffer:_m1 :@"m1shBuffer"];
    _c2Buffer = [generator shape_buffer:_c2 :@"c2shBuffer"];
    _m2Buffer = [generator shape_buffer:_m2 :@"m2shBuffer"];
    _c3Buffer = [generator shape_buffer:_c3 :@"c3shBuffer"];
    _c4Buffer = [generator shape_buffer:_c4 :@"c4shBuffer"];
    _c5Buffer = [generator shape_buffer:_c5 :@"c5shBuffer"];
    _m5Buffer = [generator shape_buffer:_m5 :@"m5shBuffer"];

    return self;
}

- (Shape)create :(uint)N :(uint)C :(uint)H :(uint)W {
    return (Shape){N, C, H, W};
}

- (uint)s_conv :(uint)side :(uint)k :(uint)p :(uint)s {
    return (side - k + 2*p)/s + 1;
}

- (uint)s_max_pool :(uint)side {
    uint k = 3;
    uint s = 2;
    return ((side - k) / s) + 1;
}

@end
