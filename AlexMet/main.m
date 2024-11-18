//
//  main.m
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 01.11.2024.
//

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "Generator.h"
#import "Shapes.h"
#import "Variables.h"
#import "NN.h"
#import "IO.h"

id<MTLBuffer> T(float* a, int R, int C, id<MTLDevice> device) {
    float* aT = malloc(R * C * sizeof(float));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            aT[j * R + i] = a[i * C + j];
    id<MTLBuffer> buffer = [device newBufferWithBytes:aT length:sizeof(float) * R * C options:MTLResourceStorageModeShared];
    free(aT);
    return buffer;
}

int main(void) {
    srandom((unsigned int)time(NULL));

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    Generator* gen = [[Generator alloc] initWithDevice:device];
    Shapes* s = [[Shapes alloc] initWithGenerator:gen];
    Variables* v = [[Variables alloc] initWithGenerator:gen :s :device];
    IO* io = [[IO alloc] initWithDevice:device];
    NN* nn = [[NN alloc] initWithDevice:device :commandQueue];

    // MARK: - Load Data

    id<MTLBuffer> x = [device newBufferWithLength:n(s.x)*sizeof(float) options:MTLResourceStorageModeShared];
    int y[128];
    [io load_batch:n(s.x)*sizeof(float) :x :y];

    // MARK: - Training

    for (int epoch = 0; epoch < 10; epoch++) {

        size_t epoch_timer = time(0);

        // MARK: - Forward Pass

        // 1
        [nn conv:x :s.xBuffer :v.w1 :v.b1 :s.c1_k :s.c1_p :s.c1_s :v.c1 :s.c1 :s.c1Buffer];
        [nn relu:v.c1 :v.r1 :s.c1.N * s.c1.C :s.c1.H * s.c1.W];
        [nn max_pool:v.r1 :s.c1Buffer :v.m1 :s.m1Buffer :s.m1 :v.m1_idxs];
        [nn lrn:v.m1 :s.m1Buffer :v.n1 :s.m1];

        // 2
        [nn conv:v.n1 :s.m1Buffer :v.w2 :v.b2 :s.c2_k :s.c2_p :s.c2_s :v.c2 :s.c2 :s.c2Buffer];
        [nn relu:v.c2 :v.r2 :s.c2.N * s.c2.C :s.c2.H * s.c2.W];
        [nn max_pool:v.r2 :s.c2Buffer :v.m2 :s.m2Buffer :s.m2 :v.m2_idxs];
        [nn lrn:v.m2 :s.m2Buffer :v.n2 :s.m2];

        // 3
        [nn conv:v.n2 :s.m2Buffer :v.w3 :v.b3 :s.c345_k :s.c345_p :s.c345_s :v.c3 :s.c3 :s.c3Buffer];
        [nn relu:v.c3 :v.r3 :s.c3.N * s.c3.C :s.c3.H * s.c3.W];

        // 4
        [nn conv:v.r3 :s.c3Buffer :v.w4 :v.b4 :s.c345_k :s.c345_p :s.c345_s :v.c4 :s.c4 :s.c4Buffer];
        [nn relu:v.c4 :v.r4 :s.c4.N * s.c4.C :s.c4.H * s.c4.W];

        // 5
        [nn conv:v.r4 :s.c4Buffer :v.w5 :v.b5 :s.c345_k :s.c345_p :s.c345_s :v.c5 :s.c5 :s.c5Buffer];
        [nn relu:v.c5 :v.r5 :s.c5.N * s.c5.C :s.c5.H * s.c5.W];
        [nn max_pool:v.r5 :s.c5Buffer :v.m5 :s.m5Buffer :s.m5 :v.m5_idxs];
        [nn dropout :v.m5 :v.d5 :s.m5.N * s.m5.C :s.m5.H * s.m5.W];

        // 6
        [nn matmul:v.d5 :v.w6 :s.bs :s.m5.C * s.m5.H * s.m5.W :4096 :v.fc6];
        [nn bias:v.fc6 :v.b6 :s.bs :4096];
        [nn relu:v.fc6 :v.r6 :s.bs :4096];
        [nn dropout:v.r6 :v.d6 :s.bs :4096];

        // 7
        [nn matmul:v.d6 :v.w7 :s.bs :4096 :4096 :v.fc7];
        [nn bias:v.fc7 :v.b7 :s.bs :4096];
        [nn relu:v.fc7 :v.r7 :s.bs :4096];

        // 8
        [nn matmul:v.r7 :v.w8 :s.bs :4096 :1000 :v.fc8];
        [nn bias:v.fc8 :v.b8 :s.bs :1000];
        [nn relu:v.fc8 :v.r8 :s.bs :1000];

        // 9

        float loss = [nn cross_entropy:v.r8.contents :y :s.bs];

        printf("Loss is %f\n", loss);
        printf("Forward pass took %lu s\n", time(0) - epoch_timer);

        // MARK: - Backward Pass

        epoch_timer = time(0);
        size_t bw_timer = time(0);

        // 9
        [nn cross_entropy_bw:v.r8.contents :y :s.bs :v.dr8.contents];

        printf("9 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 8
        [nn relu_bw:v.fc8 :s.bs :1000 :v.dr8 :v.dfc8];
        [nn bias_bw:v.dfc8 :s.bs :1000 :v.db8];
        [nn matmul:T(v.r7.contents, s.bs, 4096, device) :v.dfc8 :4096 :s.bs :1000 :v.dw8];
        [nn matmul:v.dfc8 :T(v.w8.contents, 4096, 1000, device) :s.bs :1000 :4096 :v.dr7];

        printf("8 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 7
        [nn relu_bw:v.fc7 :s.bs :4096 :v.dr7 :v.dfc7];
        [nn bias_bw:v.dfc7 :s.bs :4096 :v.db7];
        [nn matmul:T(v.d6.contents, s.bs, 4096, device) :v.dfc7 :4096 :s.bs :4096 :v.dw7];
        [nn matmul:v.dfc7 :T(v.w7.contents, 4096, 4096, device) :s.bs :4096 :4096 :v.dd6];

        printf("7 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 6
        [nn dropout_bw:v.d6 :s.bs :4096 :v.dd6 :v.dr6];
        [nn relu_bw:v.fc6 :s.bs :4096 :v.dr6 :v.dfc6];
        [nn bias_bw:v.dfc6 :s.bs :4096 :v.db6];
        [nn matmul:T(v.d5.contents, s.bs, s.m5.C * s.m5.H * s.m5.W, device) :v.dfc6 :s.m5.C * s.m5.H * s.m5.W :s.bs :4096 :v.dw6];
        [nn matmul:v.dfc6 :T(v.w6.contents, s.m5.C * s.m5.H * s.m5.W, 4096, device) :s.bs :4096 :s.m5.C * s.m5.H * s.m5.W :v.dd5];

        printf("6 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 5
        [nn dropout_bw:v.d5 :s.m5.N * s.m5.C: s.m5.H * s.m5.W :v.dd5 :v.dm5];
        [nn max_pool_bw:v.dm5 :s.m5Buffer :s.m5 :v.m5_idxs :v.dr5];
        [nn relu_bw:v.c5 :s.c5.N * s.c5.C :s.c5.H * s.c5.W :v.dr5 :v.dc5];
        [nn conv_bw:v.r4 :s.c4Buffer :v.w5 :v.b5 :s.c345_k :s.c345_p :s.c345_s :v.dc5 :s.c5 :s.c5Buffer :v.dr4 :v.dw5 :v.db5];

        printf("5 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 4
        [nn relu_bw:v.c4 :s.c4.N * s.c4.C :s.c4.H * s.c4.W :v.dr4 :v.dc4];
        [nn conv_bw:v.r3 :s.c3Buffer :v.w4 :v.b4 :s.c345_k :s.c345_p :s.c345_s :v.dc4 :s.c4 :s.c4Buffer :v.dr3 :v.dw4 :v.db4];

        printf("4 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 3
        [nn relu_bw:v.c3 :s.c3.N * s.c3.C :s.c3.H * s.c3.W :v.dr3 :v.dc3];
        [nn conv_bw:v.n2 :s.m2Buffer :v.w3 :v.b3 :s.c345_k :s.c345_p :s.c345_s :v.dc3 :s.c3 :s.c3Buffer :v.dn2 :v.dw3 :v.db3];

        printf("3 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 2
        [nn lrn_bw:v.m2 :s.m2Buffer :v.dn2 :s.m2 :v.dm2];
        [nn max_pool_bw:v.dm2 :s.m2Buffer :s.m2 :v.m2_idxs :v.dr2];
        [nn relu_bw:v.c2 :s.c2.N * s.c2.C :s.c2.H * s.c2.W :v.dr2 :v.dc2];
        [nn conv_bw:v.n1 :s.m1Buffer :v.w2 :v.b2 :s.c2_k :s.c2_p :s.c2_s :v.dc2 :s.c2 :s.c2Buffer :v.dn1 :v.dw2 :v.db2];

        printf("2 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 1
        [nn lrn_bw:v.m1 :s.m1Buffer :v.dn1 :s.m1 :v.dm1];
        [nn max_pool_bw:v.dm1 :s.m1Buffer :s.m1 :v.m1_idxs :v.dr1];
        [nn relu_bw:v.c1 :s.c1.N * s.c1.C :s.c1.H * s.c1.W :v.dr1 :v.dc1];
        [nn conv_bw:x :s.xBuffer :v.w1 :v.b1 :s.c1_k :s.c1_p :s.c1_s :v.dc1 :s.c1 :s.c1Buffer :v.dx :v.dw1 :v.db1];

        printf("1 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        printf("Backward pass took %lu s\n", time(0) - epoch_timer);
        epoch_timer = time(0);

        // MARK: - Gradient Descent

        id<MTLCommandBuffer> sgdBuffer = [commandQueue commandBuffer];

        [nn sgd:v.w1 :v.dw1 :v.v_w1 :s.w1.N * s.w1.C :s.w1.H * s.w1.W :s.lr :sgdBuffer];
        [nn sgd:v.w2 :v.dw2 :v.v_w2 :s.w2.N * s.w2.C :s.w2.H * s.w2.W :s.lr :sgdBuffer];
        [nn sgd:v.w3 :v.dw3 :v.v_w3 :s.w3.N * s.w3.C :s.w3.H * s.w3.W :s.lr :sgdBuffer];
        [nn sgd:v.w4 :v.dw4 :v.v_w4 :s.w4.N * s.w4.C :s.w4.H * s.w4.W :s.lr :sgdBuffer];
        [nn sgd:v.w5 :v.dw5 :v.v_w5 :s.w5.N * s.w5.C :s.w5.H * s.w5.W :s.lr :sgdBuffer];
        [nn sgd:v.w6 :v.dw6 :v.v_w6 :s.m5.C * s.m5.H * s.m5.W :4096 :s.lr :sgdBuffer];
        [nn sgd:v.w7 :v.dw7 :v.v_w7 :4096 :4096 :s.lr :sgdBuffer];
        [nn sgd:v.w8 :v.dw8 :v.v_w8 :4096 :1000 :s.lr :sgdBuffer];

        [nn sgd:v.b1 :v.db1 :v.v_b1 :12 :8 :s.lr :sgdBuffer];
        [nn sgd:v.b2 :v.db2 :v.v_b2 :16 :16 :s.lr :sgdBuffer];
        [nn sgd:v.b3 :v.db3 :v.v_b3 :24 :16 :s.lr :sgdBuffer];
        [nn sgd:v.b4 :v.db4 :v.v_b4 :24 :16 :s.lr :sgdBuffer];
        [nn sgd:v.b5 :v.db5 :v.v_b5 :16 :16 :s.lr :sgdBuffer];
        [nn sgd:v.b6 :v.db6 :v.v_b6 :64 :64 :s.lr :sgdBuffer];
        [nn sgd:v.b7 :v.db7 :v.v_b7 :64 :64 :s.lr :sgdBuffer];
        [nn sgd:v.b8 :v.db8 :v.v_b8 :10 :100 :s.lr :sgdBuffer];

        [sgdBuffer commit];
        [sgdBuffer waitUntilCompleted];

        printf("SGD took %lu s\n", time(0) - epoch_timer);
    }
    return 0;
}
