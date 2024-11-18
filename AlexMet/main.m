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

    Generator* generator = [[Generator alloc] initWithDevice:device];
    Shapes* shapes = [[Shapes alloc] initWithGenerator:generator];
    Variables* vars = [[Variables alloc] initWithGenerator:generator :shapes :device];
    IO* io = [[IO alloc] initWithDevice:device];

    // MARK: - Load Data

    id<MTLBuffer> x = [device newBufferWithLength:n(shapes.x)*sizeof(float) options:MTLResourceStorageModeShared];
    int y[128];
    [io load_batch:n(shapes.x)*sizeof(float) :x :y];

    // MARK: - Training

    NN* nn = [[NN alloc] initWithDevice:device :commandQueue];

    for (int epoch = 0; epoch < 10; epoch++) {

        size_t epoch_timer = time(0);

        // MARK: - Forward Pass

        // 1
        [nn conv:x :shapes.xBuffer :vars.w1 :vars.b1 :shapes.c1_k :shapes.c1_p :shapes.c1_s :vars.c1 :shapes.c1 :shapes.c1Buffer];
        [nn relu:vars.c1 :vars.r1 :shapes.c1.N * shapes.c1.C :shapes.c1.H * shapes.c1.W];
        [nn max_pool:vars.r1 :shapes.c1Buffer :vars.m1 :shapes.m1Buffer :shapes.m1 :vars.m1_idxs];
        [nn lrn:vars.m1 :shapes.m1Buffer :vars.n1 :shapes.m1];

        // 2
        [nn conv:vars.n1 :shapes.m1Buffer :vars.w2 :vars.b2 :shapes.c2_k :shapes.c2_p :shapes.c2_s :vars.c2 :shapes.c2 :shapes.c2Buffer];
        [nn relu:vars.c2 :vars.r2 :shapes.c2.N * shapes.c2.C :shapes.c2.H * shapes.c2.W];
        [nn max_pool:vars.r2 :shapes.c2Buffer :vars.m2 :shapes.m2Buffer :shapes.m2 :vars.m2_idxs];
        [nn lrn:vars.m2 :shapes.m2Buffer :vars.n2 :shapes.m2];

        // 3
        [nn conv:vars.n2 :shapes.m2Buffer :vars.w3 :vars.b3 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.c3 :shapes.c3 :shapes.c3Buffer];
        [nn relu:vars.c3 :vars.r3 :shapes.c3.N * shapes.c3.C :shapes.c3.H * shapes.c3.W];

        // 4
        [nn conv:vars.r3 :shapes.c3Buffer :vars.w4 :vars.b4 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.c4 :shapes.c4 :shapes.c4Buffer];
        [nn relu:vars.c4 :vars.r4 :shapes.c4.N * shapes.c4.C :shapes.c4.H * shapes.c4.W];

        // 5
        [nn conv:vars.r4 :shapes.c4Buffer :vars.w5 :vars.b5 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.c5 :shapes.c5 :shapes.c5Buffer];
        [nn relu:vars.c5 :vars.r5 :shapes.c5.N * shapes.c5.C :shapes.c5.H * shapes.c5.W];
        [nn max_pool:vars.r5 :shapes.c5Buffer :vars.m5 :shapes.m5Buffer :shapes.m5 :vars.m5_idxs];
        [nn dropout :vars.m5 :vars.d5 :shapes.m5.N * shapes.m5.C :shapes.m5.H * shapes.m5.W];

        // 6
        [nn matmul:vars.d5 :vars.w6 :shapes.bs :shapes.m5.C * shapes.m5.H * shapes.m5.W :4096 :vars.fc6];
        [nn bias:vars.fc6 :vars.b6 :shapes.bs :4096];
        [nn relu:vars.fc6 :vars.r6 :shapes.bs :4096];
        [nn dropout:vars.r6 :vars.d6 :shapes.bs :4096];

        // 7
        [nn matmul:vars.d6 :vars.w7 :shapes.bs :4096 :4096 :vars.fc7];
        [nn bias:vars.fc7 :vars.b7 :shapes.bs :4096];
        [nn relu:vars.fc7 :vars.r7 :shapes.bs :4096];

        // 8
        [nn matmul:vars.r7 :vars.w8 :shapes.bs :4096 :1000 :vars.fc8];
        [nn bias:vars.fc8 :vars.b8 :shapes.bs :1000];
        [nn relu:vars.fc8 :vars.r8 :shapes.bs :1000];

        // 9

        float loss = [nn cross_entropy:vars.r8.contents :y :shapes.bs :shapes.classes];

        printf("Metal loss is %f\n", loss);
        printf("Forward pass took %lu s\n", time(0) - epoch_timer);

        // MARK: - Backward Pass

        epoch_timer = time(0);
        size_t bw_timer = time(0);

        // 9
        [nn cross_entropy_bw:vars.r8.contents :y :shapes.bs :shapes.classes :vars.dr8.contents];

        printf("9 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 8
        [nn relu_bw:vars.fc8 :shapes.bs :1000 :vars.dr8 :vars.dfc8];
        [nn bias_bw:vars.dfc8 :shapes.bs :1000 :vars.db8];
        [nn matmul:T(vars.r7.contents, shapes.bs, 4096, device) :vars.dfc8 :4096 :shapes.bs :1000 :vars.dw8];
        [nn matmul:vars.dfc8 :T(vars.w8.contents, 4096, 1000, device) :shapes.bs :1000 :4096 :vars.dr7];

        printf("8 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 7
        [nn relu_bw:vars.fc7 :shapes.bs :4096 :vars.dr7 :vars.dfc7];
        [nn bias_bw:vars.dfc7 :shapes.bs :4096 :vars.db7];
        [nn matmul:T(vars.d6.contents, shapes.bs, 4096, device) :vars.dfc7 :4096 :shapes.bs :4096 :vars.dw7];
        [nn matmul:vars.dfc7 :T(vars.w7.contents, 4096, 4096, device) :shapes.bs :4096 :4096 :vars.dd6];

        printf("7 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 6
        [nn dropout_bw:vars.d6 :shapes.bs :4096 :vars.dd6 :vars.dr6];
        [nn relu_bw:vars.fc6 :shapes.bs :4096 :vars.dr6 :vars.dfc6];
        [nn bias_bw:vars.dfc6 :shapes.bs :4096 :vars.db6];
        [nn matmul:T(vars.d5.contents, shapes.bs, shapes.m5.C * shapes.m5.H * shapes.m5.W, device) :vars.dfc6 :shapes.m5.C * shapes.m5.H * shapes.m5.W :shapes.bs :4096 :vars.dw6];
        [nn matmul:vars.dfc6 :T(vars.w6.contents, shapes.m5.C * shapes.m5.H * shapes.m5.W, 4096, device) :shapes.bs :4096 :shapes.m5.C * shapes.m5.H * shapes.m5.W :vars.dd5];

        printf("6 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 5
        [nn dropout_bw:vars.d5 :shapes.m5.N * shapes.m5.C: shapes.m5.H * shapes.m5.W :vars.dd5 :vars.dm5];
        [nn max_pool_bw:vars.dm5 :shapes.m5Buffer :shapes.m5 :vars.m5_idxs :vars.dr5];
        [nn relu_bw:vars.c5 :shapes.c5.N * shapes.c5.C :shapes.c5.H * shapes.c5.W :vars.dr5 :vars.dc5];
        [nn conv_bw:vars.r4 :shapes.c4Buffer :vars.w5 :vars.b5 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.dc5 :shapes.c5 :shapes.c5Buffer :vars.dr4 :vars.dw5 :vars.db5];

        printf("5 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 4
        [nn relu_bw:vars.c4 :shapes.c4.N * shapes.c4.C :shapes.c4.H * shapes.c4.W :vars.dr4 :vars.dc4];
        [nn conv_bw:vars.r3 :shapes.c3Buffer :vars.w4 :vars.b4 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.dc4 :shapes.c4 :shapes.c4Buffer :vars.dr3 :vars.dw4 :vars.db4];

        printf("4 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 3
        [nn relu_bw:vars.c3 :shapes.c3.N * shapes.c3.C :shapes.c3.H * shapes.c3.W :vars.dr3 :vars.dc3];
        [nn conv_bw:vars.n2 :shapes.m2Buffer :vars.w3 :vars.b3 :shapes.c345_k :shapes.c345_p :shapes.c345_s :vars.dc3 :shapes.c3 :shapes.c3Buffer :vars.dn2 :vars.dw3 :vars.db3];

        printf("3 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 2
        [nn lrn_bw:vars.m2 :shapes.m2Buffer :vars.dn2 :shapes.m2 :vars.dm2];
        [nn max_pool_bw:vars.dm2 :shapes.m2Buffer :shapes.m2 :vars.m2_idxs :vars.dr2];
        [nn relu_bw:vars.c2 :shapes.c2.N * shapes.c2.C :shapes.c2.H * shapes.c2.W :vars.dr2 :vars.dc2];
        [nn conv_bw:vars.n1 :shapes.m1Buffer :vars.w2 :vars.b2 :shapes.c2_k :shapes.c2_p :shapes.c2_s :vars.dc2 :shapes.c2 :shapes.c2Buffer :vars.dn1 :vars.dw2 :vars.db2];

        printf("2 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        // 1
        [nn lrn_bw:vars.m1 :shapes.m1Buffer :vars.dn1 :shapes.m1 :vars.dm1];
        [nn max_pool_bw:vars.dm1 :shapes.m1Buffer :shapes.m1 :vars.m1_idxs :vars.dr1];
        [nn relu_bw:vars.c1 :shapes.c1.N * shapes.c1.C :shapes.c1.H * shapes.c1.W :vars.dr1 :vars.dc1];
        [nn conv_bw:x :shapes.xBuffer :vars.w1 :vars.b1 :shapes.c1_k :shapes.c1_p :shapes.c1_s :vars.dc1 :shapes.c1 :shapes.c1Buffer :vars.dx :vars.dw1 :vars.db1];

        printf("1 took %lu s\n", time(0) - bw_timer);
        bw_timer = time(0);

        printf("Backward pass took %lu s\n", time(0) - epoch_timer);
        epoch_timer = time(0);

        // MARK: - Gradient Descent

        id<MTLCommandBuffer> sgdBuffer = [commandQueue commandBuffer];

        [nn sgd:vars.w1 :vars.dw1 :vars.v_w1 :shapes.w1.N * shapes.w1.C :shapes.w1.H * shapes.w1.W :shapes.lr :sgdBuffer];
        [nn sgd:vars.w2 :vars.dw2 :vars.v_w2 :shapes.w2.N * shapes.w2.C :shapes.w2.H * shapes.w2.W :shapes.lr :sgdBuffer];
        [nn sgd:vars.w3 :vars.dw3 :vars.v_w3 :shapes.w3.N * shapes.w3.C :shapes.w3.H * shapes.w3.W :shapes.lr :sgdBuffer];
        [nn sgd:vars.w4 :vars.dw4 :vars.v_w4 :shapes.w4.N * shapes.w4.C :shapes.w4.H * shapes.w4.W :shapes.lr :sgdBuffer];
        [nn sgd:vars.w5 :vars.dw5 :vars.v_w5 :shapes.w5.N * shapes.w5.C :shapes.w5.H * shapes.w5.W :shapes.lr :sgdBuffer];
        [nn sgd:vars.w6 :vars.dw6 :vars.v_w6 :shapes.m5.C * shapes.m5.H * shapes.m5.W :4096 :shapes.lr :sgdBuffer];
        [nn sgd:vars.w7 :vars.dw7 :vars.v_w7 :4096 :4096 :shapes.lr :sgdBuffer];
        [nn sgd:vars.w8 :vars.dw8 :vars.v_w8 :4096 :1000 :shapes.lr :sgdBuffer];

        [nn sgd:vars.b1 :vars.db1 :vars.v_b1 :12 :8 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b2 :vars.db2 :vars.v_b2 :16 :16 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b3 :vars.db3 :vars.v_b3 :24 :16 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b4 :vars.db4 :vars.v_b4 :24 :16 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b5 :vars.db5 :vars.v_b5 :16 :16 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b6 :vars.db6 :vars.v_b6 :64 :64 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b7 :vars.db7 :vars.v_b7 :64 :64 :shapes.lr :sgdBuffer];
        [nn sgd:vars.b8 :vars.db8 :vars.v_b8 :10 :100 :shapes.lr :sgdBuffer];

        [sgdBuffer commit];
        [sgdBuffer waitUntilCompleted];

        printf("SGD took %lu s\n", time(0) - epoch_timer);
    }
    return 0;
}
