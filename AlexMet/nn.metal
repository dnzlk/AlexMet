//
//  add.metal
//  AlexNetC+Metal
//
//  Created by Denis Khabarov on 01.11.2024.
//

#include <metal_stdlib>

using namespace metal;

struct Shape {
    uint N, C, H, W;
};

// Generate a random float in the range [0.0f, 1.0f] using x, y, and z (based on the xor128 algorithm)
float rand(uint x, uint y, uint z) {
    uint seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}

uint index(Shape shape, uint n, uint c, uint h, uint w) {
    return n * shape.C * shape.H * shape.W + c * shape.H * shape.W + h * shape.W + w;
}

kernel void conv(device const float* in [[buffer(0)]],
                 constant Shape& insh   [[buffer(1)]],
                 device const float* W  [[buffer(2)]],
                 device const float* b  [[buffer(3)]],
                 constant uint& k       [[buffer(4)]],
                 constant uint& p       [[buffer(5)]],
                 constant uint& s       [[buffer(6)]],
                 device float* out      [[buffer(7)]],
                 constant Shape& osh    [[buffer(8)]],
                 uint3 _thread          [[thread_position_in_grid]]) {
    if (_thread.y >= osh.H || _thread.x >= osh.W || _thread.z >= osh.C) return;

    uint x = _thread.x;
    uint y = _thread.y;
    uint oc = _thread.z;

    int in_y = y * s - p;
    int in_x = x * s - p;

    for (uint n = 0; n < osh.N; n++) {
        float sum = 0;
        for (uint c = 0; c < insh.C; c++) {
            for (uint i = 0; i < k; i++) {
                if (in_y + (int)i < 0 || in_y + (int)i >= (int)insh.H) // padding
                    continue;
                for (uint j = 0; j < k; j++) {
                    if (in_x + (int)j < 0 || in_x + (int)j >= (int)insh.W) // padding
                        continue;
                    sum += W[c * osh.C * k * k + oc * k * k + i * k + j] * in[index(insh, n, c, in_y + i, in_x + j)];
                }
            }
        }
        out[index(osh, n, oc, y, x)] = sum + b[oc];
    }
}

kernel void conv_bw(device const float* in   [[buffer(0)]],
                    constant Shape& insh     [[buffer(1)]],
                    device const float* W    [[buffer(2)]],
                    device const float* b    [[buffer(3)]],
                    constant uint& k         [[buffer(4)]],
                    constant uint& p         [[buffer(5)]],
                    constant uint& s         [[buffer(6)]],
                    device const float* dout [[buffer(7)]],
                    constant Shape& osh      [[buffer(8)]],
                    device float* din        [[buffer(9)]],
                    device atomic_float* dW  [[buffer(10)]],
                    device atomic_float* db  [[buffer(11)]],
                    uint2 _thread            [[thread_position_in_grid]]) {
    if (_thread.y >= osh.H || _thread.x >= osh.W) return;

    uint x  = _thread.x;
    uint y  = _thread.y;

    int in_y = y * s - p;
    int in_x = x * s - p;

    for (uint n = 0; n < osh.N; n++) {
        for (uint oc = 0; oc < osh.C; oc++) {
            uint out_index = index(osh, n, oc, y, x);
            for (uint c = 0; c < insh.C; c++) {
                for (uint i = 0; i < k; i++) {
                    if (in_y + (int)i < 0 || in_y + (int)i >= (int)insh.H) // padding
                        continue;
                    for (uint j = 0; j < k; j++) {
                        if (in_x + (int)j < 0 || in_x + (int)j >= (int)insh.W) // padding
                            continue;
                        uint inIndex = index(insh, n, c, in_y + i, in_x + j);
                        uint wIndex = c * osh.C * k * k + oc * k * k + i * k + j;
                        din[inIndex] += W[wIndex] * dout[out_index];
                        atomic_fetch_add_explicit(&dW[wIndex], in[inIndex] * dout[out_index], memory_order_relaxed);
                    }
                }
            }
            atomic_fetch_add_explicit(&db[oc], dout[index(osh, n, oc, y, x)], memory_order_relaxed);
        }
    }
}

kernel void max_pool(device const float* in [[buffer(0)]],
                     constant Shape& insh   [[buffer(1)]],
                     device float* out      [[buffer(2)]],
                     constant Shape& osh    [[buffer(3)]],
                     device uint* idxs      [[buffer(4)]],
                     uint3 _thread          [[thread_position_in_grid]]) {
    if (_thread.y >= osh.H || _thread.x >= osh.W || _thread.z >= osh.C) return;

    uint k = 3;
    uint s = 2;
    uint x = _thread.x;
    uint y = _thread.y;
    uint c = _thread.z;
    uint in_x = x * s;
    uint in_y = y * s;

    for (uint n = 0; n < osh.N; n++) {
        uint max_index = index(insh, n, c, in_y, in_x);
        for (uint i = 0; i < k; i++) {
            for (uint j = 0; j < k; j++) {
                uint _index = index(insh, n, c, in_y + i, in_x + j);
                if (in[_index] - in[max_index] > 1e-2)
                    max_index = _index;
            }
        }
        uint out_index = index(osh, n, c, y, x);
        out[out_index] = in[max_index];
        idxs[out_index] = max_index;
    }
}

kernel void max_pool_bw(device const float* dout [[buffer(0)]],
                        constant Shape& osh      [[buffer(1)]],
                        device const uint* idxs  [[buffer(2)]],
                        device atomic_float* din [[buffer(3)]],
                        uint2 gid                [[thread_position_in_grid]]) {
    if (gid.y >= osh.H || gid.x >= osh.W)
        return;
    for (uint n = 0; n < osh.N; n++) {
        for (uint c = 0; c < osh.C; c++) {
            uint out_index = index(osh, n, c, gid.y, gid.x);
            atomic_fetch_add_explicit(&din[idxs[out_index]], dout[out_index], memory_order_relaxed);
        }
    }
}

kernel void lrn(device const float* in [[buffer(0)]],
                constant Shape& insh   [[buffer(1)]],
                device float* out      [[buffer(2)]],
                uint3 _thread          [[thread_position_in_grid]]) {
    if (_thread.y >= insh.H || _thread.x >= insh.W || _thread.z >= insh.N)
        return;

    uint k = 2;
    uint n = 5;
    float alpha = 0.0001;
    float beta = 0.75;

    uint i = _thread.z;

    for (uint c = 0; c < insh.C; c++) {
        uint point = index(insh, i, c, _thread.y, _thread.x);

        float sum = pow(in[point], 2);

        for (uint cl = 1; cl <= n / 2 && c - cl >= 0; cl++)
            sum += pow(in[index(insh, i, c - cl, _thread.y, _thread.x)], 2);

        for (uint cr = 1; cr <= n / 2 && c + cr < insh.C; cr++)
            sum += pow(in[index(insh, i, c + cr, _thread.y, _thread.x)], 2);

        float divider = k + alpha * sum;
        out[point] = in[point] / pow(divider, beta);
    }
}

kernel void lrn_bw(device const float* in [[buffer(0)]],
                   constant Shape& insh   [[buffer(1)]],
                   device float* dout     [[buffer(2)]],
                   device float* din      [[buffer(3)]],
                   uint2 gid              [[thread_position_in_grid]]) {
    uint k = 2;
    uint n = 5;
    float alpha = 0.0001;
    float beta = 0.75;

    for (uint i = 0; i < insh.N; i++) {
        for (uint c = 0; c < insh.C; c++) {
            uint point = index(insh, n, c, gid.y, gid.x);

            float sum = pow(in[point], 2);

            for (uint cl = 1; cl <= n / 2 && c - cl >= 0; cl++)
                sum += pow(in[index(insh, i, c - cl, gid.y, gid.x)], 2);

            for (uint cr = 1; cr <= n / 2 && c + cr < insh.C; cr++)
                sum += pow(in[index(insh, i, c + cr, gid.y, gid.x)], 2);

            float divider = k + alpha * sum;
            float pow_result = pow(divider, beta);
            float pow_minus_one_result = pow(divider, beta - 1);

            din[point] = (pow_result - in[point] * beta * pow_minus_one_result * 2 * alpha / n * in[point]) / (pow_result * pow_result) * dout[point];
        }
    }
}

kernel void relu(device const float* in [[buffer(0)]],
                 constant uint& width   [[buffer(1)]],
                 device float* out      [[buffer(2)]],
                 uint2 gid              [[thread_position_in_grid]]) {
    uint index = gid.y * width + gid.x;
    out[index] = in[index] > 0 ? in[index] : 0;
}

kernel void relu_bw(device const float* in   [[buffer(0)]],
                    constant uint& width     [[buffer(1)]],
                    device const float* dout [[buffer(2)]],
                    device float* din        [[buffer(3)]],
                    uint2 gid                [[thread_position_in_grid]]) {
    uint index = gid.y * width + gid.x;
    din[index] = in[index] > 0 ? dout[index] : 0;
}

kernel void dropout(device const float* in [[buffer(0)]],
                    constant uint& width   [[buffer(1)]],
                    device float* out      [[buffer(2)]],
                    uint2 gid              [[thread_position_in_grid]]) {
    uint index = gid.y * width + gid.x;
    bool isZero = rand(gid.y, width, gid.x) > 0.5;
    out[index] = isZero ? 0 : in[index];
}

kernel void dropout_bw(device const float* out  [[buffer(0)]],
                       constant uint& width     [[buffer(1)]],
                       device const float* dout [[buffer(2)]],
                       device float* din        [[buffer(3)]],
                       uint2 gid                [[thread_position_in_grid]]) {
    uint index = gid.y * width + gid.x;
    din[index] = out[index] == 0 ? 0 : dout[index];
}

kernel void matmul(device const float* in [[buffer(0)]],
                   device const float* W  [[buffer(1)]],
                   constant uint& fan_in  [[buffer(2)]],
                   constant uint& fan_out [[buffer(3)]],
                   device float* out      [[buffer(4)]],
                   uint2 gid              [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint i = 0; i < fan_in; i++) {
        sum += in[gid.y * fan_in + i] * W[i * fan_out + gid.x];
    }
    out[gid.y * fan_out + gid.x] = sum;
}

kernel void bias(device float* in       [[buffer(0)]],
                 device const float* b  [[buffer(1)]],
                 constant uint& fan_out [[buffer(2)]],
                 uint2 gid              [[thread_position_in_grid]]) {
    in[gid.y * fan_out + gid.x] += b[gid.x];
}

kernel void bias_bw(device float* dout      [[buffer(0)]],
                    constant uint& fan_out  [[buffer(1)]],
                    device atomic_float* db [[buffer(2)]],
                    uint2 gid               [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&db[gid.x], dout[gid.y * fan_out + gid.x], memory_order_relaxed);
}

kernel void sgd(device float* w          [[buffer(0)]],
                device const float* dw   [[buffer(1)]],
                device float* v          [[buffer(2)]],
                constant uint& width     [[buffer(3)]],
                constant float& lr       [[buffer(4)]],
                uint2 gid                [[thread_position_in_grid]]) {
    uint index = gid.y * width + gid.x;
    v[index] = 0.9 * v[index] - 0.0005 * lr * w[index] - lr * dw[index];
    w[index] += v[index];
}
