#include "base.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // for memcpy
#include <math.h> // for exp()

int n(Shape shape) {
    return shape.N * shape.C * shape.H * shape.W;
}

float sum(float* a, int N) {
    float result = 0;
    for (int i = 0; i < N; i++)
        result += a[i];
    return result;
}

void div_all(float* a, int N, float divider) {
    for (int i = 0; i < N; i++)
        a[i] /= divider;
}

void exp_(float* in, int N, float* out) {
    for (int i = 0; i < N; i++)
        out[i] = exp(in[i]);
}

void softmax(float* in, float* out) {
    float e[1000];
    exp_(in, 1000, e);
    float exp_sum = sum(e, 1000);

    for (int i = 0; i < 1000; i++)
        out[i] = exp(in[i]) / exp_sum;
}

void sub_max(float* in) {
    float max = in[0];
    for (int i = 1; i < 1000; i++)
        if (in[i] > max)
            max = in[i];
    for (int i = 0; i < 1000; i++)
        in[i] -= max;
}

void create_random_batch(int n, char* name) {
    float* a = malloc(n * sizeof(float));
    for (int i = 0; i < n; a[i++] = (float)random()/(float)RAND_MAX);
    FILE *file = fopen(name, "wb");
    fwrite(a, sizeof(float), n, file);
    fclose(file);
    free(a);
    return;
}
