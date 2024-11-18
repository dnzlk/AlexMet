//
//  base.h
//  AlexNetC
//
//  Created by Denis Khabarov on 29.10.2024.
//

#include <stdbool.h>
#ifndef base_h
#define base_h

typedef struct {
    unsigned int N, C, H, W;
} Shape;

int n(Shape shape);

void exp_(float* in, int N, float* out);
void softmax(float* in, float* out);
void sub_max(float* a);

float sum(float* a, int N);
void div_all(float* a, int N, float divider);

void create_random_batch(int n, char* name);

#endif /* base_h */
