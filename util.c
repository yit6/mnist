#include <math.h>
#include <stdio.h>
#include "main.h"

float sigmoid(float x) {
	return 1/(1+exp(-x));
}

float d_sigmoid(float x) {
	return exp(x)/(exp(2*x)+2*exp(x)+1);
}

float relu(float x) {
	return x > 0 ? x : 0;
}

float d_relu(float x) {
	return x > 0 ? 1 : 0;
}

float my_tan(float x) {
	return atan(x);
}

float d_tan(float x) {
	return 1/(1+x*x);
}

float mse(float *a, float *b, int num) {
	float err = 0;
	for (int i = 0; i < num; ++i) {
		err += powf(a[i]-b[i], 2);
	}
	return err/num;
}

void activate(float *vec, int num) {
	for (int i = 0; i < num; ++i) {
		vec[i] = ACTIVATION(vec[i]);
	}
}
