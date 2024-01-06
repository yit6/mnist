#include <math.h>
#include <stdio.h>
#include "main.h"

float sigmoid(float x) {
	return 1/(1+exp(-x));
}

float d_sigmoid(float x) {
	return exp(x)/(exp(2*x)+2*exp(x)+1);
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
