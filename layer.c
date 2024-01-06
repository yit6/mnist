#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "main.h"

Layer *create_layer(int inputs, int outputs) {
	Layer *layer = (Layer *) malloc(sizeof(Layer));
	layer->inputs = inputs;
	layer->outputs = outputs;
	layer->weights = (float *) malloc(inputs*outputs*sizeof(float));
	layer->biases = (float *) malloc(outputs*sizeof(float));

	for (int i = 0; i < inputs*outputs; ++i)
		layer->weights[i] = (float)rand()/(float)RAND_MAX;

	for (int i = 0; i < outputs; ++i)
		layer->biases[i] = (float)rand()/(float)RAND_MAX;

	return layer;
}

float get_weight(Layer *layer, int i, int j) {
	return layer->weights[j+layer->inputs*i];
}

float *apply_layer(float *inputs, Layer *layer) {
	float *activations = (float *) malloc(layer->outputs*sizeof(float));

	for (int i = 0; i < layer->outputs; ++i) {
		float activation = layer->biases[i];
		for (int j = 0; j < layer->inputs; ++j) {
			activation += inputs[j]*get_weight(layer, i, j);
		}
		activations[i] = activation;
	}
	return activations;
}
