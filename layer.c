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

float *apply_layer(float *inputs, Layer *layer) {
	float *activations = (float *) malloc(layer->outputs*sizeof(float));

	for (int output = 0; output < layer->outputs; ++output) {
		float activation = layer->biases[output];
		for (int input = 0; input < layer->inputs; ++input) {
			activation += inputs[input]*layer->weights[input+layer->inputs*output];
		}
		activations[output] = activation;
	}
	return activations;
}

void print_layer(Layer *layer) {
	printf("\nLayer:\n");
	printf("%d inputs\n", layer->inputs);
	printf("%d outputs\n", layer->outputs);

	printf("\nBiases:\n");
	for (int i = 0; i < layer->outputs; ++i)
		printf("%f ", layer->biases[i]);

	printf("\n\nWeights:\n");

	for (int input = 0; input < layer->inputs; ++input) {
		printf("Input %d: ", input);
		for (int output = 0; output < layer->outputs; ++output) {
			printf("%f ", layer->weights[input+layer->inputs*output]);
		}
		printf("\n");
	}
}
