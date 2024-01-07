#include <stdio.h>
#include <stdlib.h>

#include "main.h"

Network *create_network() {
	Network *net = (Network *) malloc(sizeof(Network));

	net->inputs = 2;
	net->outputs = 1;
	net->num_layers = 2;

	net->layers = (Layer **) malloc(2*sizeof(Layer));
	net->layers[0] = create_layer(2, 2);
	net->layers[1] = create_layer(2, 1);

	return net;
}

float *apply_network(float *inputs, Network *net) {
	float *last = 0;
	for (int i = 0; i < net->num_layers; ++i) {
		inputs = apply_layer(inputs, net->layers[i]);
		activate(inputs, net->layers[i]->outputs);
		if (last)
			free(last);
		last = inputs;
	}
	return inputs;
}

void train_network_sample(float *inputs, float *targets, Network *net) {

	// Store all of the activations
	float **activations = (float **) malloc((net->num_layers+1)*sizeof(float *));
	float **unactivated = (float **) malloc((net->num_layers+1)*sizeof(float *));
	activations[0] = inputs;
	for (int i = 0; i < net->num_layers; ++i) {
		unactivated[i+1] = apply_layer(activations[i], net->layers[i]);
		activations[i+1] = (float *) malloc(net->layers[i]->outputs*sizeof(float));
		for (int output = 0; output < net->layers[i]->outputs; ++output)
			activations[i+1][output] = unactivated[i+1][output];
		activate(activations[i+1], net->layers[i]->outputs);
	}

	float **d_activations = (float **) malloc((net->num_layers+1)*sizeof(float *));

	// Calculate partial derivative for final layer outputs
	d_activations[net->num_layers] = (float *) malloc(net->outputs*sizeof(float));
	for (int output = 0; output < net->outputs; ++output) {
		d_activations[net->num_layers][output] = 2*(activations[net->num_layers][output]-targets[output]);
	}

	// Backpropagation
	for (int l = net->num_layers-1; l >= 0; --l) {
		Layer *layer = net->layers[l];

		d_activations[l] = (float *) malloc(layer->inputs*sizeof(float));

		for (int input = 0; input < layer->inputs; ++input) {
			d_activations[l][input] = 0;
			for (int output = 0; output < layer->outputs; ++output) {
				d_activations[l][input] += D_ACTIVATION(unactivated[l+1][output])*layer->weights[input+layer->inputs*output]*d_activations[l+1][output];
			}
		}

		for (int output = 0; output < layer->outputs; ++output) {
			float d_b = d_activations[l+1][output]*D_ACTIVATION(unactivated[l+1][output]);
			layer->biases[output] -= d_b*STEP_SIZE;

			for (int input = 0; input < layer->inputs; ++input) {
				layer->weights[input+layer->inputs*output] -= d_b*activations[l][input]*STEP_SIZE;
			}
		}
	}

	for (int i = 1; i < net->num_layers+1; ++i) {
		free(activations[i]);
		free(d_activations[i]);
		free(unactivated[i]);
	}
	free(d_activations[0]);
	free(activations);
	free(unactivated);
	free(d_activations);
}

void print_network(Network *net) {
	printf("Network:\n");
	printf("%d inputs\n", net->inputs);
	printf("%d outputs\n", net->outputs);
	printf("%d layers\n", net->num_layers);

	for (int i = 0; i < net->num_layers; ++i) {
		print_layer(net->layers[i]);
	}
}
