#include <stdio.h>
#include <stdlib.h>

#include "main.h"

extern float step_size;

Network *create_network() {
	Network *net = (Network *) malloc(sizeof(Network));

	net->inputs = 28*28;
	net->outputs = 10;
	net->num_layers = 3;

	net->layers = (Layer **) malloc(net->num_layers*sizeof(Layer));
	net->layers[0] = create_layer(28*28, 128);
	net->layers[1] = create_layer(128, 64);
	net->layers[2] = create_layer(64, 10);

	net->layers[2]->activation = TANGENT;

	net->activations = (float **) malloc((net->num_layers+1)*sizeof(float *));
	net->unactivated = (float **) malloc((net->num_layers+1)*sizeof(float *));
	net->d_activated = (float **) malloc((net->num_layers+1)*sizeof(float *));

	for (int i = 0; i < net->num_layers; ++i) {
		net->activations[i] = (float *) malloc(net->layers[i]->inputs*sizeof(float *));
		net->unactivated[i] = (float *) malloc(net->layers[i]->inputs*sizeof(float *));
		net->d_activated[i] = (float *) malloc(net->layers[i]->inputs*sizeof(float *));
	}
	net->activations[net->num_layers] = (float *) malloc(net->outputs*sizeof(float *));
	net->unactivated[net->num_layers] = (float *) malloc(net->outputs*sizeof(float *));
	net->d_activated[net->num_layers] = (float *) malloc(net->outputs*sizeof(float *));

	return net;
}

void apply_network(float *inputs, float *outputs, Network *net) {
	for (int i = 0; i < net->inputs; ++i) {
		net->activations[0][i] = inputs[i];
	}
	for (int i = 0; i < net->num_layers; ++i) {
		apply_layer(net->activations[i], net->activations[i+1], net->layers[i]);
		activate(net->activations[i+1], net->layers[i]->outputs, net->layers[i]->activation);
	}
	for (int i = 0; i < net->outputs; ++i) {
		outputs[i] = net->activations[net->num_layers][i];
	}
}

void train_network_sample(float *inputs, float *targets, Network *net) {

	// Store all of the activations
	float **activations = net->activations;
	float **unactivated = net->unactivated;

	for (int i = 0; i < net->inputs; ++i) {
		activations[0][i] = inputs[i];
	}

	for (int i = 0; i < net->num_layers; ++i) {
		apply_layer(activations[i], unactivated[i+1], net->layers[i]);
		for (int output = 0; output < net->layers[i]->outputs; ++output)
			activations[i+1][output] = unactivated[i+1][output];
		activate(activations[i+1], net->layers[i]->outputs, net->layers[i]->activation);
	}

	float **d_activations = net->d_activated;

	// Calculate partial derivative for final layer outputs
	for (int output = 0; output < net->outputs; ++output) {
		d_activations[net->num_layers][output] = 2*(activations[net->num_layers][output]-targets[output]);
	}

	// Backpropagation
	for (int l = net->num_layers-1; l >= 0; --l) {
		Layer *layer = net->layers[l];

		for (int input = 0; input < layer->inputs; ++input) {
			d_activations[l][input] = 0;
			for (int output = 0; output < layer->outputs; ++output) {
				float (*d_activation)(float);
				switch (layer->activation) {
					case SIGMOID:
						d_activation = &d_sigmoid;
						break;
					case RELU:
						d_activation = &d_relu;
						break;
					case TANGENT:
						d_activation = &d_tan;
						break;
				}
				d_activations[l][input] += d_activation(unactivated[l+1][output])*layer->weights[input+layer->inputs*output]*d_activations[l+1][output];
			}
		}

		for (int output = 0; output < layer->outputs; ++output) {
			float (*d_activation)(float);
			switch (layer->activation) {
				case SIGMOID:
					d_activation = &d_sigmoid;
					break;
				case RELU:
					d_activation = &d_relu;
					break;
				case TANGENT:
					d_activation = &d_tan;
					break;
			}
			float d_b = d_activations[l+1][output]*d_activation(unactivated[l+1][output]);
			layer->biases[output] -= d_b*step_size;

			for (int input = 0; input < layer->inputs; ++input) {
				layer->weights[input+layer->inputs*output] -= d_b*activations[l][input]*step_size;
			}
		}
	}
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
