#include <stdio.h>
#include <stdlib.h>

#include "main.h"

extern float step_size;

Network *create_network() {
	Network *net = (Network *) malloc(sizeof(Network));

	net->inputs = 784;
	net->outputs = 10;
	net->num_layers = 3;

	net->layers = (Layer **) malloc(net->num_layers*sizeof(Layer));
	net->layers[0] = create_layer(784, 128);
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

void save_network(Network *net, FILE *fp) {

	// Write number of layers
	fwrite(&net->num_layers, sizeof(int), 1, fp);

	// Write number of activations in input
	fwrite(&net->inputs, sizeof(int), 1, fp);

	// Write number of activations in output for each layer
	for (int i = 0; i < net->num_layers; ++i)
		fwrite(&net->layers[i]->outputs, sizeof(int), 1, fp);

	// Write each layer
	for (int i = 0; i < net->num_layers; ++i) {

		// Write activation function
		fwrite(&net->layers[i]->activation, sizeof(int), 1, fp);

		// Write weights
		fwrite(net->layers[i]->weights, sizeof(float), net->layers[i]->inputs*net->layers[i]->outputs, fp);

		// Write biases
		fwrite(net->layers[i]->biases, sizeof(float), net->layers[i]->outputs, fp);
	}
}

Network *load_network(FILE *fp) {
	Network *out = (Network *) malloc(sizeof(Network));

	// Read number of layers
	fread(&out->num_layers, sizeof(int), 1, fp);

	out->layers = (Layer **) malloc(out->num_layers*sizeof(Layer));

	// Read activations in each layer
	int *num_activations = (int *) malloc((out->num_layers+1)*sizeof(int));
	fread(num_activations, sizeof(int), out->num_layers+1, fp);

	// Set inputs and outputs
	out->inputs = num_activations[0];
	out->outputs = num_activations[out->num_layers];

	// Read each layer
	for (int i = 0; i < out->num_layers; ++i) {
		out->layers[i] = (Layer *) malloc(sizeof(Layer));
		Layer *layer = out->layers[i];
		layer->inputs = num_activations[i];
		layer->outputs = num_activations[i+1];

		// Read activation function
		fread(&layer->activation, sizeof(int), 1, fp);

		// Read weights
		layer->weights = (float *) malloc(layer->inputs*layer->outputs*sizeof(float));
		fread(layer->weights, sizeof(float), layer->inputs*layer->outputs, fp);

		// Read biases
		layer->biases = (float *) malloc(layer->outputs*sizeof(float));
		fread(layer->biases, sizeof(float), layer->outputs, fp);
	}

	out->activations = (float **) malloc((out->num_layers+1)*sizeof(float *));
	out->unactivated = (float **) malloc((out->num_layers+1)*sizeof(float *));
	out->d_activated = (float **) malloc((out->num_layers+1)*sizeof(float *));

	for (int i = 0; i < out->num_layers; ++i) {
		out->activations[i] = (float *) malloc(out->layers[i]->inputs*sizeof(float *));
		out->unactivated[i] = (float *) malloc(out->layers[i]->inputs*sizeof(float *));
		out->d_activated[i] = (float *) malloc(out->layers[i]->inputs*sizeof(float *));
	}
	out->activations[out->num_layers] = (float *) malloc(out->outputs*sizeof(float *));
	out->unactivated[out->num_layers] = (float *) malloc(out->outputs*sizeof(float *));
	out->d_activated[out->num_layers] = (float *) malloc(out->outputs*sizeof(float *));

	return out;
}
