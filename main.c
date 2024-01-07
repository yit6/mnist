#include <stdio.h>
#include <stdlib.h>

#include <endian.h>

#include "main.h"

void train_sample(float *inputs, float *targets, Layer *layer);

int main() {
//	unsigned char *training_images = read_idx3("data/train-images-idx3-ubyte");
//	unsigned char *labels = read_idx1("data/train-labels-idx1-ubyte");
//	for (int i = 0; i < 10; i++) {
//		for (int x = 0; x < 28; ++x) {
//			for (int y = 0; y < 28; ++y) {
//				if(training_images[28*28*i+x*28+y] > 0) {
//					putchar('#');
//				} else {
//					putchar(' ');
//				}
//			}
//			printf("\n");
//		}
//		printf("label: %d\n", labels[i]);
//	}

	float inputs[4][2] = {
		{ 0.0, 0.0 },
		{ 0.0, 1.0 },
		{ 1.0, 0.0 },
		{ 1.0, 1.0 },
	};

	float outputs[4][1] = {
		{ 0.0 },
		{ 1.0 },
		{ 1.0 },
		{ 0.0 },
	};

	Network *net = create_network();

	print_network(net);

	for (int epoch = 0; epoch < 100000; ++epoch) {
		if (epoch % 100 == 0) printf("\nEpoch %d:\n", epoch);
		for (int i = 0; i < 4; ++i) {
			train_network_sample(inputs[i], outputs[i], net);
		}
		float loss = 0;
		for (int i = 0; i < 4; ++i) {
			float *out = apply_network(inputs[i], net);
			loss += mse(out, outputs[i], 1);
			if (epoch % 100 == 0) printf("Input %f, %f, Output %f, Target: %f\n", inputs[i][0], inputs[i][1], out[0], outputs[i][0]);
			free(out);
		}
		if (epoch % 100 == 0) printf("Loss: %f\n", loss);
	}

	print_network(net);
}

void train_sample(float *inputs, float *targets, Layer *layer) {
	float *outputs = apply_layer(inputs, layer);
	activate(outputs, layer->outputs);
	float *d_out_activations = (float *) malloc(layer->outputs*sizeof(float));

	for (int i = 0; i < layer->outputs; ++i) {
		d_out_activations[i] = 2*(outputs[i]-targets[i]);
	}

	float *d_weights = (float *) malloc(layer->inputs*layer->outputs);
	float *d_biases = (float *) malloc(layer->outputs);

	for (int i = 0; i < layer->outputs; ++i) {
		for (int j = 0; j < layer->inputs; ++j) {
			d_weights[j+layer->inputs*i] = d_out_activations[i]*D_ACTIVATION(outputs[i])*inputs[j];
		}
	}

	for (int i = 0; i < layer->outputs; ++i) {
		d_biases[i] = d_out_activations[i]*D_ACTIVATION(outputs[i]);
	}

	for (int i = 0; i < layer->outputs*layer->inputs; ++i) {
		layer->weights[i] -= d_weights[i] * STEP_SIZE;
	}
	for (int i = 0; i < layer->outputs; ++i) {
		layer->biases[i] -= d_biases[i] * STEP_SIZE;
	}
}
