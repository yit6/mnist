#include <stdio.h>
#include <stdlib.h>

#include <endian.h>

#include "main.h"

void total_loss(Layer *layer);
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

	Layer *layer = create_layer(2, 2);
	//total_loss(layer);
	
	float inputs[9][2] = {
		{ 0.1, 0.1 },
		{ 0.1, 0.5 },
		{ 0.1, 0.9 },
		{ 0.5, 0.1 },
		{ 0.5, 0.5 },
		{ 0.5, 0.9 },
		{ 0.9, 0.1 },
		{ 0.9, 0.5 },
		{ 0.9, 0.9 },
	};

	for (int epoch = 0; epoch < 10000; ++epoch) {
		printf("Epoch %d:\n", epoch);
		total_loss(layer);
		for (int i = 0; i < 9; ++i)
			train_sample(inputs[i], inputs[i], layer);
	}
	total_loss(layer);
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

	for (int i = 0; i < layer->outputs; ++i) {
		printf("Output was %f, target is %f, delta is %f\n", outputs[i], targets[i], d_out_activations[i]);
	}
}

void total_loss(Layer *layer) {
	float *inputs = malloc(1*sizeof(float));
	float *outputs;
	float loss = 0;

	inputs[0] = 0.0;
	inputs[1] = 0.0;

	outputs = apply_layer(inputs, layer);
	activate(outputs, layer->outputs);
	loss += mse(inputs, outputs, 2);
	free(outputs);

	inputs[0] = 0.0;
	inputs[1] = 1.0;

	outputs = apply_layer(inputs, layer);
	activate(outputs, layer->outputs);
	loss += mse(inputs, outputs, 2);
	free(outputs);

	inputs[0] = 1.0;
	inputs[1] = 0.0;

	outputs = apply_layer(inputs, layer);
	activate(outputs, layer->outputs);
	loss += mse(inputs, outputs, 2);
	free(outputs);

	inputs[0] = 1.0;
	inputs[1] = 1.0;

	outputs = apply_layer(inputs, layer);
	activate(outputs, layer->outputs);
	loss += mse(inputs, outputs, 2);
	free(outputs);

	printf("loss: %f\n", loss);
	free(inputs);
}
