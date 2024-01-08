#include <stdio.h>
#include <stdlib.h>

#include <endian.h>

#include "main.h"

float step_size = 0.05;

void train_sample(float *inputs, float *targets, Layer *layer);

int main(int argc, char **argv) {
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

	if (argc == 2) {
		step_size = atof(argv[1]);
		printf("Using step size %f\n", step_size);
	}

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

	float *out = (float *) malloc(net->outputs*sizeof(float));

	for (int epoch = 0; epoch < 100000; ++epoch) {
		//if (epoch % 100 == 0) printf("\nEpoch %d:\n", epoch);
		for (int i = 0; i < 4; ++i) {
			train_network_sample(inputs[i], outputs[i], net);
		}
		float loss = 0;
		for (int i = 0; i < 4; ++i) {
			apply_network(inputs[i], out, net);
			loss += mse(out, outputs[i], 1);
			if (epoch % 100 == 0) printf("Input %f, %f, Output %f, Target: %f\n", inputs[i][0], inputs[i][1], out[0], outputs[i][0]);
		}
		if (epoch % 100 == 0) {
			printf("%d, %f\n", epoch, loss);
		}
	}

	print_network(net);
}
