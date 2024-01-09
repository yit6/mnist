#include <stdio.h>
#include <stdlib.h>

#include <endian.h>

#include "main.h"

float step_size = 0.05;

void show_sample(float *input, float *target, Network *net);
void testing_accuracy(Network *net);

float training_inputs[60000][28*28] = { { 0.0 } };
float training_outputs[60000][10] = { { 0.0 } };
float testing_inputs[10000][28*28] = { { 0.0 } };
float testing_outputs[10000][10] = { { 0.0 } };

int main(int argc, char **argv) {

	unsigned char *training_images = read_idx3("data/train-images-idx3-ubyte");
	unsigned char *training_labels = read_idx1("data/train-labels-idx1-ubyte");

	unsigned char *testing_images = read_idx3("data/t10k-images-idx3-ubyte");
	unsigned char *testing_labels = read_idx1("data/t10k-labels-idx1-ubyte");

	for (int i = 0; i < 60000; i++) {
		for (int j = 0; j < 28*28; ++j) {
			training_inputs[i][j] = ((float) training_images[i*28*28+j]) / 256;
		}
		training_outputs[i][training_labels[i]] = 1;
	}

	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 28*28; ++j) {
			testing_inputs[i][j] = ((float) testing_images[i*28*28+j]) / 256;
		}
		testing_outputs[i][testing_labels[i]] = 1;
	}

	if (argc == 2) {
		step_size = atof(argv[1]);
		printf("Using step size %f\n", step_size);
	}

	Network *net = create_network();

	for (int i = 0; i < 10; ++i)
		show_sample(training_inputs[i], training_outputs[i], net);

	float *out = (float *) malloc(net->outputs*sizeof(float));

	for (int epoch = 0; epoch < 100; ++epoch) {
		printf("\nEpoch %d:\n", epoch);
		for (int i = 0; i < 60000; ++i) {
			if (i % 600 == 0) { putchar('.'); fflush(stdout); }
			train_network_sample(training_inputs[i], training_outputs[i], net);
		}
		printf("\nCalculating loss\n");
		float loss = 0;
		for (int i = 0; i < 60000; ++i) {
			apply_network(training_inputs[i], out, net);
			loss += mse(out, training_outputs[i], 1);
			//if (epoch % 100 == 0) printf("Input %f, %f, Output %f, Target: %f\n", [i][0], inputs[i][1], out[0], outputs[i][0]);
		}
		printf("%d, %f\n", epoch, loss);
	}

	testing_accuracy(net);

	for (int i = 0; i < 10; ++i)
		show_sample(training_inputs[i], training_outputs[i], net);
}

void show_sample(float *input, float *target, Network *net) {
	float *out = (float *) malloc(net->outputs*sizeof(float));
	apply_network(input, out, net);

	for (int y = 0; y < 28; ++y) {
		for (int x = 0; x < 28; ++x) {
			if (input[x+y*28] > 0.1) {
				putchar('#');
			} else {
				putchar(' ');
			}
		}
		putchar('\n');
	}

	for (int i = 0; i < 10; ++i) {
		if (target[i] == 1.0) {
			printf("[%d] ", i);
		} else {
			printf(" %d  ", i);
		}
		for (int j = 0; j < clamp(out[i], 0, 1)*50; ++j) {
			putchar('#');
		}
		putchar('\n');
	}
}

void testing_accuracy(Network *net) {
	int num_correct = 0;
	float *out = (float *) malloc(10*sizeof(float));
	for (int i = 0; i < 10000; ++i) {
		apply_network(testing_inputs[i], out, net);
		int max_idx = -1, label = -1;
		float max_val = -1.0;
		for (int j = 0; j < 10; ++j) {
			if (out[j] > max_val) {
				max_idx = j;
				max_val = out[j];
			}
			if (testing_outputs[i][j] == 1.0) {
				label = j;
			}
		}
		if (label == max_idx)
			++num_correct;
		else
			show_sample(testing_inputs[i], testing_outputs[i], net);
	}
	printf("%d out of 10000\n", num_correct);
}
