#define ACTIVATION sigmoid
#define D_ACTIVATION d_sigmoid

// Read idx file of dimension 3 into a flat array
unsigned char *read_idx3(char *filename);

// Read idx file of dimension 1
unsigned char *read_idx1(char *filename);

// Evaluate a sigmoid activiation function for a float
float sigmoid(float x);
float d_sigmoid(float x);

float relu(float x);
float d_relu(float x);

float my_tan(float x);
float d_tan(float x);

// Activate a vector
void activate(float *vec, int num);

// Find the mean squared error between two lists
float mse(float *a, float *b, int num);

// Layers
typedef struct layer {
	int inputs;
	int outputs;
	float *weights;
	float *biases;
} Layer;

Layer *create_layer(int inputs, int outputs);
void apply_layer(float *inputs, float *outputs, Layer *layer);
void print_layer(Layer *layer);

// Networks
typedef struct network {
	Layer **layers;
	int num_layers;
	int inputs;
	int outputs;

	float **activations;
	float **unactivated;
	float **d_activated;
} Network;

Network *create_network();
void apply_network(float *inputs, float *outputs, Network *net);
void train_network_sample(float *inputs, float *targets, Network *net);
void print_network(Network *net);
