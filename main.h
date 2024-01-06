#define ACTIVATION sigmoid
#define D_ACTIVATION d_sigmoid

#define STEP_SIZE 0.05

// Read idx file of dimension 3 into a flat array
unsigned char *read_idx3(char *filename);

// Read idx file of dimension 1
unsigned char *read_idx1(char *filename);

// Evaluate a sigmoid activiation function for a float
float sigmoid(float x);
float d_sigmoid(float x);

// Activate a vector
void activate(float *vec, int num);

// Find the mean squared error between two lists
float mse(float *a, float *b, int num);

typedef struct layer {
	int inputs;
	int outputs;
	float *weights;
	float *biases;
} Layer;

Layer *create_layer(int inputs, int outputs);
float *apply_layer(float *inputs, Layer *layer);
