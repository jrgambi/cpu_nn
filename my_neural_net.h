#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <math.h> // !!! compiler needs "gcc MyNeuralNet.c -lm"
//#include "MyNeuralNet.c"

// activation function type
//  Types:  (in OpenNN)
// 	Linear x
// 	Symmetric Threshold if x > 0 return 1 else return -1
// 	Threshold if x > 0 return 1 else return 0
// 	Hyperbolic Tangent ((2 / (1 + e ^ (-2 * x))) - 1)
// 	Sigmoid (1 / (1 + e ^ -x))
// 	ReLu (max(0, x)) // if (x > 0) return x, else return 0
enum Activation_Function_Type {
	Linear,
	Symmetric_Threshold,
	Threshold,
	Hyperbolic_Tangent,
	Sigmoid,
	ReLu
};

enum Error_Training_Type {
	Sum_Squared
};

enum Training_Algorithmn {
	Gradient_Descent
};

struct Perceptron {
int num_inputs;
double bias;
double output;//TODO: make a pointer, (void*)?
enum Activation_Function_Type aft;
struct Perceptron_Input* inputs;
};

struct Perceptron_Input {
	float weight;
	double input;//TODO: make a pointer, (void*)?
};


struct Perceptron_Layer {
	int num_layers;
	int* layer_sizes;
	int num_parameters;
	struct Perceptron** pl;
	enum Error_Training_Type errtt;
	enum Training_Algorithmn traina;
	double training_rate;
};

double 
calculate_input_summation (struct Perceptron p);

double 
calculate_activation_function (struct Perceptron p);

double 
calculate_activation_function_derivative (struct Perceptron p);

struct Perceptron_Layer 
initialize_perceptron_layer (int num_layers, int* layer_sizes, enum Activation_Function_Type aft);

void 
print_perceptron_layer (struct Perceptron_Layer PL);

void
set_perceptron_layer_inputs (struct Perceptron_Layer PL, double* inputs);

char* 
aft_to_string (enum Activation_Function_Type aft);

enum Activation_Function_Type 
string_to_aft (char* string);

void 
write_perceptron_layer_to_file (char* filename, struct Perceptron_Layer PL);

struct Perceptron_Layer read_perceptron_layer_from_file (char* filename);

void generate_outputs (struct Perceptron_Layer PL);

double sum_squared_error (struct Perceptron_Layer PL, double* targets);

void normailze_training_direction (double* training_direction, int num_params);

void perform_training_pass (struct Perceptron_Layer PL, double* targets);


