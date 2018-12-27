#include "my_neural_net.h"

int
main (int argc, char *argv[])
{
	
	// test for calculate_input_summation
	struct Perceptron first;
	first.num_inputs = 2;
	first.inputs = malloc (first.num_inputs * sizeof (struct Perceptron_Input));
	first.bias = 0;
	first.inputs[0].weight = 0.4f;
	//first.inputs[0].input = malloc (1);
	first.inputs[0].input = 10;
	first.inputs[1].weight = 0.5f;
	//first.inputs[1].input = malloc (1);
	first.inputs[1].input = 12;

	double first_sum = calculate_input_summation (first);

	printf ("first sum = %f\n", first_sum);


	
	//initialization test
	int layer_sizes[] = {2, 3, 7, 2};
	struct Perceptron_Layer perceptron_layer = initialize_perceptron_layer (4, layer_sizes, Sigmoid);

	print_perceptron_layer (perceptron_layer);
	write_perceptron_layer_to_file ("test.txt", perceptron_layer);


	struct Perceptron_Layer pl2 = read_perceptron_layer_from_file ("test.txt"); 

	pl2.pl[0][0].inputs[0].input = 10;
	pl2.pl[0][1].inputs[0].input = 5;

	//pl2.pl[0][0].inputs[0].weight = 1.0f;
	//pl2.pl[0][1].inputs[0].weight = 1.0f;

	generate_outputs (pl2);

	print_perceptron_layer (pl2);

	double targets[] = {0.794356548, 0.387986545};

	for (int i = 0; i < 1000; i++) {
		generate_outputs (pl2);
		perform_training_pass (pl2, targets);
	}

	print_perceptron_layer (pl2);

	for (int i = 0; i < 2; i++)
	{
		printf ("target[%d] %.9g\n", i, targets[i]);
	}
}
