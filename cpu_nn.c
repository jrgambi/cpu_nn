#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>

#include "my_cpu.h"
#include "my_neural_net.h"


#define FUNCTION_SIZE 1024

int
main (int argc, char *argv[])
{	
	time_t t;
	const time_t start_time = time (NULL);
	time_t last_save = start_time;
	int run_time = 0;
	char* file_name = "reg_nn.dat";
	double sum_error = 0;
	struct Perceptron_Layer PL;
	int size = 1; // number of instructions to generate, 
			// only 1 works with nn atm

	unsigned int *content = malloc (size * sizeof (unsigned int));

	double* nn_inputs = malloc (INPUTS_SIZE * sizeof (double));

	double* targets = malloc (OUTPUTS_SIZE * sizeof (double));
	
	unsigned int* outputs = mmap (NULL, OUTPUTS_SIZE * sizeof (double), 
			PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
	
	// FIXME: use register input size
	unsigned int* inputs = malloc (INPUTS_SIZE * sizeof (double)); 
	
        unsigned int* invocation = malloc (FUNCTION_SIZE);
	
	char *func = mmap (0, FUNCTION_SIZE, 
			PROT_READ | PROT_WRITE
			| PROT_EXEC, MAP_SHARED | MAP_ANON, -1, 0);	

	int my_func_return = 0;

	srandom ((unsigned) time(&t));
	
	if (access (file_name, F_OK) != -1)
	{
		PL = read_perceptron_layer_from_file (file_name);
	}
	else
	{
		int layer_sizes[] = {INPUTS_SIZE, 10, 10, OUTPUTS_SIZE};
		PL = initialize_perceptron_layer (sizeof (layer_sizes) / sizeof (int), layer_sizes, Sigmoid);
	}

	
	while (run_time < 3600)
	{
		generate_register_inputs (inputs);
		generate_instructions (content, size);
		create_invocation (content, size, invocation);


		//printf ("sizeof invocation = %d\n", sizeof (invocation));

		memset (func, '0', FUNCTION_SIZE);

		memcpy (func, invocation, FUNCTION_SIZE);

		//printf ("invoc gen done...\n");

		// sets output to error.  wrapper_function modifies value 
		// if successful  
		outputs[OUTPUTS_SIZE - 1] = 1; 

		wrapper_function (func, (unsigned int) &inputs[0], 
				(unsigned int) outputs, 
				inputs[INPUTS_SIZE - 2]);
			
		// last input equals instruction
		inputs[INPUTS_SIZE - 1] = content[0]; 
		

		// FIXME: use register input size
		for (int a = 0; a < INPUTS_SIZE; a++)
		{ // scale each input
			nn_inputs[a] = (double) inputs[a] / UINT_MAX;
			//printf ("input[%d] = %x\n", a, inputs[a]);
		}


		// store decimall representation of output bytes at a time
		// values will be nummerical value of bytes 
		// with decimal point shifted so that value is between 0 and 1
		for (int b = 0; b < OUTPUTS_SIZE; b++)
		{
			//printf ("output[%d] = %x\n", b, outputs[b]);
			// on instruction error set outputs to inputs
			// FIXME: handle R2
			if (outputs[OUTPUTS_SIZE - 1] != 0 && b < INPUTS_SIZE - 2) 
			{ 
				outputs[b] = inputs[b];
				//printf ("Illegal Instruction? %f\n", outputs[b]);
			}
			if (b == OUTPUTS_SIZE - 2) // condition bytes output
			{

				targets[b] = (double) (outputs[b] 
						/ UINT_MAX);
				break;
			}
			//printf ("outputs[%d] = %x\n", b, outputs[b]);

			unsigned int temp = outputs[b] & 0x0000FFFF;
			//printf ("temp1 = %x\n", temp);
			targets[b] = (double) temp / 65536; 

			temp = (outputs[b] & 0xFFFF0000) >> 16;
			targets[b+1] = (double) temp / 65536;
			//printf ("temp2 = %x\n", temp);
		}

		//print_perceptron_layer (PL);

		//printf ("NN junk...\n");

		set_perceptron_layer_inputs (PL, nn_inputs);
		generate_outputs (PL);
		perform_training_pass (PL, targets);
		
		run_time = time (NULL) - start_time;
		

		if ((time (NULL) - last_save) > 300)
		{// save every ~5 minutes
			last_save = time (NULL);
			write_perceptron_layer_to_file (file_name, PL);
			print_perceptron_layer (PL);
		}
	}	
	sum_error = sum_squared_error (PL, targets);
	print_perceptron_layer (PL);
	write_perceptron_layer_to_file (file_name, PL);

	double* outs = malloc (OUTPUTS_SIZE * sizeof (double));
	for (int z = 0; z < OUTPUTS_SIZE; z++)
	{
		printf ("target[%d] = %.9g ... nn_output = %.9g\n",
			       	z, targets[z], 
				PL.pl[PL.num_layers - 1][z].output);
		outs[z] = PL.pl[PL.num_layers - 1][z].output;
	}

	printf ("error = %f\n", sum_error);

/*
 	// TODO:
 	// convert outputs to more readable version
	for (int zz = 0; zz < 10; zz++)
	{
		double cur_nn_out1 = PL.pl[PL.num_layers - 1][2 * zz].output;
		double cur_nn_out2 = PL.pl[PL.num_layers - 1][(2 * zz) + 1].output;
		unsigned int t = (unsigned int) (cur_nn_out1 * 65536);

//		printf ("o1 = %f ... o2 = %f ... t = %x\n", cur_nn_out1, 
//				cur_nn_out2, t);

		unsigned int temp = t;
		t = (unsigned int) (cur_nn_out2 * 65536);
		temp = temp | (t << 16);
		printf ("output[%d] = %x ... mod nn_o = %x\n", zz, 
				outputs[zz], temp);
	}
*/
}

