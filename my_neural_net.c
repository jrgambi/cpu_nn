#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <math.h> // !!! compiler needs "gcc MyNeuralNet.c -lm"

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
	int num_parameters; // FIXME: not used currently
	struct Perceptron** pl;
	enum Error_Training_Type errtt;
	enum Training_Algorithmn traina;
	double training_rate;
};

// calculate input sumation
// sum += foreach { weight[i] * input[i]} + bias
double
calculate_input_summation (struct Perceptron p){

	double sum = p.bias; 
	//printf ("sum bias = %.9g\n", sum);
	
	for (int i = 0; i < p.num_inputs; i++){
	//	printf ("weight = %f ... input = %f\n", p.inputs[i].weight, p.inputs[i].input);
		sum += p.inputs[i].weight * p.inputs[i].input;
	}
	//printf ("sum = %.9g\n", sum);

	return sum;
}

double
calculate_activation_function (struct Perceptron p){
	
	double sum = calculate_input_summation (p);

	switch (p.aft){
// 	Linear x
		case Linear:
			//printf ("Linear aft\n");
			return sum;
			break;

// 	Symmetric Threshold if x > 0 return 1 else return -1
		case Symmetric_Threshold:
			if (sum < 0)
				return -1.0;
			else 
				return 1.0;
			break;

// 	Threshold if x > 0 return 1 else return 0
		case Threshold:
			if (sum < 0)
				return 0.0;
			else 
				return 1.0;
			break;


// 	Hyperbolic Tangent ((2 / (1 + e ^ (-2 * x))) - 1)
		case Hyperbolic_Tangent:
			return ((2 / (1 + exp(-2 * sum))) -1);
			break;

// 	Sigmoid (1 / (1 + e ^ -x))
		case Sigmoid:
			//printf ("Sigimoid...\n");
			return (1 / (1 + exp(-1 * sum)));
			break;

// 	ReLu (max(0, x)) // if (x > 0) return x, else return 0
		case ReLu:
			if (sum > 0)
				return sum;
			else
				return 0;
			break;
		default:
			return 0;
	}
}


// activation function derivative // FIXME: cases where derivative not used
double
calculate_activation_function_derivative (struct Perceptron p){
	
	double sum = calculate_input_summation (p);

	switch (p.aft){
// 	Linear x
		case Linear:
			return 1.0;
			break;

// 	Symmetric Threshold derivative not used
		case Symmetric_Threshold:// 
			if (sum != 0.0)
				return 0.0;
			else 
				return 1.0;
			break;

// 	Threshold derivative not used
		case Threshold:
			if (sum != 0)
				return 0.0;
			else 
				return 1.0;
			break;


// 	Hyperbolic Tangent ((2 / (1 + e ^ (-2 * x))) - 1)
		case Hyperbolic_Tangent:
			//double tan_sum = tanh (sum);
			return (1 - (tanh (sum) * tanh (sum)));
			break;

// 	Sigmoid (1 / (1 + e ^ -x))
		case Sigmoid:
//			printf ("sig' = %f\n", (1.0 - (1.0 / (1.0 + (exp (-1 * sum))))));
			return ((1.0 / (1.0 + (exp (-1 * sum)))) * (1.0 - (1.0 / (1.0 + (exp (-1 * sum)))))); // not sure which is correct

			//return (sum * (1 - sum));
			break;

// 	ReLu derivative not used
		case ReLu:
			if (sum > 0)
				return sum;
			else
				return 0;
			break;
		default:
			return 0;
	}
}

// setup starting Perceptron_Layer with number of layers and layer sizes 
// num_layers must equal size of layer_sizes
struct Perceptron_Layer
initialize_perceptron_layer (int num_layers, int* layer_sizes, enum Activation_Function_Type aft){
	time_t t;
	srandom ((unsigned) time(&t));

	int num_params = 0;

	struct Perceptron_Layer PL;
	PL.errtt = Sum_Squared;
	PL.traina = Gradient_Descent;

	PL.training_rate = 0.1;

	PL.num_layers = num_layers;
	PL.layer_sizes = layer_sizes;

	struct Perceptron** pl = (struct Perceptron**) malloc (num_layers 
			* sizeof (struct Perceptron*));
	
	PL.pl = pl;
	for (int layer = 0; layer < num_layers; layer++){
		pl[layer] = (struct Perceptron*) malloc (layer_sizes [layer] 
				* sizeof (struct Perceptron));
	}


	
	
	int previous_lsize = 1;
	// skip first layer as that needs to be defined outside this function 
	for (int l = 0; l < num_layers; l++){
		int lsize = layer_sizes[l];
		//printf ("layer = %d ... lsize = %d\n", l, lsize);
		for (int n = 0; n < lsize; n++){
			
			struct Perceptron temp; 
			temp.inputs = malloc (previous_lsize *
				       	sizeof (struct Perceptron_Input));
			temp.aft = aft;
			temp.num_inputs = previous_lsize;
			
			temp.bias = (double)random (); 
			temp.bias = temp.bias / DBL_MAX;

			//num_params++; // bias is junk

			temp.output = 0;//malloc (1);
			if (l == 0) {
				pl[l][n] = temp;
				continue;
			}
			//printf ("\tpre lsize = %d\n", previous_lsize);

			for (int j = 0; j < previous_lsize; j++){
				//printf ("\t\tl - 1 = %d, j = %d\n", l - 1, j);
				temp.inputs[j].weight = (float)(random () / (float) RAND_MAX);
				temp.inputs[j].input = pl[l - 1][j].output;
				num_params++;
			}

			pl[l][n] = temp;
		}
		previous_lsize = lsize;
	}
	PL.num_parameters = num_params;

	return PL;
}

// function to set inputs. 
// size of first layer must equal size of in
void
set_perceptron_layer_inputs (struct Perceptron_Layer PL, double* in)
{
	for (int i = 0; i < PL.layer_sizes[0]; i++)
	{
		//printf ("in[%d]\n", i);
		PL.pl[0][i].inputs[0].input = in[i];
	}
}

// display some info of perceptron layer
void
print_perceptron_layer (struct Perceptron_Layer PL) {
	
	for (int l = 0; l < PL.num_layers; l++){

		printf ("layer = %d ... layer_size = %d\n", 
				l, PL.layer_sizes[l]);
		for (int s = 0; s < PL.layer_sizes[l]; s++){

			printf ("\toutput ptr = %.9g ... num_inputs = %d\n", 
					PL.pl[l][s].output, 
					PL.pl[l][s].num_inputs);
			
			for (int i = 0; i < PL.pl[l][s].num_inputs; i++){
				printf ("\t\tinput[%d] = %f ... weight = %.9g\n"
						, i, PL.pl[l][s].inputs[i].input
						, PL.pl[l][s].inputs[i].weight);
			}
		}
	}

}

// enum to string
char*
aft_to_string (enum Activation_Function_Type aft) 
{
	
	switch (aft){
// 	Linear x
		case Linear:
			return "Linear";
			break;

// 	Symmetric Threshold if x > 0 return 1 else return -1
		case Symmetric_Threshold:
			return "Symmetric Threshold";
			break;

// 	Threshold if x > 0 return 1 else return 0
		case Threshold:
			return "Threshold";
			break;


// 	Hyperbolic Tangent ((2 / (1 + e ^ (-2 * x))) - 1)
		case Hyperbolic_Tangent:
			return "Hyperbolic Tangent";
			break;

// 	Sigmoid (1 / (1 + e ^ -x))
		case Sigmoid:
			return "Sigmoid";
			break;

// 	ReLu (max(0, x)) // if (x > 0) return x, else return 0
		case ReLu:
			return "ReLu";
			break;
		default:
			return "Linear";
	}
}

// string to enum
enum Training_Algorithmn
string_to_traina (char* string)
{
	if (strcmp (string, "Gradient_Descent") == 0)
		return Gradient_Descent;

	return Gradient_Descent;
}

//enum to string
char*
traina_to_string (enum Training_Algorithmn traina)
{

	switch (traina){
		case Gradient_Descent:
			return "Gradient_Descent";
			break;
		
		default:
			return "Gradient_Descent";
	}

}

// convert string to enum
enum Activation_Function_Type
string_to_aft (char* string)
{
// 	Linear x
	if (strcmp (string, "Linear") == 0)
		return Linear;

// 	Symmetric Threshold if x > 0 return 1 else return -1
	if (strcmp (string, "Symmetric Threshold") == 0)
		return Symmetric_Threshold;

// 	Threshold if x > 0 return 1 else return 0
	if (strcmp (string, "Threshold") == 0)
		return Threshold;

// 	Hyperbolic Tangent ((2 / (1 + e ^ (-2 * x))) - 1)
	if (strcmp (string, "Hyperbolic Tangent") == 0)
		return Hyperbolic_Tangent;

// 	Sigmoid (1 / (1 + e ^ -x))
	if (strcmp (string, "Sigmoid") == 0)
		return Sigmoid;

// 	ReLu (max(0, x)) // if (x > 0) return x, else return 0
	if (strcmp (string, "ReLu") == 0)
		return ReLu;
	
	return Linear;
}

// write struct Perceptron_Layer out in file format
// each perceptron in layer tab seperated, each layer newline
// aft,bias, weight[0], weight[1], ..., weight[n]\taft,bias,weight[0-n]
void
write_perceptron_layer_to_file (char* filename, struct Perceptron_Layer PL)
{
	FILE* fp = fopen (filename, "w+"); 

	char* aft_string; 
	int err = 0;

	err += fprintf (fp, ":::%s:::\n", traina_to_string (PL.traina));

	for (int l = 0; l < PL.num_layers; l++){
		for (int n = 0; n < PL.layer_sizes[l]; n++){
			aft_string = aft_to_string (PL.pl[l][n].aft);
			err += fprintf (fp, "%s,%.9g", aft_string, PL.pl[l][n].bias);
			
			for (int i = 0; i < PL.pl[l][n].num_inputs; i++){
				err += fprintf (fp, ",%.9g", PL.pl[l][n].inputs[i].weight);
			}
			err += fprintf (fp, "\t");
		}
		err += fprintf (fp, "\n");
	}

	fclose (fp);
}

// read file and convert it into struct Perceptron_Layer
struct Perceptron_Layer 
read_perceptron_layer_from_file (char* filename)
{
	FILE* fp = fopen (filename, "r");

	char* line = NULL;
	size_t len = 0;
	ssize_t read;

	int num_layers = 0;
	int curr_layer = 0;
	int curr_percep = 0;


	struct Perceptron_Layer PL;
	PL.errtt = Sum_Squared;
	PL.traina = Gradient_Descent;

	PL.training_rate = 0.1;

	PL.num_parameters = 0;
	// get num_layers
	while ((read = getline (&line, &len, fp)) != -1){
		num_layers++;
	}

	PL.pl = (struct Perceptron**) malloc (num_layers 
			* sizeof (struct Perceptron*));
	PL.layer_sizes = malloc (num_layers * sizeof (int));

	PL.num_layers = num_layers - 1; 
	// minus 1 because first line doesn't hold any perceptrons

	int* input_sizes = malloc (num_layers * sizeof (int));
	char* str; 
	char* pstr;
        char* str2;
        char* token;
        char* token2;
	struct Perceptron temp;

	rewind (fp);

	//printf ("num layers read = %d\n", num_layers);
	
	//first line is overall info, :::training algorithmn:::
	read = getline (&line, &len, fp);
	str = strdup (line);
	while (token = strsep (&str, ":::"))
	{
		if (*token == '\n')
			break;
		if (strlen (token) == 0)
			continue;
		PL.traina = string_to_traina (token);

	}
	
	// memory allocation
	while ((read = getline (&line, &len, fp)) != -1){
		//get number of perceptrons for current layer
		//should equal number of tabs plus 1

		int num_perceps = 1;
		int num_inputs = 0;
		str = strdup (line);
		pstr = strdup (str);

		
		while (token = strsep (&str, "\t")){
			if (*token == '\n')
				break;
			//printf ("token = %s\n", token);
			str2 = strdup (token);
			
			num_inputs = 0;
			num_perceps++;
			while (token2 = strsep (&str2, ",")){
				num_inputs++;
			}
			num_inputs -= 2;
			if (num_inputs < 1)
				num_inputs = 1; 
			input_sizes[curr_layer] = num_inputs;
			if (str2)
				free (str2);
		}
		PL.layer_sizes[curr_layer] = num_perceps - 1;
		PL.pl[curr_layer] = (struct Perceptron*) malloc (num_perceps 
				* sizeof (struct Perceptron));


		for (int i = 0; i < num_perceps; i++){
			PL.pl[curr_layer][i].inputs = malloc 
				(input_sizes[curr_layer] 
					* sizeof (struct Perceptron_Input));
		}
		
		// setting junk
		curr_percep = 0;
		while (token = strsep (&pstr, "\t")){
			str2 = strdup (token);

			temp.inputs = malloc (input_sizes[curr_layer] 
					* sizeof (struct Perceptron_Input));
			temp.num_inputs = input_sizes[curr_layer];
			temp.output = 0;
			int ccount = 0;
			while (token2 = strsep (&str2, ",")){
				if (*token2 == '\n')
					break;

				//printf ("token2 = %s\n", token2);
				if (ccount == 0){//AFT
					temp.aft = string_to_aft (token2);
					ccount++;
					continue;
				}
				else if (ccount == 1){//bias
					temp.bias = atof (token2);
					ccount++;
					continue;
				}
				// weight
				PL.num_parameters++;
				temp.inputs[ccount - 2].weight = atof (token2); 
				
				temp.inputs[ccount - 2].input = 0;

				ccount++;
				
			}
			//printf ("curr layer = %d ... curr per = %d\n", 
			//		curr_layer, curr_percep); 
			
			PL.pl[curr_layer][curr_percep] = temp;

			if (str2)
				free (str2);
			curr_percep++;
		}

		curr_layer++;
		if (str)
			free (str);
		if (pstr)
			free (pstr);

	}

	
	// close file
	fclose (fp);

	// free memory
	if (line)
		free (line);
	if (str)
		free (str);
	if (pstr)
		free (pstr);
	if (str2)
		free (str2);
	if (token)
		free (token);
	if (token2)
		free (token2);


	return PL;
}

// generates output values for current values of inputs
// Perceptron.inputs.input are values set in layer 0, outputs are values set in
// last layer Perceptron.output
void
generate_outputs (struct Perceptron_Layer PL){
	
	for (int l = 0; l < PL.num_layers; l++){
		for (int s = 0; s < PL.layer_sizes[l]; s++){
			if (l > 0){ 
				//TODO: remove when refactor input/output
				// to use pointers instead
				for (int c = 0; c < PL.layer_sizes[l - 1]; c++){
					PL.pl[l][s].inputs[c].input = PL.pl[l - 1][c].output;
				}
			}
			PL.pl[l][s].output = calculate_activation_function (PL.pl[l][s]);	
			//printf ("output = %f\n", PL.pl[l][s].output);
		}
	}
}

// calculates sum squared error... I think
double
sum_squared_error (struct Perceptron_Layer PL, double* targets)
{
	double sum = 0;

	for (int i = 0; i < PL.layer_sizes[PL.num_layers -1]; i++){
		//printf ("diff o vs t = %f\n", outputs[i] - targets[i]);
		sum += (PL.pl[PL.num_layers - 1][i].output - targets[i]) * (PL.pl[PL.num_layers - 1][i].output - targets[i]);
	}



	//printf ("sum^2 error = %f\n", sum);

	return sum;
}

// targets size == PL output layer size
// comapres targets to output layer and adjusts weights and bias for each
// Perceptron. TODO: switches on PL training algorithmn, 
// currently gradient descent probably?
void
perform_training_pass (struct Perceptron_Layer PL, double* targets) {
	double* err_sums = malloc (sizeof (double));

	// for each layer starting at output layer
	for (int l = PL.num_layers - 1; l >= 0; l--){

		for (int p = 0; p < PL.layer_sizes[l]; p++){
			if (l == PL.num_layers - 1) 
			{ // output layer
				double dw = calculate_activation_function_derivative (PL.pl[l][p]);
				dw = dw * (PL.pl[l][p].output - targets[p]); 

				PL.pl[l][p].bias -= dw * PL.pl[l][p].bias * 
					PL.training_rate;

				for (int i = 0; i < PL.pl[l][p].num_inputs; i++)
				{
					//printf ("output.old weight = %f\n", PL.pl[l][p].inputs[i].weight);
					dw = dw * PL.pl[l][p].inputs[i].input;
					PL.pl[l][p].inputs[i].weight -= dw *
						PL.training_rate;
					//printf ("output.new weight = %f\n", PL.pl[l][p].inputs[i].weight);
				}
			}
			else
			{
				//printf ("use err_sums[%d] = %f\n", p, err_sums[p]);
				//printf ("l = %d ... p = %d out = %f\n", l, p, PL.pl[l][p].output);
				double d = err_sums[p] * calculate_activation_function_derivative (PL.pl[l][p]);
				PL.pl[l][p].bias -= d * PL.pl[l][p].bias * 
					PL.training_rate;
				for (int i = 0; i < PL.pl[l][p].num_inputs; i++)
				{
					//printf ("old weight = %f\n", PL.pl[l][p].inputs[i].weight);
					d = d * PL.pl[l][p].inputs[i].input;
					PL.pl[l][p].inputs[i].weight -= d *
						PL.training_rate;
					//printf ("new weight = %f\n", PL.pl[l][p].inputs[i].weight);
				}
			}

			if (l > 0) // no need to calculate error if input layer
			{
				
				err_sums = realloc (err_sums, PL.layer_sizes[l - 1] * sizeof (double)); // holds error calculation for each perceptron

				for (int j = 0; j < PL.pl[l][p].num_inputs; j++)
				{
				 err_sums[j] = 0;
  				 for (int z = 0; z < PL.layer_sizes[PL.num_layers - 1]; z++)
				 {
				  //printf ("output = %f  -  target = %f\n", PL.pl[PL.num_layers - 1][z].output, targets[z]);
				  
				  // sums each weight and bias multiplied by 
				  // error of each output and activation
				  // function derivative.  
				  // TODO: is this correct for each layer?
				  err_sums[j] += (PL.pl[PL.num_layers - 1][z].output - targets[z]) * calculate_activation_function_derivative (PL.pl[l][p]) * PL.pl[l][p].inputs[j].weight;
				  err_sums[j] += (PL.pl[PL.num_layers - 1][z].output - targets[z]) * calculate_activation_function_derivative (PL.pl[l][p]) * PL.pl[l][p].bias;
				 }
				 //printf ("layer[%d].err_sums[%d] = %f\n", l, j, err_sums[j]);
				}
			}
		}
	}

	if (err_sums)
		free (err_sums);

}

/*
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

	double targets[] = {0.394356548, 0.387986545};

	for (int i = 0; i < 3000; i++) {
		generate_outputs (pl2);
		perform_training_pass (pl2, targets);
	}
	print_perceptron_layer (pl2);

	printf ("target1 = %u\ntarget2 = %u\n", (unsigned int) (394356548), (unsigned int) (387986545));

	printf ("out1 = %u\nout2 = %u\n", (unsigned int) (pl2.pl[3][0].output * 10000000000), (unsigned int) (pl2.pl[3][1].output * 10000000000));




	write_perceptron_layer_to_file ("test.txt", pl2);
}

*/
