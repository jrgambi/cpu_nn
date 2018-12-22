#ifndef MYREGISTERS_H_
#define MYREGISTERS_H_

#define INPUTS_SIZE 12
#define OUTPUTS_SIZE 22

void wrapper_function (void *func_ptr, unsigned int input, unsigned int output, unsigned int condition); 

// generates number of instructions, store in content
void generate_instructions (unsigned int* content, int number);

void create_invocation(unsigned int* vcontent, int size_vcontent, unsigned int* final_invocation);

void generate_register_inputs (unsigned int* inputs); 

#endif // MYREGISTERS_H_
