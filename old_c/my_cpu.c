#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>

#include "my_cpu.h"
#include "my_neural_net.h"

// ARMv8-a implementation

// TODO: wrapper handles register cleanup?
// wrapper forks and executes function defined at func_ptr.  
// setting registers to values starting at (unsigned int*) input
// storing resulting register values at memory location (unsigned int*) output
// FIXME: condition could probably be defaulted to a value in input so removed as parameter
void wrapper_function (void *func_ptr, unsigned int input, unsigned int output, unsigned int condition); 

// generates number of instructions, store in content. 
// attempts to generate somewhat valid and randomish instructions
// can add further constraints here
void generate_instructions (unsigned int* content, int number);

// forms surrounding commands for setup of inputs and writing outputs
// around vcontent and stores in final_invocation
void create_invocation(unsigned int* vcontent, int size_vcontent, unsigned int* final_invocation);

// random generation of input values
void generate_register_inputs (unsigned int* inputs); 

/*
int
main (int argc, char *argv[])
{	
	time_t t;
	srandom ((unsigned) time(&t));
	//TODO: tests here for now?
	//content = 0xE2800001; // R0 ++:  ADD R0, R0, #1

}
*/

void wrapper_function (void *func_ptr, unsigned int input, unsigned int output, unsigned int condition) {

	int my_func_return;
	pid_t parent = getpid();
	//printf ("function_ptr = %x\noutput_ptr = %x\ninput_ptr=%x\n", &func_ptr, output, input);

	pid_t child = fork();
	if (child == 0)
	{
		// set r0 to address of input
		register int *r0 asm ("r0") = input;
		// set r1 to address of output
		register int *r1 asm ("r1") = output;

		unsigned int in;
		unsigned int out;

		void (*foo)(void) = func_ptr;

		// sets apsr conditional byte
		asm volatile ("mov %r0, %0\n\t"
				"msr apsr_nzcvq, %r0\n\t"
				: //no output
				: "r" (condition)
				:);

		asm volatile ( "mov %r0, %1\n\t" // input r0
			       "mov %r1, %2\n\t" // output r1
				: "=r" (in)//, "=rm"
				: "r" (input), "r" (output)
				: "memory", "cc");
		
		
		foo();

		// reads resulting apsr condition byte
		asm volatile ("mrs %0, apsr\n\t"
				: "=r" (out)
				: "0" (out)
				:);

		//printf ("apsr out = %x\n", out);
		unsigned int* temp = (unsigned int*) output;
		// conditional stored
		temp [OUTPUTS_SIZE - 2] = out;
		
		// error value.  error if != 0
		temp [OUTPUTS_SIZE - 1] = 0;
		
		exit (EXIT_SUCCESS);
	}
	
	if (getpid() != parent)
	{
		exit (EXIT_SUCCESS);
	}

	int timeout = 0;

	// FIXME: sometimes hangs,
	// this can catch some instances and fixes it
	while (waitpid (-1, &my_func_return, WNOHANG) == 0)
	{
		if (timeout > 3 || timeout < 0)
		{
			printf ("\n\n\n\n\nchild hung...\n\n\n\n\n");
			kill (child, SIGKILL);
			break;
		}
		sleep (0.1);
		timeout += 0.1;
	}

}

void
generate_instructions (unsigned int* content, int number) // TODO: add usable memory range parameters and setup instructions for accessing it, also register input seed 
{
	
	int count = 0;

	// seed for register destination values R0,R1,R3-R10
	unsigned int rd_seed_array[] = {0x00000000, 0x00001000, 0x00003000, 0x00004000, 0x00005000, 0x00006000, 0x00007000, 0x00008000, 0x00009000, 0x0000A000};

	while (count < number){

		unsigned int base_instruction = random();
		unsigned int base_rd = rd_seed_array [random() % 10];
		int instruct_arch_type = random() % 2; // TODO: add more instructions later
		//printf ("base = %x\n", base_instruction);
		switch (instruct_arch_type) {
			
			case 1 : // Multiply : xxxx000000xxxxxxxxxxxxxx1001xxxx
			  base_instruction = base_instruction | 0x00000090;
			  base_instruction = base_instruction & (~0x0FC00060);

			  // use base_rd to force valid destination
			  // MUL needs a shift left 1 byte
			  base_instruction = (base_instruction & (~0x000F0000)) 
				  | (base_rd << 8);

			  break;
			
			default : // Data Processing : xxxx00xx->x
			  base_instruction = base_instruction & (~0x0C000000);

			  // use base_rd to force valid destination
			  base_instruction = (base_instruction & (~0x0000F000))
				  | base_rd;

			  break;
			  
		} 

		//printf ("instruction = %x\n", base_instruction);
		content[count] = base_instruction;
		count++;
	}
}

void
generate_register_inputs (unsigned int* inputs)
{

	for (int i = 0; i < INPUTS_SIZE; i++)
	{
		if (i == INPUTS_SIZE - 1) // instruction
		{
			inputs[i] = 0x00000000; // generated elsewhere
			break;
		}
		if (i == INPUTS_SIZE - 2) // condition
		{

			inputs[i] = random() & 0xF0000000;
			continue;
		}
		inputs[i] = random();
		//printf ("input[%d] = %x\n", i, inputs[i]);
	}
}

// WARNING: FIXME: commenting out of date
// lots of crazy, modifiy at your own risk.   
void
create_invocation(unsigned int* vcontent, int size_vcontent, unsigned int* final_invocation)
{

	static unsigned int no_op =  0xE1A00000;

	//0xE89007FB, //LDM R0, {R0-R1,R3-R10} 
	//// FIXME: ^ should work, but needs testing

	// Load input values starting at R0
	unsigned int prefix_invoc[] = {
		0xE5901004,// LDR R1, [R0, #4]
		0xE590300C,// LDR R3, [R0, #12]
		0xE5904010,// LDR R4, [R0, #16]
		0xE5905014,// LDR R5, [R0, #20]
		0xE5906018,// LDR R6, [R0, #24]
		0xE590701C,// LDR R7, [R0, #28]
		0xE5908020,// LDR R8, [R0, #32]
		0xE5909024,// LDR R9, [R0, #36]
		0xE590A028,// LDR R10, [R0, #40]
		0xE5900000,// LDR R0, [R0]
		no_op, no_op, no_op, no_op, no_op, no_op, no_op, no_op, no_op, no_op, no_op
	};
	
	// store registers into location at R2 
	//FIXME: R2 is hacky, and was set elsewhere. (Function call??)
	unsigned int suffix_invoc[] = { 
		0xE882FFFF, // STM R2, {R0-R15}
		no_op, no_op, no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op,no_op, no_op, no_op, no_op, no_op, //ILL
		0xE12FFF1E}; // BX LR // return


// conditional test
		//0xE5810000, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op,		
		//0x45813000, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op, no_op,no_op,no_op,// str r0, [r1] 
		// end of conditional test


	//printf ("size of invocation = %d\n", sizeof (prefix_invoc) + sizeof (suffix_invoc));


	memcpy (final_invocation, prefix_invoc, sizeof (prefix_invoc));

	memcpy (final_invocation + (sizeof (prefix_invoc) / sizeof (unsigned int)), vcontent, sizeof (size_vcontent));
	
	memcpy (final_invocation + ((sizeof (prefix_invoc) / sizeof (unsigned int)) + size_vcontent), suffix_invoc, sizeof (suffix_invoc));

}

