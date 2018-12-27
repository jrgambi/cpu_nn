# source, executable, includes, libray definitions
INCL	= my_neural_net.h my_cpu.h
SRC	= cpu_nn.c my_neural_net.c my_cpu.c
OBJ	= $(SRC:.c=.o)
LIBS	= -lm
EXE	= cpu_nn
TESTEXE	= test
TESTSRC	= test.c

# compiler, linker definitions
CC	= /usr/bin/gcc
#CFLAGS	= 
LIBPATH	= -L
LDFLAGS	= -o $(EXE) $(LIBPATH) $(LIBS)
CFDEBUG	= -g -DDEBUG
RM	= /bin/rm -f

#compile and assemble
%.o: %.c
	$(CC) -c $*.c $(LIBS)

# Link all objs with external libs
$(EXE): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) $(LIBS)

# objects depend on these libs
$(OBJ): $(INCL)

# debug flags
debug:
	$(CC) $(CFDEBUG) $(SRC)

# compile tests
test:
	$(CC) -o my_neural_net.o -c my_neural_net.c -lm
	$(CC) -o $(TESTEXE) $(TESTSRC) my_neural_net.o $(LIBS) 

#clean up
clean:
	$(RM) $(OBJ) $(EXE) $(TESTEXE) core a.out
