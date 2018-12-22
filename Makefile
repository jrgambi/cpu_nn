# source, executable, includes, libray definitions
INCL	= my_cpu.h my_neural_net.h
SRC	= cpu_nn.c my_cpu.c my_neural_net.c
OBJ	= $(SRC:.c=.o)
LIBS	= -lm
EXE	= cpu_nn

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

#clean up
clean:
	$(RM) $(OBJ) $(EXE) core a.out
