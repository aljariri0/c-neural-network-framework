C Neural Network Framework

A lightweight dynamic Neural Network framework completely built from scratch in pure C (uses no ML libraries such as PyTorch or TensorFlow-only standard C libraries and math functions).

The main features of the framework:

* Dynamic architecture: you create the structure of your Neural Network by declaring its dimensions with a size_t array, e.g:
  c
  arch[] = {2, 2, 1}; // 2 inputs nodes, 2 hidden nodes, 1 output node
  
* Custom Matrix Library: completely built from scratch with simple operations: allocate matrices, add two matrices, calculate the dot product of two matrices and initialize matrices with random weights.

* Forward Propagation: passes input data through the network with Sigmoid as the activation function.

* Training / Learning: calculate the gradient and Minimize a Mean Squared Error cost function, adjusting weights and biases.

* Logic Gate Solvers: in main.c you will find an implementation where the trained neural network learns to solve the OR, AND and XOR logic gates.

Building and Running (Linux):
you can use build.sh file or (`gcc -o main main.c -ml`)
We use math.h, therefore we have to link it with: -lm.


