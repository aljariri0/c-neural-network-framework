#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>


#define NN_IMPLEMENTATION
#include "nn.h"

float td_xor[] = {
    
    0,0,0,
    1,0,1,
    0,1,1,
    1,1,0,
};


float td_or[] = {
    
    0,0,0,
    1,0,1,
    0,1,1,
    1,1,1,
};


float td_and[] = {
    
    0,0,0,
    1,0,0,
    0,1,0,
    1,1,1,
};


int main(void)
{
    // srand(time(0));
    srand(42);
  

    float *td = td_or;

    size_t stride = 3;
    size_t n = 4; // sizeof(*td) / sizeof(td[0]) / stride; // number of rows
    
    Matrix train_input = {

        .rows = n,
        .cols = 2,
        .stride = stride,
        .p11 = td
    };

    Matrix train_output = {

        .rows = n,
        .cols = 1,
        .stride = stride,
        .p11 = &td[2]
    };   

    size_t arch[] = {2,2,1};
    float h = 1e-2;
    float learning_rate = 1e-1;

    NN nn = nn_allocate(arch, ARRAY_LEN(arch));
    NN gradient = nn_allocate(arch, ARRAY_LEN(arch));

    nn_rand(nn, 0, 1);

    for (size_t i=0; i < 100 * 1000; ++i)
    {
        printf("Cost = %f\n", nn_cost_function(nn, train_input, train_output));
        nn_finite_diff(nn, gradient, h, train_input, train_output);
        nn_learn(nn, gradient, learning_rate);
    }


    for (size_t i=0; i<2; ++i)
    {
        for (size_t j=0; j<2; ++j)
        {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);

            printf("%zu ^ %zu = %f\n", i, j, MATRIX_AT(NN_OUTPUT(nn), 0, 0));
        }
    }


   // NN_PRINT(nn);

    //free(&nn);
    //free(&gradient);

    return 0;
}
