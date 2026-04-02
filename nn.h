#ifndef NN_H
#define NN_H

typedef struct {

    size_t rows;
    size_t cols;
    size_t stride;
    float *p11;

} Matrix;


Matrix allocate_matrix(size_t rows, size_t cols);

float rand_float();
float sigmoidf(float x);

void matrix_dot(Matrix result, Matrix a, Matrix b);
void matrix_sum(Matrix result, Matrix a);
void matrix_print(Matrix m, const char* name, size_t padding);
void matrix_rand(Matrix m, float low, float high);
void matrix_fill(Matrix m, float x);
void matrix_activation_sigmoid(Matrix m);
Matrix matrix_row(Matrix m, size_t row);
void matrix_copy(Matrix dst, Matrix src);


typedef struct {

    size_t layer_count; // number of layers

    Matrix *ws; // arrays of matrices
    Matrix *bs;
    Matrix *as; // the amount of activation + 1 (for input layer)

} NN;

NN nn_allocate(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float hight);
void nn_forward(NN nn);
float nn_cost_function(NN nn, Matrix train_input, Matrix train_output);
void nn_finite_diff(NN m, NN gradient, float h, Matrix train_input, Matrix train_output);
void nn_learn(NN nn, NN gradient,float learning_rate);

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).layer_count]



#define MATRIX_AT(m, row, col) (m).p11[(row) * (m).stride + (col)]
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

#endif // NN_H

// here for implemnation
#ifdef NN_IMPLEMENTATION


Matrix allocate_matrix(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.p11 = malloc(sizeof(*m.p11) * rows * cols);
    assert(m.p11 != NULL);
    
    return m;
}

float sigmoidf(float x)
{
    return 1.f / (1+ exp(-x));
}

void matrix_dot(Matrix result, Matrix a, Matrix b)
{
    // 1x2  2x3
    assert(a.cols == b.rows);
    size_t n = a.cols;

    for (size_t i=0; i < result.rows ; ++i)
    {
        for (size_t j=0; j < result.cols; ++j)
        {
            MATRIX_AT(result, i, j) = 0.f;

            for (size_t k=0; k < n; ++k)
            {
                // i (k   k) j
                MATRIX_AT(result, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
        }
    }

    assert(result.cols == b.cols);
    assert(result.rows == a.rows);
}

void matrix_sum(Matrix result, Matrix a)
{

    assert(result.rows == a.rows);
    assert(result.cols == a.cols);

    for (size_t i=0; i < result.rows ; ++i)
    {
        for (size_t j=0; j < result.cols; ++j)
        {
            MATRIX_AT(result, i, j) += MATRIX_AT(a, i, j);
        }
    }
}

void matrix_print(Matrix m, const char* name, size_t padding)
{

    printf("%*s%s = [\n",(int) padding, "", name);
    
    for (size_t i=0; i < m.rows; ++i)
    {

        printf("%*s     ",(int) padding,""); 
                                                                   
        for(size_t j=0; j < m.cols; ++j)
        {
            printf("%f ", MATRIX_AT(m, i, j)); // the size of the row is the number of cols
        }

        printf("\n");
    }

    printf("%*s]\n", (int) padding, "");
}

float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}

void matrix_rand(Matrix m, float low, float high)
{

   for (size_t i=0; i < m.rows; ++i)
   {
        for(size_t j=0; j < m.cols; ++j)
        {
           MATRIX_AT(m, i, j) = rand_float() * (high - low) + low;
        }
   }
}


void matrix_fill(Matrix m, float x)
{

   for (size_t i=0; i < m.rows; ++i)
   {
        for(size_t j=0; j < m.cols; ++j)
        {
           MATRIX_AT(m, i, j) = x;
        }
   }
}

void matrix_activation_sigmoid(Matrix m)
{
    for (size_t i=0; i < m.rows; ++i) 
    {
        for (size_t j=0; j < m.cols; ++j)
        {
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}


Matrix matrix_row(Matrix m, size_t row)
{
    return (Matrix) {

        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .p11 = &MATRIX_AT(m, row, 0)

    };
}

void matrix_copy(Matrix dst, Matrix src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i=0; i < dst.rows; ++i)
    {
        for (size_t j=0; j < dst.cols; ++j)
        {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
}

// size_t arch[] = {2,2,1};
// NN nn = nn_allocate(arch, ARRAY_LEN(arch));

NN nn_allocate(size_t *arch, size_t arch_count)
{
    assert(arch_count > 0);

    NN nn;
    nn.layer_count = arch_count - 1; // the amount of inner neuron without inputs
                                     
    // allocate enough array of matrices 
    nn.ws = malloc(sizeof(*nn.ws) * nn.layer_count); 
    assert(nn.ws != NULL);

    nn.bs = malloc(sizeof(*nn.bs) * nn.layer_count);
    assert(nn.bs != NULL);

    nn.as = malloc(sizeof(*nn.as) * (nn.layer_count + 1));
    assert(nn.as != NULL);

    nn.as[0] = allocate_matrix(1, arch[0]);

    for (size_t i=1; i<arch_count; ++i)
    {
        nn.ws[i-1] = allocate_matrix(nn.as[i - 1].cols, arch[i]);
        nn.bs[i-1] = allocate_matrix(1, arch[i]);
        nn.as[i] = allocate_matrix(1, arch[i]);
    }

    return nn;
}


void nn_print(NN nn, const char *name)
{
    char buff[256];
    printf("%s = [\n", name);

    for (size_t i=0; i<nn.layer_count; ++i)
    {
        snprintf(buff, sizeof(buff), "ws%zu", i);
        matrix_print(nn.ws[i], buff, 4);

        snprintf(buff, sizeof(buff), "bs%zu", i);
        matrix_print(nn.bs[i], buff, 4);
    }

    printf("]\n");
}


void nn_rand(NN nn, float low, float hight)
{
    for (size_t i=0; i < nn.layer_count; ++i)
    {
        matrix_rand(nn.ws[i], low, hight);
        matrix_rand(nn.bs[i], low, hight);
    }
}


void nn_forward(NN nn)
{
   for (size_t i=0; i < nn.layer_count; ++i)
   {
       matrix_dot(nn.as[i+1] ,nn.as[i], nn.ws[i]);
       matrix_sum(nn.as[i+1], nn.bs[i]);
       matrix_activation_sigmoid(nn.as[i+1]);
   }
}

float nn_cost_function(NN nn, Matrix train_input, Matrix train_output)
{
    assert(train_input.rows == train_output.rows);
    assert(train_output.cols == NN_OUTPUT(nn).cols);
    size_t n = train_input.rows;

    float cost = 0;

    for (size_t i=0; i < n; ++i)
    {
        Matrix x = matrix_row(train_input, i); // expected input
        Matrix y = matrix_row(train_output, i); // expected output

        matrix_copy(NN_INPUT(nn), x);
        nn_forward(nn); // for get the prediction
        
        size_t k = train_output.cols;

        for (size_t j=0; j < k; ++j)
        {
            float distance = MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
            cost += distance * distance;
        }
    }

    return cost / n; 
}

void nn_finite_diff(NN nn, NN gradient, float h, Matrix train_input, Matrix train_output)
{
   float saved = 0;
   float cost = nn_cost_function(nn, train_input, train_output);

   for (size_t layer=0; layer<nn.layer_count; ++layer)
   {
        for (size_t j=0; j<train_input.rows; ++j)
        {
            for (size_t k=0; k<train_input.cols; ++k)
            {
               saved = MATRIX_AT(nn.ws[layer], j, k); 

               MATRIX_AT(nn.ws[layer], j, k) += h;
               MATRIX_AT(gradient.ws[layer], j, k) = (nn_cost_function(nn, train_input, train_output) - cost) / h; 

               MATRIX_AT(nn.ws[layer], j, k) = saved;
            }
        }


        for (size_t j=0; j<train_input.rows; ++j)
        {
            for (size_t k=0; k<train_input.cols; ++k)
            {
               saved = MATRIX_AT(nn.bs[layer], j, k); 

               MATRIX_AT(nn.bs[layer], j, k) += h;
               MATRIX_AT(gradient.bs[layer], j, k) = (nn_cost_function(nn, train_input, train_output) - cost) / h; 

               MATRIX_AT(nn.bs[layer], j, k) = saved;
            }
        }
   }
}


void nn_learn(NN nn, NN gradient,float learning_rate)
{

   for (size_t layer=0; layer<nn.layer_count; ++layer)
   {
        for (size_t j=0; j<nn.ws[layer].rows; ++j)
        {
            for (size_t k=0; k<nn.ws[layer].cols; ++k)
            {
               MATRIX_AT(nn.ws[layer], j, k) -= learning_rate * MATRIX_AT(gradient.ws[layer], j, k); 
            }
        }


        for (size_t j=0; j<nn.bs[layer].rows; ++j)
        {
            for (size_t k=0; k<nn.bs[layer].cols; ++k)
            {
               MATRIX_AT(nn.bs[layer], j, k) -= learning_rate * MATRIX_AT(gradient.bs[layer], j, k); 
            }
        }
   }
}



#endif // NN_IMPLEMENTATION
