#define _XOPEN_SOURCE 600 // required for barriers with -std=c99

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *parallel_reduce_inner(void *arg);

int parallel_reduce(int (*op)(int, int),
                    int *data,
                    int len);

int max(int a, int b)
{
    return (a > b) ? a : b;
}

int sum(int a, int b)
{
    return a + b;
}

int product(int a, int b)
{
    return a * b;
}

int reduce(int (*op)(int, int),
           int *data,
           int len)
{
    int i;
    int result = data[0];
    for (i = 1; i < len; ++i)
        result = op(result, data[i]);
    return result;
}

int find_max_exp_2(int n)
{
    int exp = 1;
    while (exp <= n)
    {
        exp *= 2;
    }
    return exp / 2;
}

struct kernel_arg
{
    int tid;
    int number_of_threads;
    int (*op)(int, int);
    int *data;
    int len;
    pthread_barrier_t *barrier;
};

void *kernel(void *arg)
{
    struct kernel_arg *kernel_arg = (struct kernel_arg *)arg;

    const int tid = kernel_arg->tid;
    int step_size = 1;
    int number_of_threads = kernel_arg->number_of_threads;

    while (number_of_threads > 0)
    {

        if (tid < number_of_threads)
        {
            const int fst = tid * step_size * 2;
            const int snd = fst + step_size;
            kernel_arg->data[fst] = kernel_arg->op(kernel_arg->data[fst], kernel_arg->data[snd]);
        }

        step_size = step_size << 1;
        number_of_threads = number_of_threads >> 1;

        pthread_barrier_wait(kernel_arg->barrier);
    }

    return NULL;
}

struct parallel_reduce_inner_arg
{
    int (*op)(int, int);
    int *data;
    int len;
    int result;
};

void *parallel_reduce_inner(void *arg)
{
    struct parallel_reduce_inner_arg *inner_arg = (struct parallel_reduce_inner_arg *)arg;
    int result = parallel_reduce(inner_arg->op, inner_arg->data, inner_arg->len);
    inner_arg->result = result;

    return NULL;
}

// Not Optimal
int parallel_reduce(int (*op)(int, int),
                    int *data,
                    int len)
{
    int is_exp_2 = (len & (len - 1)) == 0;

    if (!is_exp_2)
    {
        int max_exp_2 = find_max_exp_2(len);

        // int result_1 = parallel_reduce(op, data, max_exp_2);
        // int result_2 = parallel_reduce(op, data + max_exp_2, len - max_exp_2);

        struct parallel_reduce_inner_arg inner_arg_1 = {op, data, max_exp_2, 0};
        struct parallel_reduce_inner_arg inner_arg_2 = {op, data + max_exp_2, len - max_exp_2, 0};

        pthread_t thread_1;
        pthread_t thread_2;

        pthread_create(&thread_1, NULL, parallel_reduce_inner, (void *)(&inner_arg_1));
        pthread_create(&thread_2, NULL, parallel_reduce_inner, (void *)(&inner_arg_2));
        pthread_join(thread_1, NULL);
        pthread_join(thread_2, NULL);

        int result_1 = inner_arg_1.result;
        int result_2 = inner_arg_2.result;

        return op(result_1, result_2);
    }

    const int number_of_threads = len / 2;
    if (number_of_threads == 0)
    {
        return data[0];
    }

    // pthread_t threads[number_of_threads];
    // struct kernel_arg kernel_arg_array[number_of_threads];

    pthread_t *threads = malloc(sizeof(pthread_t) * number_of_threads);
    struct kernel_arg *kernel_arg_array = malloc(sizeof(struct kernel_arg) * number_of_threads);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, number_of_threads);

    for (int i = 0; i < number_of_threads; i++)
    {
        kernel_arg_array[i].tid = i;
        kernel_arg_array[i].number_of_threads = number_of_threads;
        kernel_arg_array[i].op = op;
        kernel_arg_array[i].data = data;
        kernel_arg_array[i].len = len;
        kernel_arg_array[i].barrier = &barrier;

        pthread_create(&threads[i], NULL, kernel, (void *)(kernel_arg_array + i));
    }

    for (int i = 0; i < number_of_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    free(kernel_arg_array);
    free(threads);

    return data[0];
}

int main()
{
    // Since our reduce function updates in place, we need to copy the data
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int len = 10;

    int data_2[len];
    int data_3[len];
    for (int i = 0; i < len; i++)
    {
        data_2[i] = data[i];
        data_3[i] = data[i];
    }

    int seq_sum = reduce(sum, data, len);

    int par_sum = parallel_reduce(sum, data, len);

    printf("seq sum: %i; par sum: %i\n", seq_sum, par_sum);

    int seq_max = reduce(max, data_2, len);
    int par_max = parallel_reduce(max, data_2, len);

    printf("seq max: %i; par max: %i\n", seq_max, par_max);

    int seq_product = reduce(product, data_3, len);
    int par_product = parallel_reduce(product, data_3, len);

    printf("seq product: %i; par product: %i\n", seq_product, par_product);

    return 0;
}

// gcc -o main -lpthread main.c && ./main
