#define _XOPEN_SOURCE 600
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

long get_number_of_processors()
{
    long nprocs = -1;
#ifdef _WIN32
#ifndef _SC_NPROCESSORS_ONLN
    SYSTEM_INFO info;
    GetSystemInfo(&info);
#define sysconf(a) info.dwNumberOfProcessors
#define _SC_NPROCESSORS_ONLN
#endif
#endif
#ifdef _SC_NPROCESSORS_ONLN
    nprocs = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return nprocs;
}

int max(int a, int b)
{
    return (a > b) ? a : b;
}

int sum(int a, int b)
{
    return a + b;
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

void print_data(int *data, int len)
{
    printf("Data: ");
    for (int i = 0; i < len; i++)
    {
        printf("%i ", data[i]);
    }
    printf("\n");
}

struct thread_data
{
    int id;
    int *data;
    int len;
    int start_index;
    int chunk_size;
    int (*op)(int, int);
    pthread_barrier_t *barrier;
};

void *thread_func(void *arg)
{
    struct thread_data *thread_data = (struct thread_data *)arg;

    // this is the first round of the algorithm where each thread reduces its chunk
    for (int i = thread_data->start_index + 1; i < thread_data->chunk_size + thread_data->start_index; ++i)
    {
        thread_data->data[thread_data->start_index] = thread_data->op(thread_data->data[thread_data->start_index], thread_data->data[i]);
    }

    pthread_barrier_wait(thread_data->barrier);

    // in the second round we go through a loop cutting the count of threads in half each time depending on the thread id
    // and each thread reduces its element with other element, for which no other thread is responsible
    // this is done until only one thread is left

    int count = thread_data->chunk_size;
    int offset = 1;
    while (count > 1)
    {
        if (thread_data->id % (2 * offset) == 0)
        {
            thread_data->data[thread_data->start_index] = thread_data->op(thread_data->data[thread_data->start_index], thread_data->data[thread_data->start_index + offset]);
        }
        count /= 2;
        offset *= 2;
        // pthread_barrier_wait(thread_data->barrier);
    }

    return NULL;
};

struct kernel_arg
{
    int id;
    int (*op)(int, int);
    int *data;
    int len;
    int start_index;
    int end_index;
    int total_threads;
    pthread_barrier_t *barrier;
};

void *kernel(void *arg)
{   
    struct kernel_arg *kernel_args = (struct kernel_arg *)arg;

    int id = kernel_args->id;
    int start_index = kernel_args->start_index;
    int end_index = kernel_args->end_index;
    int result = kernel_args->data[start_index];

    printf("Thread %i started\n", id);

    printf("Thread %i chunk: %i - %i | from [%i] to [%i]\n", id, start_index, end_index, kernel_args->data[start_index], kernel_args->data[end_index]);
    
    // round 1
    for (int i = start_index + 1; i < end_index + 1; ++i)
    {
        result = kernel_args->op(result, kernel_args->data[i]);
    }

    printf("Thread %i reduced own chunk: %i\n", id, result);
    pthread_barrier_wait(kernel_args->barrier);

    // at this point the amount of data to be reduced is halved and matches the number of threads
    // so lets rerrange it. every thread will copy its data to the first half of the array
    // and the second half will be filled just ignored
    printf("Thread %i rearranging data from [%i] to [%i]\n", id, start_index, id);
    kernel_args->data[id] = result;
    pthread_barrier_wait(kernel_args->barrier);
    
    // round 2

	int step_size = 1;
	int n_threads = kernel_args->len / 2;

	while (n_threads > 0)
	{
		if (id < n_threads) // still alive?
		{
			int fst = id * step_size * 2;
			int snd = fst + step_size;

            printf("Thread %i reducing [%i] and [%i]\n", id, fst, snd);

            kernel_args->data[fst] = kernel_args->op(kernel_args->data[fst], kernel_args->data[snd]);
		}

        pthread_barrier_wait(kernel_args->barrier);
		step_size <<= 1; 
		n_threads >>= 1;
	}

    return NULL;
}

int parallel_reduce(int (*op)(int, int),
                    int *data,
                    int len)
{   
    long system_threads = get_number_of_processors();

    printf("Number of processors: %li\n", system_threads);
    if (system_threads < 2)
    {
        if (system_threads == -1)
        {
            printf("Could not determine number of processors\n");
        }
        printf("Not enough processors available, using sequential reduce\n");
        return reduce(op, data, len);
    }

    // since we are updating in place we need to copy the data
    int *data_copy = malloc(len * sizeof(int));
    for (int i = 0; i < len; i++)
    {
        data_copy[i] = data[i];
    }
    
    printf("Setting up %li threads\n", system_threads);
    if (len < system_threads)
    {
        printf("Data size is smaller than number of processors\n");
        system_threads = len;
    }

    int chunk_size = len / system_threads;
    printf("Chunk size: %i\n", chunk_size);

    pthread_t threads[system_threads];
    struct kernel_arg kernel_arg_array[system_threads];

    printf("Calculating number of spawnable threads\n");

    int total_threads = 0;
    for (int i = 0; i < system_threads; i++)
    {   
        total_threads += 1;

        kernel_arg_array[i].id = i;
        kernel_arg_array[i].op = op;
        kernel_arg_array[i].data = data_copy;
        kernel_arg_array[i].len = len;

        int start_index = i * chunk_size * 2;
        int end_index = start_index + chunk_size;

        kernel_arg_array[i].start_index = start_index;
        kernel_arg_array[i].end_index = end_index;

        int start_index_next_thread = (i + 1) * chunk_size * 2;
        int end_index_next_thread = start_index_next_thread + chunk_size;

        // if end_index_next_thread is out of bounds, we include the rest of the data in the chunk
        if (end_index_next_thread >= len)
        {
            kernel_arg_array[i].end_index = len -1;
            break;
        }
    }

    printf("Spawnable threads: %i\n", total_threads);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, total_threads);

    for (int i = 0; i < total_threads; i++)
    {
        kernel_arg_array[i].total_threads = total_threads;
        kernel_arg_array[i].barrier = &barrier;
        pthread_create(&threads[i], NULL, kernel, (void *)(kernel_arg_array + i));
    }

    for (int i = 0; i < total_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    print_data(data_copy, len);
    
    int result = data_copy[0];
    pthread_barrier_destroy(&barrier);
    free(data_copy);

    return result;
}

int main()
{
    int data[] = {1, 2, 3, 4, 5,  6 , 7, 8, 9, 10};

    //int m = reduce(max, data, 10);
    //int s = reduce(sum, data, 10);

    //printf("max : %i; sum: %i\n", m, s);

    int pm = parallel_reduce(max, data, 10);
    int ps = parallel_reduce(sum, data, 10);

    printf("parallel max : %i; parallel sum: %i\n", pm, ps);
    return 0;
}

// gcc -o main -lpthread main.c && ./main
