#define _XOPEN_SOURCE 600
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <pthread.h>

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

struct thread_data
{
    int *data_chunk;
    int len;
    int result;
    pthread_barrier_t *barrier;
    int (*op)(int, int);
};

void *thread_func(void *arg)
{
    struct thread_data *thread_data = (struct thread_data *)arg;
    int i;
    thread_data->result = thread_data->data_chunk[0];
    for (i = 1; i < thread_data->len; ++i)
        thread_data->result = thread_data->op(thread_data->result, thread_data->data_chunk[i]);

    pthread_barrier_wait(thread_data->barrier);

    return NULL;
};

int parallel_reduce(int (*op)(int, int),
                    int *data,
                    int len)
{
    long system_threads = get_number_of_processors();

    printf("Number of processors: %li\n", system_threads);
    printf("Creating %li threads\n", system_threads);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, system_threads);

    int result = 0;

    pthread_t threads[system_threads];
    struct thread_data thread_data_array[system_threads];

    int chunk_size = len / system_threads;
    int remainder = len % system_threads;

    printf("Chunk size: %i\n", chunk_size);

    // init thread_data
    for (int i = 0; i < system_threads - 1; i++)
    {
        thread_data_array[i].data_chunk = data + i * chunk_size;
        thread_data_array[i].len = chunk_size;
        thread_data_array[i].op = op;
        thread_data_array[i].barrier = &barrier;
    }
    // last thread gets the remainder
    thread_data_array[system_threads - 1].data_chunk = data + (system_threads - 1) * chunk_size;
    thread_data_array[system_threads - 1].len = chunk_size + remainder;
    thread_data_array[system_threads - 1].op = op;
    thread_data_array[system_threads - 1].barrier = &barrier;

    printf("Distributing data:\n");
    for (int i = 0; i < system_threads; i++)
    {
        printf("Thread %i: ", i);
        struct thread_data thread_data = thread_data_array[i];
        for (int j = 0; j < thread_data.len; j++)
        {
            printf("%i ", thread_data.data_chunk[j]);
        }
        printf("\n");
    }

    // so lets start the threads for one round
    for (int i = 0; i < system_threads; i++)
    {
        struct thread_data *thread_data = &thread_data_array[i];
        pthread_create(&threads[i], NULL, thread_func, (void *)thread_data);
    }
    // join threads
    for (int i = 0; i < system_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // just see the results
    printf("Results:\n");
    for (int i = 0; i < system_threads; i++)
    {
        printf("Thread %i: %i\n", i, thread_data_array[i].result);
    }
}

int main()
{
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // int m = reduce(max, data, 10);
    // int s = reduce(sum, data, 10);

    // printf("max : %i; sum: %i\n", m, s);

    // int pm = parallel_reduce(max, data, 10);
    int ps = parallel_reduce(sum, data, 10);

    // printf("parallel max : %i; parallel sum: %i\n", pm, ps);
    return 0;
}

// gcc -o main -lpthread main.c && ./main
