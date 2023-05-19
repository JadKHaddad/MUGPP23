#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 10

struct barrier_t
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int max;
};

void barrier_init(struct barrier_t *barrier, int max)
{
    barrier->count = 0;
    barrier->max = max;
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
}

void barrier_wait(struct barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    barrier->count++;
    if (barrier->count == barrier->max)
    {
        pthread_cond_broadcast(&barrier->cond);
    }
    else
    {
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
    }
    pthread_mutex_unlock(&barrier->mutex);
}

void barrier_destroy(struct barrier_t *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
}

int get_rand_sleep_time()
{
    return rand() % 100 + 750000;
}

struct barrier_t barrier;

void *thread_func(void *arg)
{
    int id = *((int *)arg);
    int random_iterations = rand() % 20 + 1;

    printf("Thread %d: I will iterate %d times\n", id, random_iterations);

    for (int i = 0; i < random_iterations; i++)
    {
        printf("Thread %d: %d\n", id, i);
        usleep(get_rand_sleep_time());
    }
    printf("Thread %d: I am waiting\n", id);

    barrier_wait(&barrier);

    printf("Thread %d: We are done!\n", id);

    return NULL;
}

int main(int argc, char **argv)
{
    barrier_init(&barrier, N);

    pthread_t threads[N];
    int ids[N];

    for (int i = 0; i < N; ++i)
    {
        pthread_t thread;
        // store the id it will be lost otherwise. I love C.
        ids[i] = i;
        pthread_create(&thread, NULL, thread_func, ((void *)(ids + i)));
        threads[i] = thread;
    }

    for (int i = 0; i < N; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
