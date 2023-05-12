#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define UNUSED(expr)  \
    do                \
    {                 \
        (void)(expr); \
    } while (0)

#define MAX_ITER (100000)

struct Buffer
{
    char *buf;
    int in;
    int out;
    int count;
    int size;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

void put(struct Buffer *b, char c)
{
    pthread_mutex_lock(&b->mutex);
    printf("put\n");
    while (b->count == b->size)
        pthread_cond_wait(&b->cond, &b->mutex);
    b->buf[b->in] = c;
    b->in = (b->in + 1) % b->size;
    ++b->count;
    pthread_cond_signal(&b->cond);
    pthread_mutex_unlock(&b->mutex);
}

char get(struct Buffer *b)
{
    pthread_mutex_lock(&b->mutex);
    printf("get\n");
    while (b->count == 0)
        pthread_cond_wait(&b->cond, &b->mutex);
    char c = b->buf[b->out];
    b->buf[b->out] = '\0';
    b->out = (b->out + 1) % b->size;
    --b->count;
    pthread_cond_signal(&b->cond);
    pthread_mutex_unlock(&b->mutex);
    return c;
}

void initBuffer(struct Buffer *b, int size)
{
    b->in = 0;
    b->out = 0;
    b->count = 0;
    b->size = size;
    b->buf = (char *)malloc(sizeof(char) * size);
    if (b->buf == NULL)
        exit(-1);
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
}

void destroyBuffer(struct Buffer *b)
{
    free(b->buf);
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
}

int getRandSleepTime()
{
    return rand() % 100 + 750000;
}

void *producerFunc(void *param)
{
    struct Buffer *b = (struct Buffer *)param;
    int i;
    for (i = 0; i < MAX_ITER; ++i)
    {
        // get random small ascii character
        char c = (char)(26 * (rand() / (RAND_MAX + 1.0))) + 97;
        put(b, c);
        usleep(getRandSleepTime() * 2);
    }
    return NULL;
}

void *consumerFunc(void *param)
{
    struct Buffer *b = (struct Buffer *)param;
    int i;
    for (i = 0; i < MAX_ITER; ++i)
    {
        printf("%c\n", get(b));
        usleep(getRandSleepTime());
    }
    return NULL;
}

int main(int argc, char **argv)
{
    UNUSED(argc);
    UNUSED(argv);

    pthread_t producer, consumer;

    srand(time(NULL));

    struct Buffer b;
    initBuffer(&b, 10);

    pthread_create(&producer, NULL, producerFunc, &b);
    pthread_create(&consumer, NULL, consumerFunc, &b);

    pthread_join(producer, NULL);
    pthread_join(consumer, NULL);

    destroyBuffer(&b);
    return 0;
}
