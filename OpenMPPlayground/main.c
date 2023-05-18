#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define CHUNK 10
#define N 100
#define WIDTH 10

void hello()
{
#pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", my_rank, thread_count);
    }
}

void vectors_init(float *a, float *b, int len)
{
    for (int i = 0; i < len; i++)
        a[i] = b[i] = i * 1.0;
}

void vector_print(float *a, const char *name, int len)
{
    printf("%s: ", name);
    for (int i = 0; i < len; i++)
        printf("[%.1f] ", a[i]);
    printf("\n");
}

void vector_add(float *a, float *b, float *c, int len, int chunk)
{
    int i;

#pragma omp parallel shared(a, b, c, len, chunk) private(i)
    {
#pragma omp for schedule(dynamic, chunk)
        for (i = 0; i < len; i++)
            c[i] = a[i] + b[i];
    }
}

void vector_add_2(float *a, float *b, float *c, int len, int chunk)
{
    int i;

#pragma omp parallel for shared(a, b, c, len) private(i) \
    schedule(static, chunk)
    for (i = 0; i < len; i++)
        c[i] = a[i] + b[i];
}

void section_1()
{
    printf("Section 1\n");
}

void section_2()
{
    printf("Section 2\n");
}

void sections()
{
#pragma omp parallel
    {
#pragma omp sections nowait
        {
#pragma omp section
            section_1();
#pragma omp section
            section_2();
        }
    }
}

float reduction(float *a, float *b, int len, int chunk)
{
    int i;
    float result = 0.0;

#pragma omp parallel for default(shared) private(i) \
    schedule(static, chunk)                         \
    reduction(+ : result)
    for (i = 0; i < len; i++)
        result = result + (a[i] * b[i]);

    return result;
}

void matrix_mul(float *m, float *n, float *p, int width)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k)
            {
                sum += m[i * width + k] * n[k * width + j];
            }
            p[i * width + j] = sum;
        }
}

void matrix_print(float *m, const char *name, int width)
{
    printf("%s:\n", name);
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            printf("[%.1f]\t", m[i * width + j]);
        }
        printf("\n");
    }
}

void matrix_init(float *m, int width)
{
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
            m[i * width + j] = i * width + j;
}

typedef struct list
{
    int data;
    struct list *next;
} l_list;

l_list *create_l_list(int data)
{
    l_list *elem = malloc(sizeof(l_list));
    elem->data = data;
    elem->next = NULL;
    return elem;
}

void free_l_list(l_list *head)
{
    l_list *elem = head;
    while (elem != NULL)
    {
        l_list *next = elem->next;
        free(elem);
        elem = next;
    }
}

void work_on_l_list_elem(l_list *elem)
{
    printf("Working on element %d\n", elem->data);
    elem->data = elem->data * 2;
}

void print_l_list(l_list *head)
{
    l_list *elem = head;
    while (elem != NULL)
    {
        printf("%d ", elem->data);
        elem = elem->next;
    }
    printf("\n");
}

void do_tasks_with_l_list(l_list *elem)
{
#pragma omp parallel
    {
#pragma omp single nowait
        {
            while (elem != NULL)
            {
#pragma omp task firstprivate(elem)
                {
                    work_on_l_list_elem(elem);
                }
                elem = elem->next;
            }
        }
    }
}

long fib(int n)
{
    long fnm1, fnm2, fn;
    if (n == 0 || n == 1)
        return n;
#pragma omp task shared(fnm1)
    fnm1 = fib(n - 1);
#pragma omp task shared(fnm2)
    fnm2 = fib(n - 2);
#pragma omp taskwait
    fn = fnm1 + fnm2;
    return fn;
}

int sum_from_one_to_n(int n)
{
    int i;
    int sum = 0;
#pragma omp parallel for private(i) shared(n) reduction(+ : sum) schedule(static)
    for (i = 1; i <= n; ++i)
    {
        sum += i;
    }
    return sum;
}

int main(int argc, char *argv[])
{
    hello();
    printf("\n");

    float a[N], b[N], c[N];
    vectors_init(a, b, N);
    vector_print(a, "a", N);
    printf("\n");
    vector_print(b, "b", N);
    printf("\n");

    vector_add(a, b, c, N, CHUNK);
    vector_print(c, "c", N);
    printf("\n");

    vector_add_2(a, b, c, N, CHUNK);
    vector_print(c, "c", N);
    printf("\n");

    sections();
    printf("\n");

    float red = reduction(a, b, N, CHUNK);
    printf("Reduction: %f\n", red);
    printf("\n");

    l_list *head = create_l_list(1);
    l_list *elem_1 = create_l_list(2);
    l_list *elem_2 = create_l_list(3);
    head->next = elem_1;
    elem_1->next = elem_2;
    elem_2->next = NULL;
    printf("Initial list: ");
    print_l_list(head);
    do_tasks_with_l_list(head);
    printf("Final list: ");
    print_l_list(head);
    free_l_list(head);
    printf("\n");

    printf("Fibonacci of 10: %ld\n", fib(10));
    printf("\n");

    float m[WIDTH * WIDTH], n[WIDTH * WIDTH], p[WIDTH * WIDTH];
    matrix_init(m, WIDTH);
    matrix_init(n, WIDTH);
    matrix_print(m, "m", WIDTH);
    printf("\n");
    matrix_print(n, "n", WIDTH);
    printf("\n");
    matrix_mul(m, n, p, WIDTH);
    matrix_print(p, "p", WIDTH);
    printf("\n");

    int sum = sum_from_one_to_n(N);
    printf("Sum from 1 to %d: %d\n", N, sum);

    return 0;
}

// Compile: gcc -fopenmp main.c -o main
// Run: ./main
// Compile and run: gcc -fopenmp main.c -o main && ./main
