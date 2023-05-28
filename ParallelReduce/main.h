
void *parallel_reduce_inner(void *arg);

int parallel_reduce(int (*op)(int, int),
                    int *data,
                    int len);
