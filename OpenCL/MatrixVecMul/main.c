#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <string.h>

// Matrix
float *M;
// Vectoren
float *V;
float *R_opencl;
float *R_seq;

int Width;
int Num_Threads;

const float delta = 0.0001;

// fill f width size many random float values
void fillRandom(float *f, int size)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < size; i += 1)
        f[i] = ((float)rand()) / RAND_MAX;
}

void fillIncremental(float *f, int size)
{
    int i;
    for (i = 0; i < size; i += 1)
        f[i] = i;
}

// compares every pair lhs[i] and rhs[i] for i < width
void compare(float *lhs, float *rhs, int width)
{
    int errors = 0;
    int i;
    for (i = 0; i < width; i += 1)
    {
        if (fabs(lhs[i] - rhs[i]) >= 0.0001)
        {
            printf("%f : %f\n", lhs[i], rhs[i]);
            errors += 1;
        }
    }
    if (errors > 0)
        printf("%d errors occured.\n", errors);
    else
        printf("no errors occured.\n");
}

// sequentiell matrix multiplication
void MatrixVecMulSeq()
{
    int Row, k;
    for (Row = 0; Row < Width; ++Row)
    {
        float sum = 0;
        for (k = 0; k < Width; k += 1)
        {
            sum += M[Row * Width + k] * V[k];
        }
        R_seq[Row] = sum;
    }
}
// ######################################################
// Start OpenCL section
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernel;

// check err for an OpenCL error code
void checkError(cl_int err)
{
    if (err != CL_SUCCESS)
        printf("Error with errorcode: %d\n", err);
}

void initOpenCL()
{
    cl_int err;

    // Speichere 1 Plattform in platform
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err);
    printf("platform selected\n");

    // Speichere 1 Device beliebigen Typs in device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    checkError(err);
    printf("device selected\n");

    // erzeuge Context fuer das Device device
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err);
    printf("context created\n");

    // erzeuge Command Queue zur Verwaltung von device
    commandQueue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err);
    printf("commandQueue created\n");
}

void printBuildLog(cl_program program, cl_device_id device)
{
    cl_int err;
    char *build_log;
    size_t build_log_size;
    // Speichere den Build Log fuer program und device in build_log
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
    checkError(err);

    build_log = (char *)malloc(build_log_size);

    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
    checkError(err);

    printf("Log:\n%s\n", build_log);

    free(build_log);
}

void makeKernel()
{
    cl_int err;

    // Kernel Quellcode
    const char *kernelSource = "__kernel \
void MatrixVecMultKernel(__global float* Md, \
                         __global float* Vd, \
                         __global float* Rd, int width) { \
    int Row = get_global_id(0); \
    if (Row >= width) return; \
    float sum = 0; \
    for (int k = 0; k < width; k += 1) { \
        sum += Md[Row * width + k] * Vd[k]; \
    } \
    Rd[Row] = sum; \
};";

    // Laenge des Kernel Quellcodes
    size_t sourceLength = strlen(kernelSource);
    cl_program program;
    // Ein Programm aus dem Kernel Quellcode wird erzeugt
    program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceLength, &err);
    checkError(err);
    printf("program created\n");
    // Das Programm wird fuer alle Devices des Contextes gebaut
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
        printBuildLog(program, device);
    else
        printf("program build successfully\n");
    kernel = clCreateKernel(program, "MatrixVecMultKernel", &err);
    checkError(err);
    printf("kernel created\n");
}

void MatrixVecMulOpenCL(float *m, float *v, float *r, int width)
{
    cl_int err;
    int m_size = width * width * sizeof(float);
    int v_size = width * sizeof(float);

    // Buffer Md erzeugen und direkt auf das Device kopieren
    cl_mem Md = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m_size, m, &err);
    checkError(err);
    printf("buffer md created and copied\n");

    // Buffer VD erzeugen ohne zu kopieren
    cl_mem Vd = clCreateBuffer(context, CL_MEM_READ_ONLY, v_size, NULL, &err);
    checkError(err);
    printf("buffer vd created\n");
    // Daten explizit auf das Device kopieren
    // Dieser Aufruf ist nicht blockierend (CL_FALSE)
    err = clEnqueueWriteBuffer(commandQueue, Vd, CL_FALSE, 0, v_size, v, 0, NULL, NULL);
    checkError(err);
    printf("enqueued write buffer nd\n");

    // Speicher fuer Ergebnis Vector reservieren
    cl_mem Rd = clCreateBuffer(context, CL_MEM_READ_WRITE, v_size, NULL, &err);
    checkError(err);
    printf("buffer rd created and memory allocated\n");

    // Setze Argument fuer den Kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &Md);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &Vd);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &Rd);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    checkError(err);
    printf("kernel arguments set\n");

    size_t globalSize[] = {(size_t)width};
    // Starte Kernel width  mal
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);
    checkError(err);
    printf("enqueued kernel\n");

    // Daten vom Device kopieren
    // Dieser Aufruf ist blockierend (CL_TRUE)
    err = clEnqueueReadBuffer(commandQueue, Rd, CL_TRUE, 0, v_size, r, 0, NULL, NULL);
    checkError(err);
    printf("enqueued read buffer pd\n");
}

// end OpenCL section
// ######################################################

void init()
{
    Width = 1024;
    M = (float *)malloc(Width * Width * sizeof(float));
    V = (float *)malloc(Width * sizeof(float));
    R_opencl = (float *)malloc(Width * sizeof(float));
    R_seq = (float *)malloc(Width * sizeof(float));

    fillRandom(M, Width * Width);
    fillRandom(V, Width);
    // fillIncremental(M, Width * Width);
    // fillIncremental(V, Width);

    initOpenCL();
    makeKernel();
};

void printVector(float *f, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", f[i]);
    }
    printf("\n");
}

void printMatrix(float *f, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", f[i]);
        if ((i + 1) % Width == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}

int main(void)
{
    struct timeval start, end;
    init();

    gettimeofday(&start, NULL);
    MatrixVecMulOpenCL(M, V, R_opencl, Width);
    gettimeofday(&end, NULL);
    printf("Time elapsed OpenCL: %fmsecs\n",
           (float)(1000.0 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec)));

    gettimeofday(&start, NULL);
    MatrixVecMulSeq();
    gettimeofday(&end, NULL);
    printf("Time elapsed Seq: %fmsecs\n",
           (float)(1000.0 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec)));

    compare(R_seq, R_opencl, Width);

    return 0;
}

// gcc -o main ./main.c -lOpenCL && ./main