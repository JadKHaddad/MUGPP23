#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <string.h>

float *M;
float *N;
float *P_opencl;
float *P_seq;
int Width;
int Num_Threads;

const float delta = 0.0001;

// fill f width size many random float values
void fill(float *f, int size)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < size; i += 1)
        f[i] = ((float)rand()) / RAND_MAX;
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
void MatrixMulSeq()
{
    int Col, Row, k;
    for (Col = 0; Col < Width; ++Col)
        for (Row = 0; Row < Width; ++Row)
        {
            float sum = 0;
            for (k = 0; k < Width; k += 1)
            {
                sum += M[Row * Width + k] * N[k * Width + Col];
            }
            P_seq[Row * Width + Col] = sum;
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
void MatrixMultKernel(__global float* Md, \
                      __global float* Nd, \
                      __global float* Pd, int width) { \
  int col = get_global_id(0); \
  int row = get_global_id(1); \
  \
  float sum = 0; \
  for (int k = 0; k < width; k+=1) \
    sum += Md[row * width + k] * Nd[k * width + col]; \
  \
  Pd[row * width + col] = sum; \
}";
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
    kernel = clCreateKernel(program, "MatrixMultKernel", &err);
    checkError(err);
    printf("kernel created\n");
}

void MatrixMulOpenCL(float *M, float *N, float *P, int width)
{
    cl_int err;
    int size = width * width * sizeof(float);

    // Buffer Md erzeugen und direkt auf das Device kopieren
    cl_mem Md = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, M, &err);
    checkError(err);
    printf("buffer md created and copied\n");

    // Buffer ND erzeugen ohne zu kopieren
    cl_mem Nd = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
    checkError(err);
    printf("buffer nd created\n");
    // Daten explizit auf das Device kopieren
    // Dieser Aufruf ist nicht blockierend (CL_FALSE)
    err = clEnqueueWriteBuffer(commandQueue, Nd, CL_FALSE, 0, size, N, 0, NULL, NULL);
    checkError(err);
    printf("enqueued write buffer nd\n");

    // Speicher fuer Ergebnis Matrix reservieren
    cl_mem Pd = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    checkError(err);
    printf("buffer pd created and memory allocated\n");

    // Setze Argument fuer den Kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &Md);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &Nd);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &Pd);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    checkError(err);
    printf("kernel arguments set\n");

    size_t globalSize[] = {width, width};
    // Starte Kernel width * width mal
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    checkError(err);
    printf("enqueued kernel\n");

    // Daten vom Device kopieren
    // Dieser Aufruf ist blockierend (CL_TRUE)
    err = clEnqueueReadBuffer(commandQueue, Pd, CL_TRUE, 0, size, P, 0, NULL, NULL);
    checkError(err);
    printf("enqueued read buffer pd\n");
}

// end OpenCL section
// ######################################################

void init()
{
    Width = 1024;
    M = (float *)malloc(Width * Width * sizeof(float));
    N = (float *)malloc(Width * Width * sizeof(float));
    P_opencl = (float *)malloc(Width * Width * sizeof(float));
    P_seq = (float *)malloc(Width * Width * sizeof(float));

    fill(M, Width * Width);
    fill(N, Width * Width);
    initOpenCL();
    makeKernel();
};

int main(void)
{
    struct timeval start, end;
    init();

    gettimeofday(&start, NULL);
    MatrixMulOpenCL(M, N, P_opencl, Width);
    gettimeofday(&end, NULL);
    printf("Time elapsed OpenCL: %fmsecs\n",
           (float)(1000.0 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec)));

    gettimeofday(&start, NULL);
    MatrixMulSeq();
    gettimeofday(&end, NULL);
    printf("Time elapsed Seq: %fmsecs\n",
           (float)(1000.0 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec)));

    compare(P_seq, P_opencl, Width * Width);

    return 0;
}

// gcc -o main ./main.c -lOpenCL && ./main