#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernel;

const char *kernelSource = "__kernel \
void gaussFilter(__global unsigned char * flatRValues, \
                    __global unsigned char * flatGValues, \
                    __global unsigned char * flatBValues, \
                    __global float * flatWeights, \
                    const int width) \
    { \
        int x = get_global_id(0); \
        int y = get_global_id(1); \
        float r = 0.0; \
        float g = 0.0; \
        float b = 0.0; \
        for (int i = 0; i < 5; i++) \
        { \
            for (int j = 0; j < 5; j++) \
            { \
                r += flatRValues[(x + 2 + i) + (y + 2 + j) * width] * flatWeights[i + j * 5]; \
                g += flatGValues[(x + 2 + i) + (y + 2 + j) * width] * flatWeights[i + j * 5]; \
                b += flatBValues[(x + 2 + i) + (y + 2 + j) * width] * flatWeights[i + j * 5]; \
            } \
        } \
        flatRValues[x + y * width] = r; \
        flatGValues[x + y * width] = g; \
        flatBValues[x + y * width] = b; \
    }";

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

unsigned char *extractRValues(Pixel *image, int width, int height)
{
    unsigned char *rValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        rValues[i] = image[i].r;
    }
    return rValues;
}

unsigned char *extractGValues(Pixel *image, int width, int height)
{
    unsigned char *gValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        gValues[i] = image[i].g;
    }
    return gValues;
}

unsigned char *extractBValues(Pixel *image, int width, int height)
{
    unsigned char *bValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        bValues[i] = image[i].b;
    }
    return bValues;
}

Pixel *combineRGBValues(unsigned char *rValues, unsigned char *gValues, unsigned char *bValues, int width, int height)
{
    Pixel *image = (Pixel *)malloc(sizeof(Pixel) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        image[i].r = rValues[i];
        image[i].g = gValues[i];
        image[i].b = bValues[i];
    }
    return image;
}

void printImage(Pixel *image, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (x == 5)
            {
                std::cout << std::endl;
                return;
            }
            std::cout << "(" << (int)image[x + y * width].r << ", " << (int)image[x + y * width].g << ", " << (int)image[x + y * width].b << ") ";
        }
        std::cout << std::endl;
    }
}

void compareImages(Pixel *image1, Pixel *image2, int width, int height, const unsigned char delta)
{
    bool equal = true;
    for (int i = 0; i < width * height; i++)
    {
        if (fabs(image1[i].r - image2[i].r) >= delta || fabs(image1[i].g - image2[i].g) >= delta || fabs(image1[i].b - image2[i].b) >= delta)
        {
            std::cout << "Images are not equal at pixel " << i << std::endl;
            std::cout << "values are: " << std::endl;
            std::cout << "(" << (int)image1[i].r << ", " << (int)image1[i].g << ", " << (int)image1[i].b << ") ";
            std::cout << "(" << (int)image2[i].r << ", " << (int)image2[i].g << ", " << (int)image2[i].b << ") ";
            equal = false;
            break;
        }
    }
    if (equal)
    {
        std::cout << "Images are equal, delta: " << delta << std::endl;
    }
    else
    {
        std::cout << "Images are not equal, delta: " << delta << std::endl;
    }
}

// write image to a PPM file with the given filename
void writePPM(Pixel *pixels, const char *filename, int width, int height)
{
    std::ofstream outputFile(filename, std::ios::binary);

    // write header:
    outputFile << "P6\n"
               << width << " " << height << "\n255\n";

    outputFile.write((const char *)pixels,
                     sizeof(Pixel) * width * height);
}

// Pointer returned must be explicitly freed!
Pixel *readPPM(const char *filename, int *width, int *height)
{
    std::ifstream inputFile(filename, std::ios::binary);

    // parse header
    // first line: P6\n
    inputFile.ignore(2, '\n'); // ignore P6
    // possible comments:
    while (inputFile.peek() == '#')
    {
        inputFile.ignore(1024, '\n');
    } // skip comment
    // next line: width_height\n
    inputFile >> (*width);
    inputFile.ignore(1, ' '); // ignore space
    inputFile >> (*height);
    inputFile.ignore(1, '\n'); // ignore newline
    // possible comments:
    while (inputFile.peek() == '#')
    {
        inputFile.ignore(1024, '\n');
    } // skip comment
    // last header line: 255\n:
    inputFile.ignore(4, '\n'); // ignore 255 and newline

    Pixel *data = (Pixel *)malloc(sizeof(Pixel) * (*width) * (*height));

    inputFile.read((char *)data, sizeof(Pixel) * (*width) * (*height));

    return data;
}

using namespace std;

void calculateWeights(float weights[5][5])
{
    float sigma = 1.0;
    float r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    float sum = 0.0;

    // generate weights for 5x5 kernel
    for (int x = -2; x <= 2; x++)
    {
        for (int y = -2; y <= 2; y++)
        {
            r = x * x + y * y;
            weights[x + 2][y + 2] = exp(-(r / s)) / (M_PI * s);
            sum += weights[x + 2][y + 2];
        }
    }

    // normalize the weights
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            weights[i][j] /= sum;
        }
    }
}

void calculateSimpleWeights(float weights[5][5])
{
    for (int x = -2; x <= 2; x++)
    {
        for (int y = -2; y <= 2; y++)
        {
            weights[x + 2][y + 2] = 1.0 / 25.0;
        }
    }
}

void printWeights(float weights[5][5])
{
    for (int x = 0; x < 5; x++)
    {
        for (int y = 0; y < 5; y++)
        {
            std::cout << weights[x][y] << " ";
        }
        std::cout << std::endl;
    }
}

void printFlatWeights(float *weights)
{
    for (int x = 0; x < 5; x++)
    {
        for (int y = 0; y < 5; y++)
        {
            std::cout << weights[x * 5 + y] << " ";
        }
        std::cout << std::endl;
    }
}

float *flattenWeights(float weights[5][5])
{
    float *flattenedWeights = (float *)malloc(sizeof(float) * 25);
    for (int i = 0; i < 25; i++)
    {
        flattenedWeights[i] = weights[i / 5][i % 5];
    }
    return flattenedWeights;
}

void unflattenWeights(float *flattenedWeights, float weights[5][5])
{
    for (int i = 0; i < 25; i++)
    {
        weights[i / 5][i % 5] = flattenedWeights[i];
    }
}

Pixel *gaussFilter(Pixel *image, int width, int height, float weight[5][5])
{
    Pixel *newImage = (Pixel *)malloc(sizeof(Pixel) * width * height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            float r = 0.0;
            float g = 0.0;
            float b = 0.0;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    r += image[(x + i) + (y + j) * width].r * weight[i][j];
                    g += image[(x + i) + (y + j) * width].g * weight[i][j];
                    b += image[(x + i) + (y + j) * width].b * weight[i][j];
                }
            }
            newImage[x + y * width].r = r;
            newImage[x + y * width].g = g;
            newImage[x + y * width].b = b;
        }
    }

    return newImage;
}

void checkError(cl_int err)
{
    if (err != CL_SUCCESS)
        printf("Error with errorcode: %d\n", err);
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

void makeKernel()
{
    cl_int err;
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
    kernel = clCreateKernel(program, "gaussFilter", &err);
    checkError(err);
    printf("kernel created\n");
}

Pixel *gaussFilterOpenCL(Pixel *image, int width, int height, float weight[5][5])
{
    initOpenCL();
    makeKernel();
    cl_int err;

    // data
    float *flatWeights = flattenWeights(weight);
    unsigned char *RValues = extractRValues(image, width, height);
    unsigned char *GValues = extractGValues(image, width, height);
    unsigned char *BValues = extractBValues(image, width, height);

    // do some memory allocation on the device
    cl_mem kernelRValues = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputRValues created" << std::endl;

    cl_mem kernelGValues = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputGValues created" << std::endl;

    cl_mem kernelBValues = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputBValues created" << std::endl;

    cl_mem kernelWeights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 25, NULL, &err);
    checkError(err);
    std::cout << "inputWeights created" << std::endl;

    // set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &kernelRValues);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &kernelGValues);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &kernelBValues);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &kernelWeights);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &width);
    checkError(err);
    std::cout << "kernel arguments set" << std::endl;

    // copy input data to device
    err = clEnqueueWriteBuffer(commandQueue, kernelRValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, RValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, kernelGValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, GValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, kernelBValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, BValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, kernelWeights, CL_TRUE, 0, sizeof(float) * 25, flatWeights, 0, NULL, NULL);
    checkError(err);
    std::cout << "input data copied to device" << std::endl;

    // execute kernel
    size_t globalWorkSize[2] = {width, height};
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkError(err);
    std::cout << "kernel executed" << std::endl;

    // copy output data back to host
    err = clEnqueueReadBuffer(commandQueue, kernelRValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, RValues, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commandQueue, kernelGValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, GValues, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commandQueue, kernelBValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, BValues, 0, NULL, NULL);
    checkError(err);
    std::cout << "output data copied back to host" << std::endl;

    // combine the three channels to one image
    Pixel *newImage = combineRGBValues(RValues, GValues, BValues, width, height); // must be freed by caller

    // free memory
    free(RValues);
    free(GValues);
    free(BValues);
    free(flatWeights);
    std::cout << "host memory freed" << std::endl;

    // free device memory
    err = clReleaseMemObject(kernelRValues);
    err |= clReleaseMemObject(kernelGValues);
    err |= clReleaseMemObject(kernelBValues);
    err |= clReleaseMemObject(kernelWeights);
    checkError(err);
    std::cout << "device memory freed" << std::endl;

    return newImage;
}

int main(int argc, char **argv)
{
    const char *inFilename = (argc > 1) ? argv[1] : "lena.ppm";
    const char *outFilename_seq = (argc > 2) ? argv[2] : "output_seq.ppm";
    const char *outFilename_opencl = (argc > 3) ? argv[3] : "output_opencl.ppm";

    float weights[5][5];
    calculateWeights(weights);
    // calculateSimpleWeights(weights);
    int width;
    int height;

    std::cout << "Reading image " << inFilename << std::endl;
    Pixel *image = readPPM(inFilename, &width, &height);
    std::cout << "Done reading image" << std::endl;

    std::cout << "Applying opencl filter" << std::endl;
    Pixel *newImageOpenCL = gaussFilterOpenCL(image, width, height, weights);
    std::cout << "Done applying opencl filter" << std::endl;

    std::cout << "Writing opencl image " << outFilename_opencl << std::endl;
    writePPM(newImageOpenCL, outFilename_opencl, width, height);
    std::cout << "Done writing opencl image" << std::endl;

    std::cout << "Applying seq. filter" << std::endl;
    Pixel *newImageSeq = gaussFilter(image, width, height, weights);
    std::cout << "Done applying seq. filter" << std::endl;

    std::cout << "Writing seq. image " << outFilename_seq << std::endl;
    writePPM(newImageSeq, outFilename_seq, width, height);
    std::cout << "Done writing seq. image" << std::endl;

    std::cout << "Comparing images" << std::endl;
    const unsigned char delta = 24;
    compareImages(newImageSeq, newImageOpenCL, width, height, delta);
    std::cout << "Done comparing images" << std::endl;

    printImage(image, width, height);
    std::cout << "-----------------------" << std::endl;
    printImage(newImageOpenCL, width, height);
    std::cout << "-----------------------" << std::endl;
    printImage(newImageSeq, width, height);
    std::cout << "-----------------------" << std::endl;

    std::cout << "Freeing memory" << std::endl;
    free(image); // must be explicitly freed
    free(newImageOpenCL);
    free(newImageSeq);
    std::cout << "ALL done!" << std::endl;
    return 0;
}

// g++ -o main main.cpp -lOpenCL && ./main