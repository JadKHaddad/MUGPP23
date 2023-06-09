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
                    __global unsigned char * flatRValuesOut, \
                    __global unsigned char * flatGValuesOut, \
                    __global unsigned char * flatBValuesOut, \
                    __global float * flatWeights, \
                    const int width, \
                    const int height) \
    { \
        int x = get_global_id(0); \
        int y = get_global_id(1); \
        float r = 0.0; \
        float g = 0.0; \
        float b = 0.0; \
        for (int i = -2; i <= 2; i++) \
        { \
            for (int j = -2; j <= 2; j++) \
            { \
                r += flatRValues[(x + i) + (y + j) * width] * flatWeights[(i + 2) + (j + 2) * 5]; \
                g += flatGValues[(x + i) + (y + j) * width] * flatWeights[(i + 2) + (j + 2) * 5]; \
                b += flatBValues[(x + i) + (y + j) * width] * flatWeights[(i + 2) + (j + 2) * 5]; \
            } \
        } \
        flatRValuesOut[x + y * width] = r; \
        flatGValuesOut[x + y * width] = g; \
        flatBValuesOut[x + y * width] = b; \
    }";

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

unsigned char *flattenRValues(Pixel *image, int width, int height)
{
    unsigned char *rValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        rValues[i] = image[i].r;
    }
    return rValues;
}

unsigned char *flattenGValues(Pixel *image, int width, int height)
{
    unsigned char *gValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        gValues[i] = image[i].g;
    }
    return gValues;
}

unsigned char *flattenBValues(Pixel *image, int width, int height)
{
    unsigned char *bValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    for (int i = 0; i < width * height; i++)
    {
        bValues[i] = image[i].b;
    }
    return bValues;
}

Pixel *unflattenRGBValues(unsigned char *rValues, unsigned char *gValues, unsigned char *bValues, int width, int height)
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
            std::cout << "(" << (int)image[x + y * width].r << ", " << (int)image[x + y * width].g << ", " << (int)image[x + y * width].b << ") ";
        }
        std::cout << std::endl;
    }
}

void compareImages(Pixel *image1, Pixel *image2, int width, int height, const float delta)
{
    bool equal = true;
    for (int i = 0; i < width * height; i++)
    {
        if (fabs(image1[i].r - image2[i].r) >= delta || fabs(image1[i].g - image2[i].g) >= delta || fabs(image1[i].b - image2[i].b) >= delta)
        {
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

void printWeights(float weights[5][5])
{
    for (int x = -2; x <= 2; x++)
    {
        for (int y = -2; y <= 2; y++)
        {
            std::cout << weights[x + 2][y + 2] << " ";
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

void gaussFilter(Pixel *image, int width, int height, float weight[5][5])
{
    Pixel *newImage = (Pixel *)malloc(sizeof(Pixel) * width * height);

    for (int x = 2; x < width - 2; x++)
    {
        for (int y = 2; y < height - 2; y++)
        {
            float r = 0.0;
            float g = 0.0;
            float b = 0.0;
            for (int i = -2; i <= 2; i++)
            {
                for (int j = -2; j <= 2; j++)
                {
                    r += image[(x + i) + (y + j) * width].r * weight[i + 2][j + 2];
                    g += image[(x + i) + (y + j) * width].g * weight[i + 2][j + 2];
                    b += image[(x + i) + (y + j) * width].b * weight[i + 2][j + 2];
                }
            }
            newImage[x + y * width].r = r;
            newImage[x + y * width].g = g;
            newImage[x + y * width].b = b;
        }
    }

    for (int x = 2; x < width - 2; x++)
    {
        for (int y = 2; y < height - 2; y++)
        {
            image[x + y * width].r = newImage[x + y * width].r;
            image[x + y * width].g = newImage[x + y * width].g;
            image[x + y * width].b = newImage[x + y * width].b;
        }
    }

    free(newImage);
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

    // flattening input data
    float *flatWeights = flattenWeights(weight);
    unsigned char *flatRValues = flattenRValues(image, width, height);
    unsigned char *flatGValues = flattenGValues(image, width, height);
    unsigned char *flatBValues = flattenBValues(image, width, height);

    // create output data
    unsigned char *outRValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    unsigned char *outGValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    unsigned char *outBValues = (unsigned char *)malloc(sizeof(unsigned char) * width * height);

    // do some memory allocation on the device
    cl_mem inputRValues = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputRValues created" << std::endl;

    cl_mem inputGValues = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputGValues created" << std::endl;

    cl_mem inputBValues = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "inputBValues created" << std::endl;

    cl_mem inputWeights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 25, NULL, &err);
    checkError(err);
    std::cout << "inputWeights created" << std::endl;

    cl_mem outputRValues = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "outputRValues created" << std::endl;

    cl_mem outputGValues = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "outputGValues created" << std::endl;

    cl_mem outputBValues = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, &err);
    checkError(err);
    std::cout << "outputBValues created" << std::endl;

    // set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputRValues);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputGValues);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputBValues);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputRValues);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &outputGValues);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputBValues);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &inputWeights);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &height);
    checkError(err);
    std::cout << "kernel arguments set" << std::endl;

    // copy input data to device
    err = clEnqueueWriteBuffer(commandQueue, inputRValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, flatRValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, inputGValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, flatGValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, inputBValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, flatBValues, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, inputWeights, CL_TRUE, 0, sizeof(float) * 25, flatWeights, 0, NULL, NULL);
    checkError(err);
    std::cout << "input data copied to device" << std::endl;

    // execute kernel
    size_t globalWorkSize[2] = {width, height};
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkError(err);
    std::cout << "kernel executed" << std::endl;

    // copy output data back to host
    err = clEnqueueReadBuffer(commandQueue, outputRValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, outRValues, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commandQueue, outputGValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, outGValues, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commandQueue, outputBValues, CL_TRUE, 0, sizeof(unsigned char) * width * height, outBValues, 0, NULL, NULL);
    checkError(err);
    std::cout << "output data copied back to host" << std::endl;

    // combine the three channels to one image
    Pixel *newImage = unflattenRGBValues(outRValues, outGValues, outBValues, width, height); // must be freed by caller

    // free memory
    free(flatWeights);
    free(flatRValues);
    free(flatGValues);
    free(flatBValues);
    free(outRValues);
    free(outGValues);
    free(outBValues);
    std::cout << "host memory freed" << std::endl;

    // free device memory
    err = clReleaseMemObject(inputRValues);
    err |= clReleaseMemObject(inputGValues);
    err |= clReleaseMemObject(inputBValues);
    err |= clReleaseMemObject(inputWeights);
    err |= clReleaseMemObject(outputRValues);
    err |= clReleaseMemObject(outputGValues);
    err |= clReleaseMemObject(outputBValues);
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
    int width;
    int height;

    std::cout << "Reading image " << inFilename << std::endl;
    Pixel *image = readPPM(inFilename, &width, &height);
    std::cout << "Done reading image" << std::endl;

    // seq gaussFilter updates the image in place so we will call the kernel first
    std::cout << "Applying opencl filter" << std::endl;
    Pixel *newImage = gaussFilterOpenCL(image, width, height, weights);
    std::cout << "Done applying opencl filter" << std::endl;

    std::cout << "Writing opencl image " << outFilename_opencl << std::endl;
    writePPM(newImage, outFilename_opencl, width, height);
    std::cout << "Done writing opencl image" << std::endl;

    std::cout << "Applying seq. filter" << std::endl;
    gaussFilter(image, width, height, weights);
    std::cout << "Done applying seq. filter" << std::endl;

    std::cout << "Writing seq. image " << outFilename_seq << std::endl;
    writePPM(image, outFilename_seq, width, height);
    std::cout << "Done writing seq. image" << std::endl;

    std::cout << "Comparing images" << std::endl;
    const float delta = 0.0001;
    compareImages(image, newImage, width, height, delta);
    std::cout << "Done comparing images" << std::endl;

    std::cout << "Freeing memory" << std::endl;
    free(image); // must be explicitly freed
    free(newImage);
    std::cout << "ALL done!" << std::endl;
    return 0;
}

// g++ -o main main.cpp -lOpenCL && ./main