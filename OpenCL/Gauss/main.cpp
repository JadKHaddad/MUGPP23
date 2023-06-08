#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

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

void gaussFilter(Pixel *image, int width, int height, float weight[5][5])
{

    for (int x = 0; x < width; ++x)
    {
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << (float)x / width * 100 << "%\r";
        for (int y = 0; y < height; ++y)
        {
            Pixel new_value;
            new_value.r = 0;
            new_value.g = 0;
            new_value.b = 0;
            for (int xl = -2; xl <= 2; ++xl)
            {
                for (int yl = -2; yl <= 2; ++yl)
                {
                    new_value.r += image[(x + xl) + (y + yl) * width].r * weight[xl + 2][yl + 2];
                    new_value.g += image[(x + xl) + (y + yl) * width].g * weight[xl + 2][yl + 2];
                    new_value.b += image[(x + xl) + (y + yl) * width].b * weight[xl + 2][yl + 2];
                }
            }
            image[x + y * width] = new_value;
        }
    }
}

int main(int argc, char **argv)
{
    const char *inFilename = (argc > 1) ? argv[1] : "lena.ppm";
    const char *outFilename = (argc > 2) ? argv[2] : "output.ppm";

    float weights[5][5];
    calculateWeights(weights);
    int width;
    int height;

    std::cout << "Reading image " << inFilename << std::endl;
    Pixel *image = readPPM(inFilename, &width, &height);
    std::cout << "Done reading image" << std::endl;

    std::cout << "Applying filter" << std::endl;
    gaussFilter(image, width, height, weights);
    std::cout << "Done applying filter" << std::endl;

    std::cout << "Writing image" << outFilename << std::endl;
    writePPM(image, outFilename, width, height);
    std::cout << "Done writing image" << std::endl;

    std::cout << "Freeing memory" << std::endl;
    free(image); // must be explicitly freed
    std::cout << "ALL done!" << std::endl;
    return 0;
}

// g++ -o main main.cpp && ./main