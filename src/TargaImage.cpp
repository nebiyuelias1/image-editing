///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.cpp                          Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//
//      Implementation of TargaImage methods.  You must implement the image
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#include "Globals.h"
#include "TargaImage.h"
#include "libtarga.h"
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <unordered_map>
#include <functional>
#include <cfloat>
#include <random>
#include <algorithm>

using namespace std;

// constants
const int           RED             = 0;                // red channel
const int           GREEN           = 1;                // green channel
const int           BLUE            = 2;                // blue channel
const unsigned char BACKGROUND[3]   = { 0, 0, 0 };      // background color


// Computes n choose s, efficiently
double Binomial(int n, int s)
{
    double        res;

    res = 1;
    for (int i = 1 ; i <= s ; i++)
        res = (n - i + 1) * res / i ;

    return res;
}// Binomial


///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage() : width(0), height(0), data(NULL)
{}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h) : width(w), height(h)
{
   data = new unsigned char[width * height * 4];
   ClearToBlack();
}// TargaImage



///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables to values given.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h, unsigned char *d)
{
    int i;

    width = w;
    height = h;
    data = new unsigned char[width * height * 4];

    for (i = 0; i < width * height * 4; i++)
	    data[i] = d[i];
}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Copy Constructor.  Initialize member to that of input
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(const TargaImage& image) 
{
   width = image.width;
   height = image.height;
   data = NULL; 
   if (image.data != NULL) {
      data = new unsigned char[width * height * 4];
      memcpy(data, image.data, sizeof(unsigned char) * width * height * 4);
   }
}


///////////////////////////////////////////////////////////////////////////////
//
//      Destructor.  Free image memory.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::~TargaImage()
{
    if (data)
        delete[] data;
}// ~TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Converts an image to RGB form, and returns the rgb pixel data - 24 
//  bits per pixel. The returned space should be deleted when no longer 
//  required.
//
///////////////////////////////////////////////////////////////////////////////
unsigned char* TargaImage::To_RGB(void)
{
    unsigned char   *rgb = new unsigned char[width * height * 3];
    int		    i, j;

    if (! data)
	    return NULL;

    // Divide out the alpha
    for (i = 0 ; i < height ; i++)
    {
	    int in_offset = i * width * 4;
	    int out_offset = i * width * 3;

	    for (j = 0 ; j < width ; j++)
        {
	        RGBA_To_RGB(data + (in_offset + j*4), rgb + (out_offset + j*3));
	    }
    }

    return rgb;
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Save the image to a targa file. Returns 1 on success, 0 on failure.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Save_Image(const char *filename)
{
    TargaImage	*out_image = Reverse_Rows();

    if (! out_image)
	    return false;

    if (!tga_write_raw(filename, width, height, out_image->data, TGA_TRUECOLOR_32))
    {
	    cout << "TGA Save Error: %s\n", tga_error_string(tga_get_last_error());
	    return false;
    }

    delete out_image;

    return true;
}// Save_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Load a targa image from a file.  Return a new TargaImage object which 
//  must be deleted by caller.  Return NULL on failure.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Load_Image(char *filename)
{
    unsigned char   *temp_data;
    TargaImage	    *temp_image;
    TargaImage	    *result;
    int		        width, height;

    if (!filename)
    {
        cout << "No filename given." << endl;
        return NULL;
    }// if

    temp_data = (unsigned char*)tga_load(filename, &width, &height, TGA_TRUECOLOR_32);
    if (!temp_data)
    {
        cout << "TGA Error: %s\n", tga_error_string(tga_get_last_error());
	    width = height = 0;
	    return NULL;
    }
    temp_image = new TargaImage(width, height, temp_data);
    free(temp_data);

    result = temp_image->Reverse_Rows();

    delete temp_image;

    return result;
}// Load_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Convert image to grayscale.  Red, green, and blue channels should all 
//  contain grayscale value.  Alpha channel should be left unchanged.  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::To_Grayscale()
{
    if (!data) return false; // return false if no data

    int numPixels = width * height;
    for (int i = 0; i < numPixels; i++)
    {
        // Extract the RGB values from the pixel data
        unsigned char *pixel = data + i * 4;
        unsigned char r = pixel[0];
        unsigned char g = pixel[1];
        unsigned char b = pixel[2];

        // Compute the grayscale value and set all three RGB components to it
        unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }

    return true; // return true on success
}// To_Grayscale


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using uniform quantization.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Uniform()
{
    // If there is no image data quit immediately.
    if (!data) {
        return false;
    }

    // Define the color palette:
    const int num_blue_shades = 4;
    const int num_red_shades = 8;
    const int num_green_shades = 8;

    // Color shade value = (255 / (number of shades - 1)) * shade index
    const int blue_shades[num_blue_shades] = {0, 85, 170, 255};
    const int red_shades[num_red_shades] = {0, 32, 64, 96, 128, 160, 192, 224};
    const int green_shades[num_green_shades] = {0, 32, 64, 96, 128, 160, 192, 224};

    // Define the color lookup tables.
    uint8_t blue_lut[256];
    uint8_t red_lut[256];
    uint8_t green_lut[256];

    for (int i = 0; i < 256; i++)
    {
        int closest_blue_shade = 0;
        int closest_red_shade = 0;
        int closest_green_shade = 0;
        int min_distance = INT_MAX;

        for (int j = 0; j < num_blue_shades; j++)
        {
            for (int k = 0; k < num_red_shades; k++)
            {
                for (int l = 0; l < num_green_shades; l++)
                {
                    // Calculate the Euclidean distance between the original color and the palette color.
                    int distance = pow(i - blue_shades[j], 2) + pow(i - green_shades[l], 2) + pow(i - red_shades[k], 2);
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        closest_blue_shade = blue_shades[j];
                        closest_red_shade = red_shades[k];
                        closest_green_shade = green_shades[l];
                    }
                }
            }
        }

        blue_lut[i] = closest_blue_shade;
        red_lut[i] = closest_red_shade;
        green_lut[i] = closest_green_shade;
    }

    int numPixels = width * height;
    for (int i = 0; i < numPixels; i++)
    {
        uint8_t *pixel = data + i * 4;
        pixel[0] = red_lut[pixel[0]];
        pixel[1] = green_lut[pixel[1]];
        pixel[2] = blue_lut[pixel[2]];
    }

    return true;
}// Quant_Uniform

bool compare_pair(const std::pair<int, RGBColor>& lhs, const std::pair<int, RGBColor>& rhs) {
    return lhs.first > rhs.first;
}
///////////////////////////////////////////////////////////////////////////////
//
//      Convert the image to an 8 bit image using populosity quantization.  
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Populosity()
{
    // If there is no image data quit immediately.
    if (!data) {
        return false;
    }

    // Step 1: Perform uniform quantization to 32 shades of each primary
    Quant_Uniform_To_N_Shades(32, 32, 32);

    
    // Step 2: Build a color usage histogram
    std::unordered_map<unsigned int, int> histogram;
    for (int i = 0; i < width * height; i++)
    {
        unsigned int color = ((data[i * 4] >> 3) << 11) | ((data[i * 4 + 1] >> 3) << 6) | (data[i * 4 + 2] >> 3);
        histogram[color]++;
    }

    // Step 3: Sort the colors by usage frequency
    std::vector<std::pair<unsigned int, int>> colors(histogram.begin(), histogram.end());
    std::sort(colors.begin(), colors.end(), [](const std::pair<unsigned int, int>& a, const std::pair<unsigned int, int>& b) { return a.second > b.second; });

    // Step 4: Map each original color to its closest chosen color
    for (int i = 0; i < width * height; i++)
    {
        unsigned int color = ((data[i * 4] >> 3) << 11) | ((data[i * 4 + 1] >> 3) << 6) | (data[i * 4 + 2] >> 3);

        int closestColorIndex = 0;
        int closestColorDistance = std::numeric_limits<int>::max();

        for (int j = 0; j < std::min(256, (int)colors.size()); j++)
        {
            unsigned int chosenColor = colors[j].first;

            int distance = std::sqrt(
                std::pow((int)((color >> 11) & 0x1F) - (int)((chosenColor >> 11) & 0x1F), 2) +
                std::pow((int)((color >> 6) & 0x1F) - (int)((chosenColor >> 6) & 0x1F), 2) +
                std::pow((int)(color & 0x1F) - (int)(chosenColor & 0x1F), 2)
            );

            if (distance < closestColorDistance)
            {
                closestColorIndex = j;
                closestColorDistance = distance;
            }
        }

        unsigned int chosenColor = colors[closestColorIndex].first;
        data[i * 4] = ((chosenColor >> 11) & 0x1F) << 3;
        data[i * 4 + 1] = ((chosenColor >> 6) & 0x1F) << 3;
        data[i * 4 + 2] = (chosenColor & 0x1F) << 3;
    }


    return true;
}// Quant_Populosity


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image using a threshold of 1/2.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Threshold()
{
     // If there is no image data, return false
    if (!data) {
        return false;
    }

    int numPixels = width * height;

    for (int i = 0; i < numPixels; i++) {
        // Extract the RGB values from the pixel data
        unsigned char* pixel = data + i * 4;
        float gray = 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];

        // Threshold dithering
        if (gray < 128.0f) {
            // Set the pixel to black
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else {
            // Set the pixel to white
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    return true;
}// Dither_Threshold


///////////////////////////////////////////////////////////////////////////////
//
//      Dither image using random dithering.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Random()
{
    // Convert image to grayscale
    if (!To_Grayscale()) {
        return false;
    }

    // Random number generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(-0.2, 0.2);

    // Loop through all pixels
    for (int i = 0; i < width * height; i++) {
        // Calculate new pixel value with random dithering
        float new_value = data[i * 4] / 255.0f + distribution(generator);
        new_value = new_value > 0.5f ? 1.0f : 0.0f;

        // Set new pixel value
        data[i * 4] = static_cast<unsigned char>(new_value * 255.0f);
        data[i * 4 + 1] = data[i * 4];
        data[i * 4 + 2] = data[i * 4];
    }

    return true;
}// Dither_Random

///////////////////////////////////////////////////////////////////////////////
//
//      Perform Floyd-Steinberg dithering on the image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_FS()
{
    if (!data) {
        return false;
    }

    To_Grayscale();

    // Iterate over each pixel in the image
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int offset = (y * width * 4) + (x * 4);
            unsigned char * pixel = data + offset;

            uint8_t gray = *pixel;


            // Calculate the new black or white value based on the grayscale value
            uint8_t newValue = (gray < 128) ? 0 : 255;

             // Set the new value in the dithered image buffer
            pixel[0] = newValue;
            pixel[1] = newValue;
            pixel[2] = newValue;

            // Calculate the error between the old and new value
            int error = gray - newValue;

            // Distribute the error to the surrounding pixels using the Floyd-Steinberg pattern
            if (x < width - 1) {
                data[(y * width + x + 1) * 4] += error * 7.0 / 16;
                data[(y * width + x + 1) * 4 + 1] += error * 7.0 / 16;
                data[(y * width + x + 1) * 4 + 2] += error * 7.0 / 16;
            }
            if (x > 0 && y < height - 1) {
                data[((y + 1) * width + x - 1) * 4] += error * 3.0 / 16;
                data[((y + 1) * width + x - 1) * 4 + 1] += error * 3.0 / 16;
                data[((y + 1) * width + x - 1) * 4 + 2] += error * 3.0 / 16;
            }
            if (y < height - 1) {
                data[((y + 1) * width + x) * 4] += error * 5.0 / 16;
                data[((y + 1) * width + x) * 4 + 1] += error * 5.0 / 16;
                data[((y + 1) * width + x) * 4 + 2] += error * 5.0 / 16;
            }
            if (x < width - 1 && y < height - 1) {
                data[((y + 1) * width + x + 1) * 4] += error * 1.0 / 16;
                data[((y + 1) * width + x + 1) * 4 + 1] += error * 1.0 / 16;
                data[((y + 1) * width + x + 1) * 4 + 2] += error * 1.0 / 16;
            }
        }
    }

    // Clean up
    return true;
}// Dither_FS


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image while conserving the average brightness.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Bright()
{
    // If there is no image data, return false
    if (!data) {
        return false;
    }

        // Convert the image to grayscale
    if (!To_Grayscale()) {
        return false;
    }

    // Compute the average brightness of the image
    float avgBrightness = 0.0f;
    for (int i = 0; i < width * height; i++) {
        avgBrightness += static_cast<float>(data[i * 4]);
    }
    avgBrightness /= static_cast<float>(width * height);

    // Compute the threshold for threshold dithering
    int threshold = static_cast<int>(avgBrightness);

    int numPixels = width * height;

    for (int i = 0; i < numPixels; i++) {
        // Extract the RGB values from the pixel data
        unsigned char* pixel = data + i * 4;

        // Threshold dithering
        if (pixel[0] < threshold) {
            // Set the pixel to black
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else {
            // Set the pixel to white
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    return true;
}// Dither_Bright


///////////////////////////////////////////////////////////////////////////////
//
//      Perform clustered differing of the image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Cluster()
{
    // If there is no image data, return false
    if (!data) {
        return false;
    }

        // Convert the image to grayscale
    if (!To_Grayscale()) {
        return false;
    }

    // Define the dither matrix
    double dither_matrix[4][4] = {
        {0.7059, 0.3529, 0.5882, 0.2353},
        {0.0588, 0.9412, 0.8235, 0.4118},
        {0.4706, 0.7647, 0.8824, 0.1176},
        {0.1765, 0.5294, 0.2941, 0.6471}
    };

    // Loop through each pixel and apply the dither matrix
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double threshold = dither_matrix[x%4][y%4];
            int index = y*width*4 + x*4;

            if (data[index] >= threshold*255.0) {
                // Set pixel to white
                data[index] = 255;
                data[index+1] = 255;
                data[index+2] = 255;
            } else {
                // Set pixel to black
                data[index] = 0;
                data[index+1] = 0;
                data[index+2] = 0;
            }
        }
    }
    
    return true;
}// Dither_Cluster


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using Floyd-Steinberg dithering over
//  a uniform quantization - the same quantization as in Quant_Uniform.
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Color()
{
    // If there is no image data, return false
    if (!data) {
        return false;
    }

    // Define the color table corresponding to uniform quantization
    const int color_table_size = 256;
    unsigned char color_table[color_table_size][3];
    int color_index = 0;
    for (int r = 0; r <= 255; r += 36) {
        for (int g = 0; g <= 255; g += 36) {
            for (int b = 0; b <= 255; b += 85) {
                color_table[color_index][0] = r;
                color_table[color_index][1] = g;
                color_table[color_index][2] = b;
                color_index++;
            }
        }
    }
    
        // Loop through each pixel and apply Floyd-Steinberg dithering
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y*width*4 + x*4;
            int old_r = data[index];
            int old_g = data[index+1];
            int old_b = data[index+2];
            
            // Find the closest color in the color table
            int best_color = 0;
            int best_distance = 256*256*4;
            for (int i = 0; i < color_table_size; i++) {
                int r_diff = old_r - color_table[i][0];
                int g_diff = old_g - color_table[i][1];
                int b_diff = old_b - color_table[i][2];
                int distance = r_diff*r_diff + g_diff*g_diff + b_diff*b_diff;
                if (distance < best_distance) {
                    best_color = i;
                    best_distance = distance;
                }
            }
            
            // Set the pixel to the closest color
            data[index] = color_table[best_color][0];
            data[index+1] = color_table[best_color][1];
            data[index+2] = color_table[best_color][2];
            
            // Calculate the error
            int error_r = old_r - data[index];
            int error_g = old_g - data[index+1];
            int error_b = old_b - data[index+2];
            
            // Distribute the error to neighboring pixels
            if (x < width-1) {
                // Right neighbor
                int neighbor_index = index + 4;
                data[neighbor_index] = std::min(255, std::max(0, data[neighbor_index] + (7*error_r)/16));
                data[neighbor_index+1] = std::min(255, std::max(0, data[neighbor_index+1] + (7*error_g)/16));
                data[neighbor_index+2] = std::min(255, std::max(0, data[neighbor_index+2] + (7*error_b)/16));
            } if (x > 0 && y < height - 1) {
                // Bottom-left neighbor
                int neighbor_index = index + width*4 - 4;
                data[neighbor_index] = std::min(255, std::max(0, data[neighbor_index] + (3*error_r)/16));
                data[neighbor_index+1] = std::min(255, std::max(0, data[neighbor_index+1] + (3*error_r)/16));
                data[neighbor_index+2] = std::min(255, std::max(0, data[neighbor_index+2] + (3*error_r)/16));
            }
            if (y < height - 1) {
                // Bottom neighbor
                int neighbor_index = index + width*4;
                data[neighbor_index] = std::min(255, std::max(0, data[neighbor_index] + (5*error_r)/16));
                data[neighbor_index+1] = std::min(255, std::max(0, data[neighbor_index+1] + (5*error_r)/16));
                data[neighbor_index+2] = std::min(255, std::max(0, data[neighbor_index+2] + (5*error_r)/16));
            }
            if (x < width - 1 && y < height - 1) {
                // Bottom right neighbor
                int neighbor_index = index + width*4 + 4;
                data[neighbor_index] = std::min(255, std::max(0, data[neighbor_index] + (1*error_r)/16));
                data[neighbor_index+1] = std::min(255, std::max(0, data[neighbor_index+1] + (1*error_r)/16));
                data[neighbor_index+2] = std::min(255, std::max(0, data[neighbor_index+2] + (1*error_r)/16));
            }
        }
    }
    
    return true;
}// Dither_Color


///////////////////////////////////////////////////////////////////////////////
//
//      Composite the current image over the given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Over(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout <<  "Comp_Over: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Over


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "in" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_In(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_In: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_In


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "out" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Out(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Out: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Out


///////////////////////////////////////////////////////////////////////////////
//
//      Composite current image "atop" given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Atop(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Atop: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Atop


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image with given image using exclusive or (XOR).  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Xor(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Xor: Images not the same size\n";
        return false;
    }

    ClearToBlack();
    return false;
}// Comp_Xor


///////////////////////////////////////////////////////////////////////////////
//
//      Calculate the difference bewteen this imag and the given one.  Image 
//  dimensions must be equal.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Difference(TargaImage* pImage)
{
    if (!pImage)
        return false;

    if (width != pImage->width || height != pImage->height)
    {
        cout << "Difference: Images not the same size\n";
        return false;
    }// if

    for (int i = 0 ; i < width * height * 4 ; i += 4)
    {
        unsigned char        rgb1[3];
        unsigned char        rgb2[3];

        RGBA_To_RGB(data + i, rgb1);
        RGBA_To_RGB(pImage->data + i, rgb2);

        data[i] = abs(rgb1[0] - rgb2[0]);
        data[i+1] = abs(rgb1[1] - rgb2[1]);
        data[i+2] = abs(rgb1[2] - rgb2[2]);
        data[i+3] = 255;
    }

    return true;
}// Difference


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 box filter on this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Box()
{
     // If there is no image data, return false
    if (!data) {
        return false;
    }

    // Define the kernel for the box filter
    float kernel[25] = {
        1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f,
        1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f,
        1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f,
        1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f,
        1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f, 1.0f / 25.0f
    };

    // Create a temporary copy of the image data
    unsigned char* temp_data = new unsigned char[width * height * 4];
    memcpy(temp_data, data, width * height * 4);

    // Apply the box filter to the image data
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;

            // Apply the kernel to the current pixel and its neighbors
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int index = ((y + ky) * width + (x + kx)) * 4;
                    r += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index];
                    g += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 1];
                    b += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 2];
                }
            }

            // Update the pixel value in the original image data
            int index = (y * width + x) * 4;
            data[index] = (unsigned char)r;
            data[index + 1] = (unsigned char)g;
            data[index + 2] = (unsigned char)b;
        }
    }

    // Free the temporary image data
    delete[] temp_data;

    return true;
}// Filter_Box


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Bartlett filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Bartlett()
{
     // If there is no image data, return false
    if (!data) {
        return false;
    }

    // Define the kernel for the Bartlett filter
    float kernel[25] = {
        1, 2, 3, 2, 1,
        2, 4, 6, 4, 2,
        3, 6, 9, 6, 3,
        2, 4, 6, 4, 2,
        1, 2, 3, 2, 1
    };

    // Normalize the kernel
    float kernel_sum = 0.0f;
    for (int i = 0; i < 25; i++) {
        kernel_sum += kernel[i];
    }
    for (int i = 0; i < 25; i++) {
        kernel[i] /= kernel_sum;
    }

    // Create a temporary copy of the image data
    unsigned char* temp_data = new unsigned char[width * height * 4];
    memcpy(temp_data, data, width * height * 4);

    // Apply the box filter to the image data
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;

            // Apply the kernel to the current pixel and its neighbors
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int index = ((y + ky) * width + (x + kx)) * 4;
                    r += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index];
                    g += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 1];
                    b += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 2];
                }
            }

            // Update the pixel value in the original image data
            int index = (y * width + x) * 4;
            data[index] = (unsigned char)r;
            data[index + 1] = (unsigned char)g;
            data[index + 2] = (unsigned char)b;
        }
    }

    // Free the temporary image data
    delete[] temp_data;

    return true;
}// Filter_Bartlett


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Gaussian()
{
    // If there is no image data, return false
    if (!data) {
        return false;
    }

    // Define the kernel for the Gaussian filter
    float kernel[25] = {
        1, 4, 7, 4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1, 4, 7, 4, 1
    };

    // Normalize the kernel
    float kernel_sum = 0.0f;
    for (int i = 0; i < 25; i++) {
        kernel_sum += kernel[i];
    }
    for (int i = 0; i < 25; i++) {
        kernel[i] /= kernel_sum;
    }

    // Create a temporary copy of the image data
    unsigned char* temp_data = new unsigned char[width * height * 4];
    memcpy(temp_data, data, width * height * 4);

    
    // Apply the box filter to the image data
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;

            // Apply the kernel to the current pixel and its neighbors
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int index = ((y + ky) * width + (x + kx)) * 4;
                    r += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index];
                    g += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 1];
                    b += kernel[(ky + 2) * 5 + (kx + 2)] * temp_data[index + 2];
                }
            }

            // Update the pixel value in the original image data
            int index = (y * width + x) * 4;
            data[index] = (unsigned char)r;
            data[index + 1] = (unsigned char)g;
            data[index + 2] = (unsigned char)b;
        }
    }

    return true;
}// Filter_Gaussian

///////////////////////////////////////////////////////////////////////////////
//
//      Perform NxN Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Gaussian_N( unsigned int N )
{
        // If there is no image data or N is even, return false
    if (!data || N % 2 == 0) {
        return false;
    }

    // Allocate memory for the kernel
    float* kernel = new float[N * N];

    // Calculate the binomial coefficients and filter values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int k = abs(i - (static_cast<int>(N) / 2));
            int l = abs(j - (static_cast<int>(N) / 2));
            float coeff = 1.0f;
            for (int m = 1; m <= k; m++) {
                coeff *= (float)(N - m) / m;
            }
            for (int m = 1; m <= l; m++) {
                coeff *= (float)(N - m) / m;
            }
            kernel[i * N + j] = coeff * pow(0.5f, k + l + 2);
        }
    }

    // Normalize the kernel
    float kernel_sum = 0.0f;
    for (int i = 0; i < N * N; i++) {
        kernel_sum += kernel[i];
    }
    for (int i = 0; i < N * N; i++) {
        kernel[i] /= kernel_sum;
    }

    // Create a temporary copy of the image data
    unsigned char* temp_data = new unsigned char[width * height * 4];
    memcpy(temp_data, data, width * height * 4);

    // Apply the Gaussian filter to the image data
    int offset = N / 2;
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    int index = ((y + i) * width + (x + j)) * 4;
                    r += kernel[(i + offset) * N + (j + offset)] * temp_data[index];
                    g += kernel[(i + offset) * N + (j + offset)] * temp_data[index + 1];
                    b += kernel[(i + offset) * N + (j + offset)] * temp_data[index + 2];
                }
            }
            int index = (y * width + x) * 4;
            data[index] = (unsigned char)r;
            data[index + 1] = (unsigned char)g;
            data[index + 2] = (unsigned char)b;
        }
    }

    // Free the memory used by the kernel and temporary data
    delete[] kernel;
    delete[] temp_data;

    return true;
}// Filter_Gaussian_N


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 edge detect (high pass) filter on this image.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Edge()
{
    ClearToBlack();
    return false;
}// Filter_Edge


///////////////////////////////////////////////////////////////////////////////
//
//      Perform a 5x5 enhancement filter to this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Enhance()
{
    ClearToBlack();
    return false;
}// Filter_Enhance


///////////////////////////////////////////////////////////////////////////////
//
//      Run simplified version of Hertzmann's painterly image filter.
//      You probably will want to use the Draw_Stroke funciton and the
//      Stroke class to help.
// Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::NPR_Paint()
{
    ClearToBlack();
    return false;
}



///////////////////////////////////////////////////////////////////////////////
//
//      Halve the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Half_Size()
{
    ClearToBlack();
    return false;
}// Half_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Double the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Double_Size()
{
    ClearToBlack();
    return false;
}// Double_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Scale the image dimensions by the given factor.  The given factor is 
//  assumed to be greater than one.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Resize(float scale)
{
    ClearToBlack();
    return false;
}// Resize


//////////////////////////////////////////////////////////////////////////////
//
//      Rotate the image clockwise by the given angle.  Do not resize the 
//  image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Rotate(float angleDegrees)
{
    ClearToBlack();
    return false;
}// Rotate


//////////////////////////////////////////////////////////////////////////////
//
//      Given a single RGBA pixel return, via the second argument, the RGB
//      equivalent composited with a black background.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::RGBA_To_RGB(unsigned char *rgba, unsigned char *rgb)
{
    const unsigned char	BACKGROUND[3] = { 0, 0, 0 };

    unsigned char  alpha = rgba[3];

    if (alpha == 0)
    {
        rgb[0] = BACKGROUND[0];
        rgb[1] = BACKGROUND[1];
        rgb[2] = BACKGROUND[2];
    }
    else
    {
	    float	alpha_scale = (float)255 / (float)alpha;
	    int	val;
	    int	i;

	    for (i = 0 ; i < 3 ; i++)
	    {
	        val = (int)floor(rgba[i] * alpha_scale);
	        if (val < 0)
		    rgb[i] = 0;
	        else if (val > 255)
		    rgb[i] = 255;
	        else
		    rgb[i] = val;
	    }
    }
}// RGA_To_RGB


///////////////////////////////////////////////////////////////////////////////
//
//      Copy this into a new image, reversing the rows as it goes. A pointer
//  to the new image is returned.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Reverse_Rows(void)
{
    unsigned char   *dest = new unsigned char[width * height * 4];
    TargaImage	    *result;
    int 	        i, j;

    if (! data)
    	return NULL;

    for (i = 0 ; i < height ; i++)
    {
	    int in_offset = (height - i - 1) * width * 4;
	    int out_offset = i * width * 4;

	    for (j = 0 ; j < width ; j++)
        {
	        dest[out_offset + j * 4] = data[in_offset + j * 4];
	        dest[out_offset + j * 4 + 1] = data[in_offset + j * 4 + 1];
	        dest[out_offset + j * 4 + 2] = data[in_offset + j * 4 + 2];
	        dest[out_offset + j * 4 + 3] = data[in_offset + j * 4 + 3];
        }
    }

    result = new TargaImage(width, height, dest);
    delete[] dest;
    return result;
}// Reverse_Rows


///////////////////////////////////////////////////////////////////////////////
//
//      Clear the image to all black.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::ClearToBlack()
{
    memset(data, 0, width * height * 4);
}// ClearToBlack


///////////////////////////////////////////////////////////////////////////////
//
//      Helper function for the painterly filter; paint a stroke at
// the given location
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::Paint_Stroke(const Stroke& s) {
   int radius_squared = (int)s.radius * (int)s.radius;
   for (int x_off = -((int)s.radius); x_off <= (int)s.radius; x_off++) {
      for (int y_off = -((int)s.radius); y_off <= (int)s.radius; y_off++) {
         int x_loc = (int)s.x + x_off;
         int y_loc = (int)s.y + y_off;
         // are we inside the circle, and inside the image?
         if ((x_loc >= 0 && x_loc < width && y_loc >= 0 && y_loc < height)) {
            int dist_squared = x_off * x_off + y_off * y_off;
            if (dist_squared <= radius_squared) {
               data[(y_loc * width + x_loc) * 4 + 0] = s.r;
               data[(y_loc * width + x_loc) * 4 + 1] = s.g;
               data[(y_loc * width + x_loc) * 4 + 2] = s.b;
               data[(y_loc * width + x_loc) * 4 + 3] = s.a;
            } else if (dist_squared == radius_squared + 1) {
               data[(y_loc * width + x_loc) * 4 + 0] = 
                  (data[(y_loc * width + x_loc) * 4 + 0] + s.r) / 2;
               data[(y_loc * width + x_loc) * 4 + 1] = 
                  (data[(y_loc * width + x_loc) * 4 + 1] + s.g) / 2;
               data[(y_loc * width + x_loc) * 4 + 2] = 
                  (data[(y_loc * width + x_loc) * 4 + 2] + s.b) / 2;
               data[(y_loc * width + x_loc) * 4 + 3] = 
                  (data[(y_loc * width + x_loc) * 4 + 3] + s.a) / 2;
            }
         }
      }
   }
}

// Generic uniform quantization
void TargaImage::Quant_Uniform_To_N_Shades(const int num_red_shades, const int num_green_shades, const int num_blue_shades)
{
    // Convert the image to RGB format
    unsigned char* rgb_data = To_RGB();

    // Calculate the number of pixels in the image
    int num_pixels = width * height;

    // Loop over each pixel in the image
    for (int i = 0; i < num_pixels; i++) {
        // Calculate the index of the current pixel
        int index = i * 3;

        // Quantize the red channel
        int red = rgb_data[index];
        int red_quantized = (int)((red / 255.0f) * num_red_shades) * (255 / num_red_shades);

        // Quantize the green channel
        int green = rgb_data[index + 1];
        int green_quantized = (int)((green / 255.0f) * num_green_shades) * (255 / num_green_shades);

        // Quantize the blue channel
        int blue = rgb_data[index + 2];
        int blue_quantized = (int)((blue / 255.0f) * num_blue_shades) * (255 / num_blue_shades);

        // Set the quantized RGB values for the current pixel
        rgb_data[index] = red_quantized;
        rgb_data[index + 1] = green_quantized;
        rgb_data[index + 2] = blue_quantized;
    }

    // Convert the image back to pre-multiplied RGBA format
    unsigned char* rgba_data = new unsigned char[num_pixels * 4];
    for (int i = 0; i < num_pixels; i++) {
        int rgba_index = i * 4;
        int rgb_index = i * 3;
        rgba_data[rgba_index] = rgb_data[rgb_index];
        rgba_data[rgba_index + 1] = rgb_data[rgb_index + 1];
        rgba_data[rgba_index + 2] = rgb_data[rgb_index + 2];
        rgba_data[rgba_index + 3] = 255; // alpha channel set to fully opaque
    }

    // Replace the current pixel data with the quantized pixel data
    delete[] data;
    data = rgba_data;

    // Clean up
    delete[] rgb_data;
}

///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke() {}

///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke(unsigned int iradius, unsigned int ix, unsigned int iy,
               unsigned char ir, unsigned char ig, unsigned char ib, unsigned char ia) :
   radius(iradius),x(ix),y(iy),r(ir),g(ig),b(ib),a(ia)
{
}