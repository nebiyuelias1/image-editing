///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.h                            Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//
//      Class to manipulate targa images.  You must implement the image 
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _TARGA_IMAGE_H_
#define _TARGA_IMAGE_H_

#include <FL/Fl.H>
#include <FL/Fl_Widget.H>
#include <stdio.h>

class Stroke;
class DistanceImage;

class TargaImage
{
    // methods
    public:
	    TargaImage(void);
            TargaImage(int w, int h);
	    TargaImage(int w, int h, unsigned char *d);
            TargaImage(const TargaImage& image);
	    ~TargaImage(void);

        unsigned char*	To_RGB(void);	            // Convert the image to RGB format,
        bool Save_Image(const char*);               // save the image to a file
        static TargaImage* Load_Image(char*);       // Load a file and return a pointer to a new TargaImage object.  Returns NULL on failure

        bool To_Grayscale();

        bool Quant_Uniform();
        bool Quant_Populosity();
        bool Quant_Median();

        bool Dither_Threshold();
        bool Dither_Random();
        bool Dither_FS();
        bool Dither_Bright();
        bool Dither_Cluster();
        bool Dither_Color();

        bool Comp_Over(TargaImage* pImage);
        bool Comp_In(TargaImage* pImage);
        bool Comp_Out(TargaImage* pImage);
        bool Comp_Atop(TargaImage* pImage);
        bool Comp_Xor(TargaImage* pImage);

        bool Difference(TargaImage* pImage);

        bool Filter_Box();
        bool Filter_Bartlett();
        bool Filter_Gaussian();
        bool Filter_Gaussian_N(unsigned int N);
        bool Filter_Edge();
        bool Filter_Enhance();

        bool NPR_Paint();

        bool Half_Size();
        bool Double_Size();
        bool Resize(float scale);
        bool Rotate(float angleDegrees);

    private:
	// helper function for format conversion
        void RGBA_To_RGB(unsigned char *rgba, unsigned char *rgb);

        // reverse the rows of the image, some targas are stored bottom to top
	TargaImage* Reverse_Rows(void);

	// clear image to all black
        void ClearToBlack();

	// Draws a filled circle according to the stroke data
        void Paint_Stroke(const Stroke& s);
        
        // Use uniform quantization to convert image to r number shades, g number shades, b snumber shades
        void Quant_Uniform_To_N_Shades(const int r, const int g, const int b);

    // members
    public:
        int		width;	    // width of the image in pixels
        int		height;	    // height of the image in pixels
        unsigned char	*data;	    // pixel data for the image, assumed to be in pre-multiplied RGBA format.

};

class Stroke { // Data structure for holding painterly strokes.
public:
   Stroke(void);
   Stroke(unsigned int radius, unsigned int x, unsigned int y,
          unsigned char r, unsigned char g, unsigned char b, unsigned char a);
   
   // data
   unsigned int radius, x, y;	// Location for the stroke
   unsigned char r, g, b, a;	// Color
};


#endif

#ifndef _RGB_COLOR_H_
#define _RGB_COLOR_H_

/**
 * @brief The RGBColor class represents a color in the RGB color model.
 *
 * The RGBColor class provides a convenient way to represent colors using
 * red, green, and blue color channels. Each color channel is represented
 * as an unsigned integer value in the range [0, 255], where 0 is the
 * darkest intensity and 255 is the brightest intensity.
 *
 * The class provides a default constructor that creates a black color,
 * as well as a constructor that allows specifying the red, green, and blue
 * color channel values. The class also provides public getter and setter
 * methods for each color channel.
 */
class RGBColor {
public:
    /**
     * @brief Default constructor that creates a black color.
     */
    RGBColor() : r(0), g(0), b(0) {}

    /**
     * @brief Constructor that creates a color with the specified RGB values.
     *
     * @param red The red color channel value, in the range [0, 255].
     * @param green The green color channel value, in the range [0, 255].
     * @param blue The blue color channel value, in the range [0, 255].
     */
    RGBColor(unsigned int red, unsigned int green, unsigned int blue)
        : r(red), g(green), b(blue) {}

    /**
     * @brief Getter method for the red color channel value.
     *
     * @return The red color channel value, in the range [0, 255].
     */
    unsigned int getRed() const { return r; }

    /**
     * @brief Getter method for the green color channel value.
     *
     * @return The green color channel value, in the range [0, 255].
     */
    unsigned int getGreen() const { return g; }

    /**
     * @brief Getter method for the blue color channel value.
     *
     * @return The blue color channel value, in the range [0, 255].
     */
    unsigned int getBlue() const { return b; }

    /**
     * @brief Setter method for the red color channel value.
     *
     * @param red The new red color channel value, in the range [0, 255].
     */
    void setRed(unsigned int red) { r = red; }

    /**
     * @brief Setter method for the green color channel value.
     *
     * @param green The new green color channel value, in the range [0, 255].
     */
    void setGreen(unsigned int green) { g = green; }

    /**
     * @brief Setter method for the blue color channel value.
     *
     * @param blue The new blue color channel value, in the range [0, 255].
     */
    void setBlue(unsigned int blue) { b = blue; }

    /**
     * @brief Returns the squared Euclidean distance between this color and another color.
     * 
     * @param other The other RGBColor you want to find the distance to
     * @return int 
     */
    int DistanceSquared(const RGBColor& other) const {
        int dr = static_cast<int>(getRed()) - static_cast<int>(other.getRed());
        int dg = static_cast<int>(getBlue()) - static_cast<int>(other.getGreen());
        int db = static_cast<int>(getBlue()) - static_cast<int>(other.getBlue());
        return dr*dr + dg*dg + db*db;
    }

    friend bool operator>(const RGBColor &o1, const RGBColor &other)
    {
        // Compare the total color values
        return (o1.r + o1.g + o1.b) > (other.r + other.g + other.b);
    }

    friend bool operator<(const RGBColor &o1, const RGBColor &other)
    {
        // Compare the total color values
        return (o1.r + o1.g + o1.b) < (other.r + other.g + other.b);
    }

    friend bool operator==(const RGBColor &c1, const RGBColor &c2)
    {
        return c1.r == c2.r && c1.g == c2.g && c1.b == c2.b;
    }

private:
    unsigned int r; ///< The red color channel value.
    unsigned int g; ///< The green color channel value.
    unsigned int b; ///< The blue color channel value.
};

#endif // _RGB_COLOR_H_

