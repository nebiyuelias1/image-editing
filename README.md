# Project 1: An Image Editing Program

In this project I wrote an image editing program that allows you to load in one or more images and
perform various operations on them. Consider it to be a miniature Photoshop.

The operations are summarized here:

|Categories|Methods|Description|Implemented|
|----------|-------|-----------|-----------|
|Transform |To_Grayscale() |Convert image to grayscale|✅         |
|Quantized|Quant_Uniform()|Convert the image to an 8 bit image using uniform quantization|✅         |
|          |Quant_Populosity()|Convert the image to an 8 bit image using populosity quantization| ❌|
|Dithering|Dither_Threshold()|Dither an image to black and white using threshold dithering with a threshold of 0.5|✅