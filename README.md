# Project 1: An Image Editing Program

In this project I wrote an image editing program that allows you to load in one or more images and
perform various operations on them. Consider it to be a miniature Photoshop.

The operations are summarized here:

|Categories|Methods|Operation|Description|Implemented|
|----------|-------|---------|-----------|-----------|
|Transform |To_Grayscale()|gray|Convert image to grayscale|✅|
|Quantized|Quant_Uniform()|quant-unif|Convert the image to an 8 bit image using uniform quantization|✅         |
|          |Quant_Populosity()|quant-pop|Convert the image to an 8 bit image using populosity quantization| ✅|
|Dithering|Dither_Threshold()|dither-thresh|Dither an image to black and white using threshold dithering with a threshold of 0.5|✅|
||Dither_Random()|dither-rand|Dither an image to black and white using random dithering|✅|
||Dither_FS()|dither-fs|Dither an image to black and white using Floyd-Steinberg dithering|✅|
||Dither_Bright()|dither-bright|Dither an image to black and white using threshold dithering with a threshold chosen to keep the average brightness constant|✅|
||Dither_Cluster()|dither-cluster|Dither an image to black and white using cluster dithering|✅|
||Dither_Color()|dither-color|Dither an image to 8 bit color using Floyd-Steinberg dithering|✅|
|Filtering|Filter_Box()|filter-box|Apply a 5x5 box filter|✅|
||Filter_Bartlett()|filter-bartlett|Perform 5x5 Bartlett filter on this image|✅|
||Filter_Gaussian()|filter-gauss|Apply a 5x5 Gaussian filter|✅|
||Filter_Gaussian_N(N)|filter-gauss-n|Perform NxN Gaussian filter on this image|✅|
||Filter_Edge()|filter-edge|Edge detection filter|✅|
