# Pixel-Harmony 🖼️
Pixel Harmony is a Python script designed to analyze and compare the quality of two images using various metrics. It provides insights into the similarity and quality difference between the images by utilizing metrics such as SSIM, PSNR, Absolute Error, Mean Absolute Error, Normalized Cross-Correlation, Root Mean Squared Error, and Histogram Comparison.

## Metrics Overview
### Structural Similarity Index (SSIM) 📊
The Structural Similarity Index (SSIM) is a metric used to quantify the similarity between two images. It takes into account not only the global luminance (brightness) differences but also the local structural information (textures, edges) present in the images. SSIM produces a value between -1 and 1, where:

1 indicates perfect similarity between the images.

0 indicates no similarity between the images.

-1 indicates perfect dissimilarity between the images.

### Peak Signal-to-Noise Ratio (PSNR) 🔊
The Peak Signal-to-Noise Ratio (PSNR) is a metric commonly used to measure the quality difference between an original image and a compressed or distorted version of that image. It measures the ratio between the maximum possible power of a signal and the power of the noise that affects the fidelity of its representation. PSNR is expressed in decibels (dB), and a higher PSNR value indicates better image quality.

In essence, SSIM and PSNR complement each other in assessing image quality, with SSIM focusing on structural similarity and PSNR focusing on the amount of noise or distortion in the image.

### Absolute Error (AE) and Mean Absolute Error (MAE) 📉
Absolute Error (AE) measures the absolute difference between the pixel values of corresponding pixels in two images. It provides insight into the magnitude of the difference between the images at a pixel level. Mean Absolute Error (MAE) is the average of these absolute differences across all pixels, providing a single scalar value to quantify the overall difference between the images.

### Normalized Cross-Correlation (NCC) 🔄
Normalized Cross-Correlation (NCC) measures the similarity between two images by computing the normalized correlation coefficient between their pixel intensities. It evaluates the pattern correlation between the images, with higher scores indicating greater correlation and similarity.

### Root Mean Squared Error (RMSE) 📏
Root Mean Squared Error (RMSE) calculates the root mean squared difference between the pixel values of corresponding pixels in two images. It provides a measure of the average magnitude of the differences between the images, with lower values indicating higher similarity.

### Histogram Comparison 📊
Histogram Comparison calculates the correlation between the histograms of two images. It quantifies how similar the distributions of pixel intensities are between the images, with higher correlation values indicating greater similarity in pixel intensity distributions.

## Installation
```
pip install opencv-python scikit-image
```
## Usage
Save the two images you want to compare in the same directory as the script. (1.jpg & 2.jpg)

Run the script by executing python pixelharmony.py in the command line.

The script will calculate and display the SSIM and PSNR scores, along with additional metrics, for the provided images.

## Example Output
```
SSIM Score: 0.9606657154431885
Description: The images are nearly identical, indicating an extremely high level of similarity.
PSNR Score: 37.84524003455975 dB
Description: The images have moderate quality, with some loss of information.
Absolute Error: [[ 2.  1.  1. ...  6.  2.  1.]
 [ 5.  5.  5. ...  3.  5.  7.]
 [ 0.  1.  2. ...  1. 12.  2.]
 ...
 [ 2.  2.  2. ...  3.  1.  2.]
 [ 2.  2.  2. ...  5.  3.  0.]
 [ 2.  2.  2. ...  3.  1.  3.]]
Mean Absolute Error: 2.4401609757078506
Normalized Cross-Correlation: 0.9995338916778564
Root Mean Squared Error: 3.293665264711223
Histogram Comparison: 0.9994742033940714
```
## Contributions
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on GitHub.
