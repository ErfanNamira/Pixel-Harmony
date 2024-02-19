# Pixel Harmony v0.0.2
# ErfanNamira
# https://github.com/ErfanNamira/Pixel-Harmony

# SSIM (Structural Similarity Index): SSIM is a metric used to measure the similarity between two images. It considers not only the luminance (brightness) differences but also the structural information (textures, edges) present in the images. SSIM produces a value between -1 and 1, where 1 indicates perfect similarity between the images.

# PSNR (Peak Signal-to-Noise Ratio): PSNR is a metric commonly used to quantify the quality difference between an original image and a compressed or distorted version of that image. It measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. PSNR is expressed in decibels (dB), and a higher PSNR value indicates better image quality.

# pip install opencv-python scikit-image

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim

def absolute_error(image1, image2):
    return np.abs(image1.astype(np.float64) - image2.astype(np.float64))

def mean_absolute_error(image1, image2):
    ae = absolute_error(image1, image2)
    return np.mean(ae)

def normalized_cross_correlation(image1, image2):
    return cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)[0][0]

def root_mean_squared_error(image1, image2):
    return np.sqrt(np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2))

def ssim_description(ssim_score):
    if ssim_score >= 0.98:
        return "The images are nearly identical, indicating an extremely high level of similarity."
    elif ssim_score >= 0.95:
        return "The images are very similar, with minor differences."
    elif ssim_score >= 0.90:
        return "The images are moderately similar, with noticeable but minor differences."
    elif ssim_score >= 0.85:
        return "The images have some similarity, although there are noticeable differences."
    else:
        return "The images have low similarity, indicating significant differences."

def psnr_description(psnr_score):
    if psnr_score >= 50:
        return "The images have very high quality, with minimal loss of information."
    elif psnr_score >= 40:
        return "The images have high quality, with minor loss of information."
    elif psnr_score >= 30:
        return "The images have moderate quality, with some loss of information."
    elif psnr_score >= 20:
        return "The images have fair quality, with noticeable loss of information."
    else:
        return "The images have poor quality, with significant loss of information."

def histogram_comparison(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    elif width is None:
        aspect_ratio = height / float(image.shape[0])
        new_width = int(image.shape[1] * aspect_ratio)
        return cv2.resize(image, (new_width, height))
    elif height is None:
        aspect_ratio = width / float(image.shape[1])
        new_height = int(image.shape[0] * aspect_ratio)
        return cv2.resize(image, (width, new_height))
    else:
        return cv2.resize(image, (width, height))

def compare_two_images():
    # Prompt the user for the paths to the two images
    image1_path = input("Enter the path to the first image: ")
    image2_path = input("Enter the path to the second image: ")

    # Load the two frames
    frame1 = cv2.imread(image1_path)
    frame2 = cv2.imread(image2_path)

    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Resize the images to have the same dimensions
    max_height = max(gray_frame1.shape[0], gray_frame2.shape[0])
    max_width = max(gray_frame1.shape[1], gray_frame2.shape[1])
    gray_frame1 = resize_image(gray_frame1, width=max_width, height=max_height)
    gray_frame2 = resize_image(gray_frame2, width=max_width, height=max_height)

    # Calculate SSIM
    ssim_score, _ = compare_ssim(gray_frame1, gray_frame2, full=True)

    # Calculate PSNR
    mse = np.mean((gray_frame1 - gray_frame2) ** 2)
    if mse == 0:
        psnr_score = float('inf')
    else:
        max_pixel = 255.0
        psnr_score = 20 * np.log10(max_pixel / np.sqrt(mse))

    # Calculate AE, MAE, NCC, RMSE
    ae = absolute_error(gray_frame1, gray_frame2)
    mae = mean_absolute_error(gray_frame1, gray_frame2)
    ncc = normalized_cross_correlation(gray_frame1, gray_frame2)
    rmse = root_mean_squared_error(gray_frame1, gray_frame2)

    # Calculate histogram comparison
    hist_comp = histogram_comparison(gray_frame1, gray_frame2)

    # Print the SSIM, PSNR, AE, MAE, NCC, RMSE, and histogram comparison scores with descriptions
    print(f"SSIM Score: {ssim_score}")
    print(f"Description: {ssim_description(ssim_score)}\n")

    print(f"PSNR Score: {psnr_score} dB")
    print(f"Description: {psnr_description(psnr_score)}\n")

    print("Absolute Error:")
    print(np.array2string(ae, separator=', ', prefix=' ', max_line_width=np.inf) + '\n')

    print(f"Mean Absolute Error: {mae}")
    print(f"Normalized Cross-Correlation: {ncc}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Histogram Comparison: {hist_comp}\n")

    # Compare the SSIM and PSNR scores to determine the image with higher quality
    if ssim_score > 0.5 and psnr_score > 30:
        print("Both images exhibit good quality based on SSIM (>= 0.5) and PSNR (> 30 dB).")
    elif ssim_score > 0.5:
        print("The first image demonstrates higher quality based on SSIM (> 0.5).")
    elif psnr_score > 30:
        print("The second image showcases higher quality based on PSNR (> 30 dB).")
    else:
        print("Both images have poor quality according to SSIM (< 0.5) and PSNR (< 30 dB).")

def compare_images_in_directory(directory):
    # Code to compare images in a directory
    pass

def batch_process_images(directory):
    # Code to batch process images in a directory
    pass

# Main function to handle user interactions
def main():
    print("Pixel Harmony v0.0.1")
    while True:
        print("1. Compare two images")
        print("2. Compare images in a directory")
        print("3. Batch process images in a directory")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            compare_two_images()
        elif choice == '2':
            directory = input("Enter the directory containing two images: ")
            compare_images_in_directory(directory)
        elif choice == '3':
            directory = input("Enter the directory containing images to batch process: ")
            batch_process_images(directory)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
