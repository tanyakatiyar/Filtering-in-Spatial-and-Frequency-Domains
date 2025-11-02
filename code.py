# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    kernel = np.ones((k, k), dtype=float)  # create k×k matrix of ones
    total = kernel.sum()        # Calculate the sum of all elements in the kernel
    kernel = kernel /total
    return kernel


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    axis_values = np.linspace(-(k // 2), k // 2, k)
    x_coordinates, y_coordinates = np.meshgrid(axis_values, axis_values)
    gaussian_kernel = np.exp(-(x_coordinates**2 + y_coordinates**2) / (2. * sigma**2))
    total =  gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel / total
    return gaussian_kernel


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    img_height, img_width = img.shape
    kernel_height, kernel_width = h.shape
    flipped_kernel = np.flipud(np.fliplr(h)) # np.flipud flips up-down, np.fliplr flips left-right.
    pad_height = kernel_height // 2 #// performs floor division, which means it divides and then rounds down to the nearest integer
    pad_width = kernel_width // 2
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    convolved_img = np.zeros_like(img, dtype=float)
    for i in range(img_height):
        for j in range(img_width):
            current_region = padded_img[i:i+kernel_height, j:j+kernel_width] 
            convolved_img[i, j] = np.sum(current_region * flipped_kernel)

    return convolved_img        

def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    img_shape = img.shape
    kernel_shape = h.shape
    conv_shape = (img_shape[0] + kernel_shape[0] - 1,
              img_shape[1] + kernel_shape[1] - 1) #The output size of linear convolution is the sum of the input sizes minus 1 in each dimension
    pad_img = np.pad(img, ((0, conv_shape[0] - img_shape[0]),
                       (0, conv_shape[1] - img_shape[1])), mode='constant')
    pad_kernel = np.pad(h, ((0, conv_shape[0] - kernel_shape[0]),
                       (0, conv_shape[1] - kernel_shape[1])), mode='constant') #To multiply inputs in frequency domain, kernel must be padded with zeros to this exact size as well.
    fft_img = np.fft.fft2(pad_img)
    fft_kernel = np.fft.fft2(pad_kernel)
    fft_product = fft_img * fft_kernel
    conv_result = np.fft.ifft2(fft_product)

    # Crop to original image size to match spatial domain output
    start_x = (conv_result.shape[0] - img_shape[0]) // 2
    start_y = (conv_result.shape[1] - img_shape[1]) // 2
    conv_result_cropped = conv_result[start_x:start_x + img_shape[0], start_y:start_y + img_shape[1]]

    return np.real(conv_result_cropped) #Return only the real part of the convolution result. The imaginary part appears due to numerical errors in FFT computations and should be close to zero.


def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    absolute_diff = np.abs(a - b)
    mad_value = np.mean(absolute_diff) # Calculate mean over all pixels
    return mad_value

# ==========================================================

# TODO: 1. Load the grayscale image (e.g., lena.png)
img = cv2.imread('data/lena.png', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
# TODO: 2. Construct 9×9 box and Gaussian kernels (same sigma)
box_kernel = make_box_kernel(9)
gauss_kernel = make_gauss_kernel(9, 1.5) # σ≈1.0 to 2.0 for typical grayscale images to ensure effective noise reduction while avoiding excessive blur.
# TODO: 3. Apply both filters spatially (manual convolution)
img_box_spatial = conv2_same_zero(img, box_kernel)
img_gauss_spatial = conv2_same_zero(img, gauss_kernel)
# TODO: 4. Apply both filters in the frequency domain
img_box_freq = freq_linear_conv(img, box_kernel)
img_gauss_freq = freq_linear_conv(img, gauss_kernel)
# TODO: 5. Compute and print MAD between spatial and frequency outputs
mad_box = compute_mad(img_box_spatial, img_box_freq)
mad_gauss = compute_mad(img_gauss_spatial, img_gauss_freq)
print(f'MAD Box Filter: {mad_box:.2e}')
print(f'MAD Gaussian Filter: {mad_gauss:.2e}')
# TODO: 6. Visualize all results (original, box/gaussian spatial, box/gaussian frequency, spectrum)
fig, axs = plt.subplots(2, 5, figsize=(20, 8))

axs[0,0].imshow(img, cmap='gray')
axs[0,0].set_title('Original')
axs[0,0].axis('off')

axs[0,1].imshow(img_box_spatial, cmap='gray')
axs[0,1].set_title('Box Spatial')
axs[0,1].axis('off')

axs[0,2].imshow(img_box_freq, cmap='gray')
axs[0,2].set_title('Box Frequency')
axs[0,2].axis('off')

axs[0,3].imshow(img_gauss_spatial, cmap='gray')
axs[0,3].set_title('Gauss Spatial')
axs[0,3].axis('off')

axs[0,4].imshow(img_gauss_freq, cmap='gray')
axs[0,4].set_title('Gauss Frequency')
axs[0,4].axis('off')

# Row 1: Spectra
axs[1,0].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)))+1), cmap='gray')
axs[1,0].set_title('Spectrum Original')
axs[1,0].axis('off')

axs[1,1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_box_spatial)))+1), cmap='gray')
axs[1,1].set_title('Spectrum Box Spatial')
axs[1,1].axis('off')

axs[1,2].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_box_freq)))+1), cmap='gray')
axs[1,2].set_title('Spectrum Box Frequency')
axs[1,2].axis('off')

axs[1,3].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gauss_spatial)))+1), cmap='gray')
axs[1,3].set_title('Spectrum Gauss Spatial')
axs[1,3].axis('off')

axs[1,4].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gauss_freq)))+1), cmap='gray')
axs[1,4].set_title('Spectrum Gauss Frequency')
axs[1,4].axis('off')

plt.tight_layout()
plt.show()

# TODO: 7. Verify that MAD < 1×10⁻⁷ for both filters
assert mad_box < 1e-7, "MAD for Box filter exceeds threshold!"
print("MAD check passed for box filter.")
assert mad_gauss < 1e-7, "MAD for Gaussian filter exceeds threshold!"
print("MAD check passed for Gaussian filter.")
