import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

def load_image_as_gray(image_path):
    """
    Load an image from file and convert it to grayscale.
    """
    img = io.imread(image_path)
    if len(img.shape) == 3:  # color image
        img = color.rgb2gray(img)
    return img.astype(np.float64)

def dct2(block):
    """
    Compute 2D Discrete Cosine Transform of an 8x8 block.
    """
    return np.fft.fft2(block, norm='ortho')

def idct2(block_dct):
    """
    Compute the 2D Inverse Discrete Cosine Transform.
    """
    return np.fft.ifft2(block_dct, norm='ortho').real

def zigzag_indices(n=8):
    """
    Generate zigzag order of indices for an n x n block.
    Returns a list of (row, col) pairs in zigzag order.
    """
    # Simple approach for 8x8
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # even diagonal
            for x in range(s + 1):
                y = s - x
                if x < n and y < n:
                    indices.append((x, y))
        else:
            # odd diagonal
            for x in range(s, -1, -1):
                y = s - x
                if x < n and y < n:
                    indices.append((x, y))
    return indices

def partial_dct_reconstruction(img_gray, K=8, block_size=8):
    """
    Reconstruct the image by taking only K DCT coefficients (in zigzag order)
    for each 8x8 block.
    """
    rows, cols = img_gray.shape
    # Ensure image dimensions are multiples of block_size
    rows_cropped = (rows // block_size) * block_size
    cols_cropped = (cols // block_size) * block_size
    img_cropped = img_gray[:rows_cropped, :cols_cropped]

    # Prepare output
    recon_img = np.zeros_like(img_cropped)
    
    # Precompute zigzag ordering
    zz = zigzag_indices(block_size)
    
    # Process each 8x8 block
    for r in range(0, rows_cropped, block_size):
        for c in range(0, cols_cropped, block_size):
            block = img_cropped[r:r+block_size, c:c+block_size]
            
            # 2D DCT
            block_dct = dct2(block)
            
            # Zero out all but the first K coefficients in zigzag order
            mask = np.zeros((block_size, block_size), dtype=bool)
            for i in range(K):
                rr, cc = zz[i]
                mask[rr, cc] = True
            
            # Create a new DCT block with only the K coefficients
            block_dct_k = np.zeros_like(block_dct)
            block_dct_k[mask] = block_dct[mask]
            
            # Inverse DCT
            block_idct = idct2(block_dct_k)
            
            recon_img[r:r+block_size, c:c+block_size] = block_idct

    return recon_img

def psnr(original, reconstructed):
    """
    Compute Peak Signal-to-Noise Ratio.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    max_val = np.max(original)
    return 20 * np.log10(max_val / np.sqrt(mse))

def main():
    # Example usage on at least two images
    image_paths = [
        "sample_image/goldhill.png",
        "sample_image/barbara.png"
    ]
    
    K_values = [2, 4, 8, 16, 32]

    for path in image_paths:
        img_gray = load_image_as_gray(path)
        fig, axs = plt.subplots(1, len(K_values)+1, figsize=(15, 4))
        axs[0].imshow(img_gray, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

        for i, K in enumerate(K_values):
            recon_img = partial_dct_reconstruction(img_gray, K=K, block_size=8)
            psnr_val = psnr(img_gray[:recon_img.shape[0], :recon_img.shape[1]], recon_img)
            
            axs[i+1].imshow(recon_img, cmap='gray')
            axs[i+1].set_title(f"K={K}\nPSNR={psnr_val:.2f} dB")
            axs[i+1].axis('off')

        plt.suptitle(f"DCT Partial Reconstruction for {os.path.basename(path)}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
